import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder , MultiHeadEncoder, MultiHeadEncoderDyn, MultiHeadPosCompat
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import copy
import random


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 obj,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obj = obj
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_totp = problem.NAME == 'totp'
        self.feed_forward_hidden = 512
        self.normalization = normalization

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_totp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1
            # max_veh = 10
            node_dim = 4 # x,y, demand(5 vehicles)
            node_dim1 = 1  # + num_veh  # x,y, demand(5 vehicles)
            # node_veh = 3 * num_veh
            self.FF_wor = nn.Sequential(
                # nn.Linear(3*max_veh, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)

            self.FF_tour = nn.Sequential(
                # nn.Linear(max_veh * self.embedding_dim, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)
            # self.select_embed = nn.Linear(self.embedding_dim * 2, max_veh)

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            self.init_static_embed = nn.Linear(node_dim, embedding_dim)  # node_embedding
            self.init_embed_ret = nn.Linear(2 * embedding_dim, embedding_dim)


        self.init_embed = nn.Linear(node_dim, embedding_dim)  # node_embedding
        self.init_dynamic_embed = nn.Linear(node_dim1, embedding_dim)  # node_embedding

        self.pos_encoder = MultiHeadPosCompat(self.n_heads,
                                self.embedding_dim,
                                self.hidden_dim,
                           ) # for PFEs
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,
                                )
            for _ in range(self.n_encode_layers))) # for NFEs
        self.dyn_encoder = mySequential(*(
                MultiHeadEncoderDyn(self.n_heads,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,
                                )
            for _ in range(self.n_encode_layers) )) # for NFEs

        self.fusion_layer = nn.Linear(256, 128)

        # self.pattern = self.cyclic_position_encoding_pattern(tasks_size+1, embedding_dim)
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


    def positional_encoding(self, n_position, emb_dim, mean_pooling=True):

        angle_rads = self.get_angles(np.arange(n_position)[:, np.newaxis],
                                np.arange(emb_dim)[np.newaxis, :],
                                emb_dim)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pattern = angle_rads[np.newaxis, ...]
        #### ----

        return pattern


    def position_encoding(self, input, embedding_dim):
        t_loc=input['t_loc']
        batch_size, seq_length, _ = t_loc.size()
        # expand for every batch
        position_enc_new = self.positional_encoding(seq_length+1, embedding_dim)
        position_enc_new=torch.tensor(position_enc_new[0],dtype=torch.float32).expand(batch_size, seq_length+1, embedding_dim).to(t_loc.device)
        # return
        return position_enc_new

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        # embeddings: [batch_size, graph_size+1, embed_dim]
        if self.checkpoint_encoder:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(
                input))  # self._init_embed(input): [batch_size, graph_size+1, embed_dim]
        else:
            #embeddings, _ = self.embedder(self._init_embed(input))
            static_feature, dynamic_feature, st_edge_feature = self._init_embed(input)
            pos_embeddings= self.position_encoding(input, self.embedding_dim)

        _log_p, log_p_worker, pi, worker_list, tour, finish_time = self._inner(input, static_feature, dynamic_feature, pos_embeddings, st_edge_feature)  # _log_p: [batch_size, graph_size+1, graph_size+1], pi:[batch_size, graph_size+1]

        cost, mask = self.problem.get_costs(input, pi, worker_list, tour, finish_time)  # mask is None, cost:[batch_size]

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll, ll_worker = self._calc_log_likelihood(_log_p, log_p_worker, pi, mask, worker_list)  # [batch_size], 所有被选点对应的log_pro的和
        if return_pi:
            return cost, ll, ll_worker, pi

        return cost, ll, ll_worker

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, _log_p_worker, a, mask, worker_list):  # a is pi
        
        if np.ndim(a) == 1:
            a = a.unsqueeze(0)
        log_p = _log_p.gather(2, torch.tensor(a).unsqueeze(-1)).squeeze(-1)
        log_p_worker = _log_p_worker.gather(2, torch.tensor(worker_list).cuda().unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
            log_p_worker[mask] = 0
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        assert (log_p_worker > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1), log_p_worker.sum(1)  # [batch_size]

    def _init_embed(self, input):

        if self.is_totp:
            batch_size, time, _, _ = input['t_pay'].size()
            zero = torch.zeros(batch_size, time, 1, device='cuda')[:, :, None, :]
            zero1 = torch.zeros(batch_size, 1, device='cuda')[:, :, None]

            s_time = torch.cat((zero1, input['t_start'][:, :, :]), 1)
            d_time = torch.cat((zero1, input['t_deadline'][:, :, :]), 1)
            all_loc = torch.cat((input['depot'][:, None, :], input['t_loc']), 1)
            t_pay = torch.cat((zero, input['t_pay']), 2)
            t_start = input['t_start'][:, :, :] / input['t_start'][:, :, :].max()
            t_deadline = input['t_deadline'][:, :, :] / input['t_deadline'][:, :, :].max()

            distance_matrix = t_pay[:, 1, :, :] - 10 * (all_loc[:, :, None, :] - all_loc[:, None, :, :]).norm(p=2,
                                                                                                              dim=-1).to(
                device=t_pay.device)
            distance_matrix = distance_matrix / distance_matrix.max()
            distance_matrix[distance_matrix < 0] = -1000
            #time constraint
            time_cond1 = (s_time[:, :, None, :] - s_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            time_cond2 = (d_time[:, :, None, :] - s_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            time_cond3=(d_time[:, :, None, :] - d_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            time_cond4 = (s_time[:, :, None, :] - d_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            # temporal edge
            time_edge1 = torch.mul(time_cond1, time_cond2)
            time_edge1[time_edge1 <= 0] = 1000
            time_edge1[time_edge1 != 1000] = -1000

            time_edge2 = torch.mul(time_cond1, time_cond4)
            time_edge2[time_edge2 <= 0] = 1000
            time_edge2[time_edge2 != 1000] = -1000

            t_edge = time_edge1 + time_edge2
            t_edge[t_edge >= 0] = 1
            t_edge[:, :, 0] = 1
            t_edge[:, 0, :] = 1
            #spatio-temporal edge
            st_edge = t_edge + distance_matrix
            st_edge[st_edge < 0] = 0
            st_edge = st_edge / st_edge.max()
            st_edge[st_edge <= 0] = 0

            static_feature = torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_static_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        input['t_loc'],  # [batch_size, graph_size, 2]
                        # t_pay,  # [batch_size, graph_size, 1]
                        t_start,  # [batch_size, graph_size, 1]
                        t_deadline  # [batch_size, graph_size, 1]
                    ), -1))
                ),
                1
            )
            static_feature.shape

            t_pay = t_pay / t_pay.max()
            dynamic_feature = self.init_dynamic_embed(t_pay).permute(1, 0, 2, 3)

            return static_feature, dynamic_feature, st_edge

    def select_worker(self, input, state, sequences, embeddings, obj, wor_list, tour, t, T):
        current_node = state.get_current_node()  # [batch_size]
        tour_dis = state.lengths  # [batch_size, num_wor]

        batch_size, graph_size, embed_dim = embeddings.size()
        _, num_wor = current_node.size()

        SPEED = (state.speed[:, 0:num_wor, :])

        exceeds_deadline = (state.workers_deadline_time <= state.workers_decision_time).to(torch.bool).squeeze(-1)
        working_workers =  (state.capacity <= state.used_capacity).to(torch.bool) | exceeds_deadline | state.workers_done.squeeze(1)

        if sequences:

            tour1 = tour.transpose(0, 1).transpose(1, 2)
            tour1 = torch.tensor(tour1, dtype=torch.int64).cuda()

            _, all_workers, _ =tour1.size()

            embeddings = embeddings.unsqueeze(-3)
            embeddings = embeddings.expand(batch_size, num_wor, graph_size, embed_dim)

            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                2,
                (tour1[:, :, :].clone())[..., None].contiguous()  # [batch_size, num_wor, tour_len]
                    .expand(batch_size, num_wor, tour1.size(-1), embed_dim)
            )

            mean_tour = (torch.max(tour_con[:, :, :, :], dim=2)[0]).view([batch_size, -1])
            current_loc = (state.cur_coord[:, 0:num_wor, :])
            wor_context = torch.cat(((tour_dis[:, :].unsqueeze(-1) / SPEED[:, :]), current_loc[:, :]), -1).view(
                [batch_size, -1])
        else:
            current_loc = (state.cur_coord[:, 0:num_wor, :].clone())
            mean_tour = torch.zeros([batch_size, num_wor * embed_dim]).float().cuda()
            wor_context = torch.cat(((tour_dis[:, :].unsqueeze(-1) / SPEED[:, :]), current_loc[:, :]), -1).view(
                [batch_size, -1])

        output_context=torch.nn.Linear(3*num_wor, self.embedding_dim).cuda()
        wor_context = self.FF_wor(output_context(wor_context))

        output_tour = torch.nn.Linear(num_wor*self.embedding_dim, self.embedding_dim).cuda()
        tour_context = self.FF_tour(output_tour(mean_tour))

        context = torch.cat((wor_context, tour_context), -1).view(batch_size, self.embedding_dim * 2)
        select_embed = torch.nn.Linear(self.embedding_dim * 2, num_wor).to(device=SPEED.device)
        log=select_embed(context/10)
        log = torch.tanh(log) * self.tanh_clipping

        aa = torch.count_nonzero(working_workers, dim=1).reshape(-1, 1).squeeze(-1)-num_wor
        valid_index = (aa == 0).nonzero()
        if torch.isnan(valid_index).all() == False:
            working_workers[valid_index, 0] = False

        log[working_workers] = -math.inf
        log_wor = F.log_softmax(log, dim=1)

        if self.decode_type == "greedy":
            wor = torch.max(F.softmax(log), dim=1)[1]
        elif self.decode_type == "sampling":
            log = log.exp()
            wor = log.multinomial(1).squeeze(-1)

        return wor, log_wor

    def _inner(self, input, s_feature, d_feature, pos_embeddings,  st_edge_feature):
        #存储 结果
        outputs = []
        outputs_wor = []
        sequences = []
        tour = []
        prev_state=[]

        #全部任务到达的最晚时间
        arrive_time_t=input['t_start'].squeeze(-1)
        arrive_time_w=input['w_start'].squeeze(-1)

        final_arrive_time=max(arrive_time_t.max(),arrive_time_w.max()).short()
        duration=(input['t_deadline'].squeeze(-1)-arrive_time_t)[0,0].int()

        arrived_time_tasks=[]
        arrived_time_workers=[]
        arrived_time_tasks.append(torch.bincount(arrive_time_t[0].short()))

        #补齐任务和工人到达时间，  同步
        if len(arrived_time_tasks[0]) < final_arrive_time+1:
            zeros = torch.zeros(1, final_arrive_time+1-len(arrived_time_tasks[0])).cuda()
            arrived_time_tasks[0] = torch.cat((arrived_time_tasks[0], zeros[0, :].short()), -1)
        arrived_time_workers.append(torch.bincount(arrive_time_w[0].short()))
        if len(arrived_time_workers[0])<final_arrive_time+1:
            zeros = torch.zeros(1, final_arrive_time+1-len(arrived_time_workers[0])).cuda()
            arrived_time_workers[0] = torch.cat((arrived_time_workers[0], zeros[0, :].short()), -1)

        # 初始化
        wor_list = []
        all_arrived_tasks=0
        all_arrived_workers=0
        all_deadline_tasks=0
        all_deadline_workers = 0

        prev_tasks_num = 0
        prev_workers_num = 0

        # Perform decoding steps
        i = 0
        # Select the workers and tasks
        while not (self.shrink_size is None and i>=final_arrive_time+1):

            for t in range(final_arrive_time+2):
                if t <= final_arrive_time:
                    i += 1
                    all_arrived_tasks = all_arrived_tasks + arrived_time_tasks[0][t]
                    all_arrived_workers = all_arrived_workers + arrived_time_workers[0][t]
                    new_arrived_tasks = arrived_time_tasks[0][t]#all_arrived_tasks - prev_tasks_num
                    new_arrived_workers = arrived_time_workers[0][t]#all_arrived_workers - prev_workers_num
                    if t > duration:
                        all_deadline_tasks = all_deadline_tasks + arrived_time_tasks[0][t - duration]
                        all_deadline_workers = all_deadline_workers + arrived_time_workers[0][t - duration]
                    if (new_arrived_tasks != 0 or new_arrived_workers!= 0) and (all_arrived_tasks!=0 and all_arrived_workers!=0):
                        if t <= duration:
                            cur_edge = st_edge_feature[:, 0:all_arrived_tasks + 1, 0:all_arrived_tasks + 1]
                            cur_pos_embeddings = pos_embeddings[:, 0:all_arrived_tasks + 1, :]
                            cur_pos_embeddings = self.pos_encoder(cur_pos_embeddings, cur_edge)

                            cur_s_features=s_feature[:, 0:all_arrived_tasks + 1, :]
                            current_emb_ps = self.encoder(cur_s_features,cur_pos_embeddings, cur_edge)[0]

                            cur_d_features= d_feature[ 0:t+1, :, 0:all_arrived_tasks + 1, :]
                            cur_d_embeddings = (self.dyn_encoder(cur_d_features, cur_edge)[0])[-1]

                            fusion_current_emb = torch.cat((current_emb_ps, cur_d_embeddings), dim=-1)
                            all_embeddings = self.fusion_layer(fusion_current_emb)

                            fixed = self._precompute(all_embeddings)

                        else:
                            cur_edge = st_edge_feature[:, 1+all_deadline_tasks:all_arrived_tasks + 1, 1+all_deadline_tasks:all_arrived_tasks + 1]
                            cur_pos_embeddings = pos_embeddings[:, 1+all_deadline_tasks:all_arrived_tasks+1, :]

                            cur_pos_embeddings = self.pos_encoder(cur_pos_embeddings, cur_edge)

                            cur_s_features=s_feature[:, 1+all_deadline_tasks:all_arrived_tasks+1, :]
                            current_emb = self.encoder(cur_s_features, cur_pos_embeddings, cur_edge)[0]

                            cur_d_features= d_feature[ t-6:t+1, :, 1+all_deadline_tasks:all_arrived_tasks + 1, :]
                            cur_d_embeddings = (self.dyn_encoder(cur_d_features, cur_edge)[0])[-1]

                            fusion_current_emb = torch.cat((current_emb, cur_d_embeddings), dim=-1)
                            new_embeddings=self.fusion_layer(fusion_current_emb)
                            all_embeddings=torch.cat((all_embeddings, new_embeddings[:,-new_arrived_tasks:, :]), 1)

                            fixed = self._precompute(all_embeddings)

                        state = self.problem.make_state(input, t, duration, all_arrived_tasks, new_arrived_tasks, all_arrived_workers, new_arrived_workers,  prev_state)
                        current_node = state.get_current_node()
                        batch_size, num_worker = current_node.size()

                        while not (self.shrink_size is None and not (state.all_finished() == 0)):
                            if len(outputs_wor) != 0 and all_arrived_workers-prev_workers_num!=0 :
                                zeros = torch.zeros(tour.size(0), tour.size(1), all_arrived_workers - prev_workers_num).to(
                                    device=current_node.device)
                                tour=torch.cat((tour[:, :, :], zeros), -1)

                            wor, log_p_wor = self.select_worker(input, state, sequences, all_embeddings, self.obj, wor_list, tour, t, final_arrive_time)  # [batch_size, 1]
                            wor_list.append(wor.tolist())
                            if self.shrink_size is not None:
                                unfinished = torch.nonzero(state.get_finished() == 0)
                                if len(unfinished) == 0:
                                    break
                                unfinished = unfinished[:, 0]
                                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                                # (otherwise batch norm will not work well and it is inefficient anyway)
                                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                                    # Filter states
                                    state = state[unfinished]
                                    fixed = fixed[unfinished]

                                # Only the required ones goes here, so we should
                                #  We need a variable that track which all tasks are available
                            log_p, mask = self._get_log_p(fixed, state, wor, t, final_arrive_time)

                            # Select the indices of the next nodes in the sequences, result (batch_size) long
                            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :], state,
                                                         wor)  # Squeeze out steps dimension

                            state = state.update(selected, wor, t, final_arrive_time)

                            # Now make log_p, selected desired output size by 'unshrinking'
                            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                                log_p_, selected_ = log_p, selected
                                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                                selected = selected_.new_zeros(batch_size)

                                log_p[state.ids[:, 0]] = log_p_
                                selected[state.ids[:, 0]] = selected_

                            if len(outputs) == 0:
                                outputs=log_p[:, 0, :][:, None, :]
                            else:
                                if prev_tasks_num != all_arrived_tasks :
                                    zeros = torch.zeros(batch_size, outputs.size(1), all_arrived_tasks - prev_tasks_num).to(
                                        device=current_node.device)
                                    zeros = zeros - math.inf
                                    outputs=torch.cat((outputs, zeros), -1)
                                    outputs = torch.cat((outputs, log_p[:, 0, :][:, None, :]), 1)
                                else:
                                    outputs=torch.cat((outputs, log_p[:, 0, :][:, None, :]), 1)

                            if len(outputs_wor) == 0:
                                outputs_wor=log_p_wor[:, None, :]
                            else:
                                if prev_workers_num != all_arrived_workers: #note 过期的工人 新到达的工人 进行裁减和拼接
                                    zeros_new = torch.zeros(batch_size, outputs_wor.size(1), all_arrived_workers- prev_workers_num).to(
                                        device=current_node.device)
                                    if zeros_new.numel() > 0:
                                        zeros_new = zeros_new - math.inf
                                    outputs_wor=torch.cat((outputs_wor, zeros_new), -1)
                                    outputs_wor=torch.cat((outputs_wor, log_p_wor[:, None, :]),1)
                                else:
                                    outputs_wor=torch.cat((outputs_wor, log_p_wor[:, None, :]),1)
                            outputs_wor.shape

                            sequences.append(selected[torch.arange(batch_size), wor])

                            if len(tour) == 0:
                                tour=selected[:, :][None, :]
                            else:
                                tour=torch.cat((tour, selected[:, :][None, :]),0)

                            prev_tasks_num = all_arrived_tasks
                            prev_workers_num = all_arrived_workers
                        prev_state = state
                else:
                    i += 1
                    all_arrived_tasks= all_arrived_tasks
                    all_arrived_workers = all_arrived_workers
                    new_arrived_tasks = all_arrived_tasks - prev_tasks_num
                    new_arrived_workers = all_arrived_workers - prev_workers_num
                    if all_arrived_tasks> 0 and all_arrived_workers > 0:
                        state = self.problem.make_state(input, t, duration, all_arrived_tasks, new_arrived_tasks, all_arrived_workers, new_arrived_workers, prev_state)
                        current_node = state.get_current_node()
                        batch_size, num_worker = current_node.size()
                        while not (self.shrink_size is None and not (state.all_finished() == 0 )):
                            wor, log_p_wor = self.select_worker(input, state, sequences, all_embeddings, self.obj, wor_list, tour, t, final_arrive_time)
                            wor_list.append(wor.tolist())

                            if self.shrink_size is not None:
                                unfinished = torch.nonzero(state.get_finished() == 0)
                                if len(unfinished) == 0:
                                    break
                                unfinished = unfinished[:, 0]
                                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                                # (otherwise batch norm will not work well and it is inefficient anyway)
                                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                                    # Filter states
                                    state = state[unfinished]
                                    fixed = fixed[unfinished]

                                # Only the required ones goes here, so we should
                                #  We need a variable that track which all tasks are available
                            log_p, mask = self._get_log_p(fixed, state, wor, t, final_arrive_time)

                            # Select the indices of the next nodes in the sequences, result (batch_size) long
                            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :], state, wor)  # Squeeze out steps dimension

                            state = state.update(selected, wor, t, final_arrive_time)

                            # Now make log_p, selected desired output size by 'unshrinking'
                            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                                log_p_, selected_ = log_p, selected
                                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                                selected = selected_.new_zeros(batch_size)

                                log_p[state.ids[:, 0]] = log_p_
                                selected[state.ids[:, 0]] = selected_

                            # Collect output of step
                            outputs=torch.cat((outputs, log_p[:, 0, :][ :, None, :]),1)

                            outputs_wor=torch.cat((outputs_wor, log_p_wor[:, None, :]),1)

                            sequences.append(selected[torch.arange(batch_size), wor])

                            tour = torch.cat((tour, selected[:, :][None, :]), 0)

        worker_list = torch.tensor(wor_list).transpose(0, 1)

        finish_time=state.tasks_finish_time-state.tasks_start_time.squeeze(-1)

        return outputs,  outputs_wor, torch.stack(sequences, -1).squeeze(-2), worker_list, tour, finish_time

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        # print('input', input)

        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi, veh_list, tour: self.problem.get_costs(input[0], self.obj, pi, veh_list, tour),  # Don't need embeddings as input to get_costs
            (input, self.encoder(self._init_embed(input), self.pos_encoder(self.position_encoding(input, self.embedding_dim)))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask, state, wor):  # probs, mask: [batch_size, graph_size]
        assert (probs == probs).all(), "Probs should not contain any nans"

        selected = (state.get_current_node()).clone()
        batch_size, _ = (state.get_current_node()).size()

        if self.decode_type == "greedy":
            _, selected[torch.arange(batch_size), wor] = probs.max(1)
            assert not mask.gather(-1,selected[torch.arange(batch_size), wor].unsqueeze(-1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected[torch.arange(batch_size), wor] = probs.multinomial(1).squeeze(
                1)  # [batch_size]

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(-1, selected[torch.arange(batch_size), wor].unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected[torch.arange(batch_size), wor] = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        # embeddings: [batch_size, graph_size+1, embed_dim]

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # [batch_size, embed_dim]
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]  # linear(graph_embed)

        # The projection of the node embeddings for the attention is calculated once up front
        # glimpse_key_fixed size is torch.Size([batch_size, 1, graph_size+1, embed_dim])
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3,
                                                                          dim=-1)  # split tensor to three parts in dimension 1

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (  # make multihead
            self._make_heads(glimpse_key_fixed, num_steps),  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            self._make_heads(glimpse_val_fixed, num_steps),  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            logit_key_fixed.contiguous()  # [batch_size, 1, graph_size+1, embed_dim]
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, wor, t, T, normalize=True):
        # fixed: node_embeddings(embeddings), context_node_project(graph_embed), glimpse_key, glimpse_val, logits_key
        # Compute query = context node embedding, 相同维度数字相加
        # fixed.context_node_projected (graph_embedding): (batch_size, 1, embed_dim), query: [batch_size, num_veh, embed_dim]
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings,
                                                                          state, wor))  # after project: [batch_size, 1, embed_dim]

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask(wor, t, T) # [batch_size, 1, graph_size]

        # Compute logits (unnormalized log_p)  log_p:[batch_size, num_veh, graph_size], glimpse:[batch_size, num_veh, embed_dim]
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, wor)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, wor, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        current_node = (state.get_current_node()).clone()
        batch_size, num_wor = current_node.size()
        num_steps = 1

        # Embedding of previous node + remaining capacity

        w_c = [torch.tensor(state.capacity)[k, wor[k]] for k in range(wor.size(-1))]
        w_c = torch.stack(w_c, -1)[None, :]

        return torch.cat(  # [batch_size, num_veh, embed_dim+1]
            (
                torch.gather(
                    embeddings,  # [batch_size, graph_size, embed_dim]
                    1,
                    (current_node[torch.arange(batch_size), wor]).contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_step, embed_dim]
                (w_c - state.used_capacity[torch.arange(batch_size), wor]).transpose(0, 1).unsqueeze(-1)
            ),
            -1
        )

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, wor):
        batch_size, num_step, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads  # query and K both have key_size

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_step, 1, key_size)
        glimpse_Q = query.view(batch_size, num_step, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_step, 1, graph_size)
        # glimpse_K (n_heads, batch_size, 1, graph_size, key_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:  # True
            assert self.mask_logits, "Cannot mask inner without masking logits"  # True
            # mask: # [batch_size, num_veh, graph_size]
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf  # nask visited nodes and nodes cannot be visited

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_step, 1, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_step, 1, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_step, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        final_Q = glimpse
        # logits_K, (batch_size, 1, graph_size, embed_dim)
        # Batch matrix multiplication to compute logits (batch_size, num_step, graph_size)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:  # 10
            # print*(F.tanh(logits))
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:  # True
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)  # glimpse[batch_size, num_veh, embed_dim]

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=1):  # v: [batch_size, 1, graph_size+1, embed_dim]
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
        )

