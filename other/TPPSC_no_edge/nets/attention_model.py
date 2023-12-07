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
                 tasks_size,
                 workers_size,
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
        self.is_tppsc = problem.NAME == 'tppsc'
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
        if self.is_tppsc:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim +  1
            num_workers = workers_size
            node_dim = 4 #+ workers_size  # x,y, demand(5 vehicles)
            node_dim1 = 1  # + num_veh  # x,y, demand(5 vehicles)
            # node_worker = 3 * num_workers
            self.FF_veh = nn.Sequential(
                nn.Linear(6*num_workers, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)

            self.FF_tour = nn.Sequential(
                nn.Linear(num_workers * self.embedding_dim, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)
            self.select_embed = nn.Linear(self.embedding_dim * 2, num_workers)

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)

        self.init_static_embed = nn.Linear(node_dim, embedding_dim)  # node_embedding
        self.init_dynamic_embed = nn.Linear(node_dim1, embedding_dim)  # node_embedding

        self.pos_encoder = MultiHeadPosCompat(self.n_heads,
                                self.embedding_dim,
                                self.hidden_dim,
                           ) # for PFEs
        self.static_encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,
                                )
            for _ in range(self.n_encode_layers) )) # for NFEs
        self.dyn_encoder = mySequential(*(
                MultiHeadEncoderDyn(self.n_heads,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,
                                )
            for _ in range(self.n_encode_layers) )) # for NFEs

        self.fusion_layer = nn.Linear(256, 128)
        # for _ in range(self.n_encode_layers)
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
            s_embed, d_embed, st_edge = self._init_embed(input)
            d_embed.shape
            h_pos= self.position_encoding(input, self.embedding_dim)
            h_pos.shape
            pos_em = self.pos_encoder(h_pos)
            # embeddings,aa = self.encoder(s_embed, d_embed, pos_em)[0]#  self.encoder(self._init_embed(input), self.pos_encoder(self.position_encoding(input, self.embedding_dim)))[0]
            static_embeddings= self.static_encoder(s_embed, pos_em)[0]
            dyn_embeddings =(self.dyn_encoder(d_embed, pos_em)[0])[-1]
            dyn_embeddings.shape
            fusion_embeddings = torch.cat((static_embeddings, dyn_embeddings), dim=-1)
            embeddings = self.fusion_layer(fusion_embeddings)
            embeddings.shape

        embeddings.shape
        _log_p, log_p_veh, pi, veh_list, tour, time = self._inner(input, embeddings)  # _log_p: [batch_size, graph_size+1, graph_size+1], pi:[batch_size, graph_size+1]
        _log_p.shape#2-19-51
        pi.shape#2-19
        cost, mask = self.problem.get_costs(input, self.obj, veh_list, tour, time)  # mask is None, cost:[batch_size]

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll, ll_veh = self._calc_log_likelihood(_log_p, log_p_veh, pi, mask, veh_list)  # [batch_size], 所有被选点对应的log_pro的和
        if return_pi:
            return cost, ll, ll_veh, pi

        return cost, ll, ll_veh

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

    def _calc_log_likelihood(self, _log_p, _log_p_veh, a, mask, veh_list):  # a is pi

        if np.ndim(a) == 1:
            a = a.unsqueeze(0)
        log_p = _log_p.gather(2, torch.tensor(a).unsqueeze(-1)).squeeze(-1)
        log_p_veh = _log_p_veh.gather(2, torch.tensor(veh_list).to(device=log_p.device).unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
            log_p_veh[mask] = 0
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        assert (log_p_veh > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1), log_p_veh.sum(1)  # [batch_size]

    def _init_embed(self, input):

        if self.is_tppsc:
            a = input['t_pay']
            a.shape
            batch_size, time, _, _ = input['t_pay'].size()
            zero = torch.zeros(batch_size, time, 1, device=input['t_pay'].device)[:, :, None, :]
            zero1 = torch.zeros(batch_size, 1, device=input['t_pay'].device)[:, :, None]
            zero1.shape
            # t_loc=input['t_loc']
            # a=input['depot']
            s_time = input['t_start'][:, :, :]
            s_time.shape
            d_time = input['t_deadline'][:, :, :]
            s_time = torch.cat((zero1, input['t_start'][:, :, :]), 1)
            d_time = torch.cat((zero1, input['t_deadline'][:, :, :]), 1)

            all_loc = torch.cat((input['depot'][:, None, :], input['t_loc']), 1)
            all_loc.shape
            # zero = torch.zeros(batch_size, time, 1, device='cuda')[:,:, None, : ]
            t_pay = torch.cat((zero, input['t_pay']), 2)

            t_start = input['t_start'][:, :, :] / input['t_start'][:, :, :].max()
            t_deadline = input['t_deadline'][:, :, :] / input['t_deadline'][:, :, :].max()

            distance_matrix = t_pay[:, 1, :, :] - 10 * (all_loc[:, :, None, :] - all_loc[:, None, :, :]).norm(p=2,
                                                                                                              dim=-1).to(
                device=t_pay.device)
            distance_matrix.shape
            distance_matrix = distance_matrix / distance_matrix.max()
            distance_matrix[distance_matrix < 0] = -1000
            # edg_s[:, 0, :] = 0.001
            # edg_s[:, :, 0] = 0.001

            time_cond1 = (s_time[:, :, None, :] - s_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            time_cond1.shape
            time_cond2 = (d_time[:, :, None, :] - s_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            time_cond2.shape
            # time_cond3=(d_time[:, :, None, :] - d_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            # time_cond3.shape
            time_cond4 = (s_time[:, :, None, :] - d_time[:, None, :, :]).to(device=t_pay.device).squeeze(-1)
            time_cond4.shape

            time_edge1 = torch.mul(time_cond1, time_cond2)
            time_edge1.shape
            time_edge1[time_edge1 <= 0] = 1000
            time_edge1[time_edge1 != 1000] = -1000

            time_edge2 = torch.mul(time_cond1, time_cond4)
            time_edge2.shape
            time_edge2[time_edge2 <= 0] = 1000
            time_edge2[time_edge2 != 1000] = -1000
            t_edge = time_edge1 + time_edge2
            t_edge[t_edge >= 0] = 1
            t_edge[:, :, 0] = 1
            t_edge[:, 0, :] = 1
            time_matrix = torch.abs(time_cond1)
            t_edge.shape
            t_edg = t_edge * time_matrix
            t_edge = 1 - t_edge / t_edge.max()
            edge = t_edge + distance_matrix
            edge[edge < 0] = 0
            edge = edge / edge.max()
            edge[edge <= 0] = 0
            # distance_matrix = ~distance_matrix
            # time_matrix = torch.mul(distance_matrix, (1 / self.speed)).to(device=t_pay.device)  # 时间除以路程=1/速度

            # demand=(input['demand']/input['demand'].max()).unsqueeze(-1)

            # demand=input['demand'].unsqueeze(-1)

            # demand = torch.tensor([(input['demand'] / input['w_capacity'][0:1, veh]).tolist() for veh in
            #                        range(input['w_capacity'].size(-1))]).transpose(0, 1).transpose(1, 2).to(device=t_pay.device)
            # demand.shape

            aa = torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_static_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        input['t_loc'],  # [batch_size, graph_size, 2]
                        # demand,  # [batch_size, graph_size, num_veh]
                        # t_pay,  # [batch_size, graph_size, 1]
                        t_start,  # [batch_size, graph_size, 1]
                        t_deadline  # [batch_size, graph_size, 1]
                    ), -1))
                ),
                1
            )
            aa.shape

            t_pay = t_pay / t_pay.max()
            aa1 = self.init_dynamic_embed(t_pay).permute(1, 0, 2, 3)
            aa1.shape
            # aa2=relation
            return aa, aa1, edge

    def select_veh(self, input, state, sequences, embeddings, obj, veh_list, tour):
        current_node = state.get_current_node()  # [batch_size]
        tour_dis = state.lengths  # [batch_size, num_veh]

        batch_size, graph_size, embed_dim = embeddings.size()
        _, num_veh = current_node.size()

        # SPEED = (state.speed[:, 0:num_veh, :]).transpose(0, 1)
        SPEED = (state.speed[:, 0:num_veh, :])
        workers_score = (state.workers_score[:, 0:num_veh, :].clone())
        worker_decision_time = (state.workers_decision_time[:, 0:num_veh, :].clone())
        worker_deadline_time = (state.workers_deadline_time[:, 0:num_veh, :].clone())
        w_start = worker_decision_time[:, :, :] / worker_decision_time[:, :, :].max()
        w_deadline = worker_deadline_time[:, :, :] / worker_deadline_time[:, :, :].max()
        w_score = workers_score[:, :, :] / workers_score[:, :, :].max()


        exceeds_time = (state.workers_deadline_time <= state.workers_decision_time).to(torch.bool).squeeze(-1)
        working_workers =  (state.capacity <= state.used_capacity).to(torch.bool) | exceeds_time

        if sequences:
            tour1 = tour.transpose(0, 1).transpose(1, 2)
            embeddings = embeddings.unsqueeze(-3)
            embeddings = embeddings.expand(batch_size, num_veh, graph_size, embed_dim)

            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                2,
                (tour1.clone())[..., None].contiguous()  # [batch_size, num_veh, tour_len]
                    .expand(batch_size, num_veh, tour1.size(-1), embed_dim)
            )

            mean_tour = (torch.max(tour_con[:, :, :, :], dim=2)[0]).view([batch_size, -1])
            current_loc = (state.cur_coord[:, 0:num_veh, :])

            veh_context = torch.cat(((tour_dis[:, :].unsqueeze(-1) / SPEED[:, :]), current_loc[:, :],w_score[:, :],w_start[:, :],w_deadline[:, :]), -1).view(
                [batch_size, -1])
        else:

            current_loc = (state.cur_coord[:, 0:num_veh, :].clone())
            current_loc.shape
            aa=current_loc[:, :]
            aa.shape
            workers_score.shape
            mean_tour = torch.zeros([batch_size, num_veh * embed_dim]).float().to(device=current_loc.device)
            veh_context=torch.cat(((tour_dis[:, :].unsqueeze(-1) / SPEED[:, :]), current_loc[:, :],w_score[:, :],w_start[:, :],w_deadline[:, :]), -1).view([batch_size, -1])
        veh_context.shape
        veh_context = self.FF_veh(veh_context)
        tour_context = self.FF_tour(mean_tour)
        context = torch.cat((veh_context, tour_context), -1).view(batch_size, self.embedding_dim * 2)

        log = self.select_embed(context)
        log = torch.tanh(log) * self.tanh_clipping
        log.shape
        aa = torch.count_nonzero(working_workers[:, :].squeeze(1), dim=1).reshape(-1, 1)
        for i in range(batch_size):
            if aa[i] == num_veh:
                working_workers[i, :] = False

        log[working_workers] = -math.inf
        log_veh = F.log_softmax(log, dim=1)
        if self.decode_type == "greedy":
            veh = torch.max(F.softmax(log, dim=1), dim=1)[1]
        elif self.decode_type == "sampling":
            log = log.exp()
            veh = log.multinomial(1).squeeze(-1)

        return veh, log_veh

    def _inner(self, input, embeddings):
        # input: [batch_size, graph_size, node_dim], node_dim=2, location
        # embeddings: [batch_size, graph_size+1, embed_dim]
        state = self.problem.make_state(input)
        current_node = state.get_current_node()
        batch_size, num_veh = current_node.size()

        outputs = []
        outputs_veh = []
        sequences = []
        tour = []
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(
            embeddings)  # embeddings, context_node_project(graph_embed), glimpse_key, glimpse_val, logits_key

        # Perform decoding steps
        i = 0
        #veh = torch.LongTensor(batch_size).zero_()
        veh_list = []
        while not (self.shrink_size is None and not (state.all_finished() == 0 )):
            veh, log_p_veh = self.select_veh(input, state, sequences, embeddings, self.obj, veh_list, tour)  # [batch_size, 1]
            veh_list.append(veh.tolist())
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
            log_p, mask = self._get_log_p(fixed, state, veh)  # log_p: [batch_size, num_step, graph_size], mask:[batch_size, num_step, graph_size]

            # Select the indices of the next nodes in the sequences, result (batch_size) long

            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :], state, veh, sequences)  # Squeeze out steps dimension

            state = state.update(selected, veh)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
            # Collect output of step
            outputs.append(log_p[:, 0, :])#2-1-51
            outputs_veh.append(log_p_veh)#2-1-5 2-2-5 2-3-5
            # aa=torch.stack(outputs_veh, 1)
            # aa.shape
            sequences.append(selected[torch.arange(batch_size), veh])

            if len(tour) == 0:
                tour=selected[:, :][None, :]
            else:
                tour = torch.cat((tour, selected[:, :][None, :]), 0)

            i += 1
        veh_list = torch.tensor(veh_list).transpose(0, 1)

        finish_time=state.tasks_finish_time-state.tasks_start_time.squeeze(-1)
        # output:[batch_size, solu_len, graph_size+1], sequences: [batch_size, tour_len]
        return torch.stack(outputs, 1), torch.stack(outputs_veh, 1), torch.stack(sequences, -1).squeeze(-2), veh_list, \
               tour, finish_time

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        # print('input', input)
        sta, dyn, edge = self._init_embed(input)
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi, veh_list, tour, time: self.problem.get_costs(input[0], pi, veh_list, tour, time),  # Don't need embeddings as input to get_costs
            # aa1,aa2,aa3=self._init_embed(input)
            (input, self.fusion_layer( torch.cat( ((self.dyn_encoder(dyn,edge)[0])[-1],
                self.static_encoder( sta, self.pos_encoder(self.position_encoding(input, self.embedding_dim),edge),edge)[0]),-1 ))),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask, state, veh, sequences):  # probs, mask: [batch_size, graph_size]
        assert (probs == probs).all(), "Probs should not contain any nans"

        selected = (state.get_current_node()).clone()
        batch_size, _ = (state.get_current_node()).size()

        if self.decode_type == "greedy":
            _, selected[torch.arange(batch_size), veh] = probs.max(1)
            assert not mask.gather(-1,selected[torch.arange(batch_size), veh].unsqueeze(-1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            aa=probs.multinomial(1).squeeze(
                1)  # [batch_size]
            selected[torch.arange(batch_size), veh] = aa  # [batch_size]

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(-1, selected[torch.arange(batch_size), veh].unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected[torch.arange(batch_size), veh] = probs.multinomial(1).squeeze(1)

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

    def _get_log_p(self, fixed, state, veh, normalize=True):
        # fixed: node_embeddings(embeddings), context_node_project(graph_embed), glimpse_key, glimpse_val, logits_key
        # Compute query = context node embedding, 相同维度数字相加
        # fixed.context_node_projected (graph_embedding): (batch_size, 1, embed_dim), query: [batch_size, num_veh, embed_dim]
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings,
                                                                          state, veh))  # after project: [batch_size, 1, embed_dim]

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask(veh)  # [batch_size, 1, graph_size]

        # Compute logits (unnormalized log_p)  log_p:[batch_size, num_veh, graph_size], glimpse:[batch_size, num_veh, embed_dim]
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, veh)

        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, veh, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        current_node = (state.get_current_node()).clone()
        batch_size, num_veh = current_node.size()
        num_steps = 1

        # Embedding of previous node + remaining capacity

        w_c = [torch.tensor(state.capacity)[k, veh[k]] for k in range(veh.size(-1))]
        w_c = torch.stack(w_c, -1)[None, :]
        aa=current_node[torch.arange(batch_size), veh]
        return torch.cat(  # [batch_size, num_veh, embed_dim+1]
            (
                torch.gather(
                    embeddings,  # [batch_size, graph_size, embed_dim]
                    1,
                    (current_node[torch.arange(batch_size), veh]).contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_step, embed_dim]
                (w_c - state.used_capacity[torch.arange(batch_size), veh]).transpose(0, 1).unsqueeze(-1)
            ),
            -1
        )

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, veh):
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

