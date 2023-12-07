import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np
import copy


class StateTPPSC(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc, [batch_size, graph_size+1, 2]
    demand: torch.Tensor
    pay: torch.Tensor
    tasks_deadline_time: torch.Tensor
    tasks_start_time: torch.Tensor
    tasks_finish_time: torch.Tensor
    # capacity: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    veh: torch.Tensor  # numver of vehicles
    speed: torch.Tensor
    # State
    used_capacity: torch.Tensor
    capacity: torch.Tensor
    workers_deadline_time: torch.Tensor
    workers_score: torch.Tensor
    workers_done: torch.Tensor
    workers_decision_time: torch.Tensor
    robots_current_destination_location: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    #VEHICLE_CAPACITY = [20., 25., 30., 35., 40.]  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return self.visited_[:, None, :].expand(self.visited_.size(0), 1, -1).type(torch.ByteTensor)
            # return mask_long2bool(self.visited_, n=self.demand.size(-2))

    @property
    def dist(self):  # coords: []
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                veh=self.veh[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateTPPSC, self).__getitem__(key)


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['t_loc']
        demand = input['demand']
        pay = input['t_pay'].squeeze(-1)
        pay.shape
        tasks_start_time=input['t_start']
        tasks_deadline_time=input['t_deadline']
        #workers_start_time= input['w_start']
        workers_deadline_time = input['w_deadline']
        workers_decision_time=input['w_start']
        workers_score = input['w_score']
        _, w_num, _ = input['w_loc'].size()
        cur_coord1 = input['w_loc']

        batch_size, n_loc, _ = loc.size()  # n_loc = graph_size
        return StateTPPSC(
            coords=torch.cat((depot[:, None, :], loc), -2),  # [batch_size, graph_size, 2]]
            demand=torch.cat((torch.zeros(batch_size, 1, device=loc.device), demand), 1),
            pay=pay,  #torch.cat((torch.zeros(batch_size, 1, device=loc.device), pay), 1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            tasks_deadline_time=tasks_deadline_time,
            tasks_start_time=tasks_start_time,
            tasks_finish_time=torch.zeros((batch_size, n_loc), dtype=torch.float, device=loc.device),
            veh=torch.arange(w_num, dtype=torch.int64, device=loc.device)[:, None],
            speed=input['w_speed'],
            workers_score=input['w_score'],
            # prev_a is current node
            prev_a=torch.zeros(batch_size, w_num, dtype=torch.long, device=loc.device),
            cur_coord=cur_coord1,
            used_capacity=demand.new_zeros(batch_size, w_num),
            capacity=input['w_capacity'],
            workers_decision_time=workers_decision_time,
            workers_deadline_time=workers_deadline_time,
            robots_current_destination_location=cur_coord1,
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            workers_done=(
                torch.zeros(
                    batch_size, 1, w_num,
                    dtype=torch.uint8, device=loc.device)
            )
            if visited_dtype == torch.uint8
            else torch.zeros(batch_size, 1, (w_num + 63) // 64, dtype=torch.int64, device=loc.device),
            lengths=torch.zeros(batch_size, w_num, device=loc.device),
            # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):
        assert self.all_finished()
        # coords: [batch_size, graph_size+1, 2]
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, veh):  # [batch_size, num_veh]

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        prev_a = selected  # [batch_size, num_veh]
        batch_size, _ = selected.size()

        current_time = self.workers_decision_time
        # Add the length, coords:[batch_size, graph_size, 2]
        cur_coord = self.coords.gather(  # [batch_size, num_veh, 2]
            1,
            selected[:, :, None].expand(selected.size(0), len(self.veh), self.coords.size(-1))
        )

        b = cur_coord[torch.arange(batch_size), veh]
        # cur_coord=[self.cur_coord.contiguous()]
        cur_coord = torch.stack([self.cur_coord.contiguous()], 0).squeeze(0)

        cur_coord[torch.arange(batch_size), veh] = b

        time = ((cur_coord - self.cur_coord).norm(2, 2) / self.speed[torch.arange(batch_size), veh]).unsqueeze(-1)

        workers_done = self.workers_done
        workers_done.shape

        for i in range (len(veh)):
            if time[i, veh[i]]==0:
                time[i, veh[i]] = 1000
                workers_done[i, 0, veh[i]] = 1

        temp_time=time[torch.arange(batch_size), veh] + current_time[torch.arange(batch_size), veh]
        self.workers_decision_time[torch.arange(batch_size), veh] = temp_time

        aa1=selected[torch.arange(batch_size), veh]-1
        self.tasks_finish_time[torch.arange(batch_size), aa1]=(temp_time).squeeze(-1)

        workers_decision_time = self.workers_decision_time

        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        selected_demand_broad = self.demand[:, :, None].gather(  # [batch_size, num_veh]
            1,
            prev_a[torch.arange(batch_size), veh][:, None, None].expand(prev_a.size(0), len(self.veh),
                                                                        self.demand[:, :, None].size(-1))
        ).squeeze(2)
        selected_demand = torch.zeros_like(selected_demand_broad)
        selected_demand[torch.arange(batch_size), veh] = selected_demand_broad[torch.arange(batch_size), veh].clone()

        used_capacity = self.used_capacity
        used_capacity[torch.arange(batch_size), veh] = (self.used_capacity[torch.arange(batch_size), veh] +
                                                        selected_demand[torch.arange(batch_size), veh]) * (
                                                               prev_a[torch.arange(
                                                                   batch_size), veh] != 0).float()  # [batch_size, num)_veh]

        is_worker_done = ((self.used_capacity[torch.arange(batch_size)] >= self.capacity[torch.arange(batch_size)]).to(
            torch.bool) | (self.workers_deadline_time <= self.workers_decision_time).to(torch.bool).squeeze(-1))[:,
                         None, :]

        workers_done = self.workers_done
        workers_done[is_worker_done] = 1



        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[torch.arange(batch_size), veh][:, None, None].expand_as(
                self.visited_[:, :, 0:1]), 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a[torch.arange(batch_size), veh])

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,  workers_decision_time=workers_decision_time,
            cur_coord=cur_coord, i=self.i + 1, workers_done=workers_done, lengths=lengths
        )


    def all_finished(self):
        batch_size, _, len = self.visited_.size()
        aa = torch.count_nonzero(self.visited_[:, :, 1:].squeeze(1), dim=1).reshape(-1, 1).squeeze(-1)
        bb = torch.count_nonzero(self.workers_done.squeeze(1), dim=1).reshape(-1, 1).squeeze(-1)
        len1 = self.workers_done.size(-1)
        flag = torch.zeros(batch_size)
        aa = aa - (len - 1)
        bb = bb - len1
        valid_index = (aa == 0).nonzero()
        valid_index1 = (bb == 0).nonzero()
        # aa = time[valid_index, veh[valid_index]]
        if torch.isnan(valid_index).all() == False:
            flag[valid_index] = 1
        if torch.isnan(valid_index1).all() == False:
            flag[valid_index1] = 1
        bath_done = torch.count_nonzero(flag)
        if bath_done == batch_size:
            return 1
        else:
            return 0

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    # need to be modified
    def get_mask(self, veh):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        batch_size,_ ,task_size = self.visited_.size()
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, n_loc]
        else:
            visited_loc = self.visited_[:, 1:][:, None, :]  # [batch_size, 1, n_loc]

        # zero = torch.zeros(batch_size, 1, task_size-1, device='cuda')
        # all_pay=torch.cat((zero, self.pay), 1)
        cur_time=self.workers_decision_time[torch.arange(batch_size), veh]
        distance_matrix = (self.cur_coord[:, :, None, :] - self.coords[:, None, 1:, :]).norm(p=2, dim=-1).to(device=visited_loc.device)

        time_matrix = torch.mul(distance_matrix, (1 / self.speed)).to(device=visited_loc.device)  # 时间除以路程=1/速度

        cur_time=(cur_time+time_matrix[torch.arange(batch_size), veh])-self.tasks_start_time.squeeze(-1)
        finnal_time = self.tasks_deadline_time.max().int()
        # cur_time=((cur_time-self.tasks_start_time.squeeze(-1)))
        cur_time64 = torch.tensor(cur_time+1, dtype=torch.int64)
        cur_time64 = torch.where(cur_time64 < 0, 0, cur_time64)
        cur_time64 = torch.where(cur_time64 > int(finnal_time-1), int(finnal_time-1), cur_time64)
        self.pay.shape
        bb3=self.pay.gather(1, (cur_time64[:, None, :]).expand(cur_time.size(0), 1, self.pay.size(-1))).squeeze(1)

        cur_score = self.workers_score[torch.arange(batch_size), veh]
        positive_pay = (0.5*bb3+0.5*cur_score-20*(distance_matrix[torch.arange(batch_size), veh]) <= 0)[:, None, :]

        arrived_before_task=((time_matrix[torch.arange(batch_size), veh]+self.workers_decision_time[torch.arange(batch_size), veh] )
                        <= (self.tasks_start_time.squeeze(-1)))[:, None, :]

        exceeds_task_deadline=((time_matrix[torch.arange(batch_size), veh] + self.workers_decision_time[torch.arange(batch_size), veh])
        >=self.tasks_deadline_time.squeeze(-1))[:, None, :]

        exceeds_time = ((time_matrix[torch.arange(batch_size), veh]+self.workers_decision_time[torch.arange(batch_size), veh] )
                        >= (self.workers_deadline_time[torch.arange(batch_size), veh]) )[:, None, :] \
                       | exceeds_task_deadline | arrived_before_task
        # Nodes that cannot be visited are already visited or too much demand to be served now


        exceeds_cap = (self.demand[self.ids, 1:] + (self.used_capacity[torch.arange(batch_size), veh].unsqueeze(-1))[
            ..., None].expand_as(self.demand[self.ids, 1:]) >
                       (self.capacity[torch.arange(batch_size), veh]).unsqueeze(-1)[..., None].expand_as(
                           self.demand[self.ids, 1:]))

        mask_loc = visited_loc.to(torch.bool) | exceeds_cap | exceeds_time | positive_pay

        # Cannot visit the depot if just visited and still unserved nodes


        mask_depot = torch.tensor(torch.ones((mask_loc.size()[0], 1)).clone().detach(), dtype=torch.bool,
                                  device=mask_loc.device)  # [batch_size, 1]

        # mask_loc[0, :, 1] = True
        aa = mask_loc.to(torch.float)
        aa = torch.count_nonzero(aa[:, :, :].squeeze(1), dim=1).reshape(-1, 1)
        for k in range(batch_size):
            if aa[k] == task_size - 1:
                a = self.prev_a[k, veh[k]]
                if a > 0:
                    mask_loc[k, :, a - 1] = False
                else:
                    mask_depot [k, 0] = False

        # mask_depot = (
        #             (mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size]


    def construct_solutions(self, actions):
        return actions
