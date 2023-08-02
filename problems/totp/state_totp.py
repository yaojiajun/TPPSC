import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np
import copy


class StateTOTP(NamedTuple):
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
    wor: torch.Tensor  # numver of workers
    speed: torch.Tensor
    # riato: torch.Tensor
    # State
    used_capacity: torch.Tensor

    capacity: torch.Tensor
    workers_deadline_time: torch.Tensor
    workers_done: torch.Tensor
    workers_decision_time: torch.Tensor
    workers_current_time: torch.Tensor
    #workers_used_capacity: torch.Tensor
    robots_work_capacity: torch.Tensor
    robots_start_location: torch.Tensor
    robots_current_destination_location: torch.Tensor
    robots_next_decision_time: torch.Tensor  # tracks the next decision time for all the robots

    next_decision_time: torch.Tensor # time at which the next decision is made. (0 t begin with)
    robot_taking_decision: torch.Tensor
    current_time: torch.Tensor  # stores the value for the current time

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
        return super(StateTOTP, self).__getitem__(key)


    @staticmethod
    def initialize(input, time, duration, t, new_task, w, new_worker, prev_state,
                   visited_dtype = torch.uint8):

        depot = input['depot']
        loc = input['t_loc'][:, 0:t, : ]
        pay = input['t_pay'][:, :, 0:t].squeeze(-1)
        demand = input['demand'][:, 0:t ]
        _, t_num = demand.size()

        tasks_start_time=input['t_start'][:, 0:t, : ]
        tasks_deadline_time=input['t_deadline'][:, 0:t, : ]

        workers_deadline_time = input['w_deadline'][:, 0:w, : ]
        workers_decision_time=input['w_start'][:, 0:w, : ]
        capacity = input['w_capacity'][:, 0:w ]
        speed = input['w_speed'][:, 0:w ]
        # riato=input['w_riato'][:, w_d:w ].squeeze(-1)

        w_loc=input['w_loc'][:, 0:w, : ]
        _, w_num, _ = w_loc.size()

        batch_size, n_loc, _ = loc.size()  # n_loc = graph_size
        #_, w_size, _ =cur_coord.size()
        sorted_time, indices = torch.sort(workers_decision_time)
        robot_taking_decision = indices[torch.arange(batch_size), 0].to(loc.device).unsqueeze(-1)

        if len(prev_state) == 0:
            lengths = torch.zeros(batch_size, w_num, device=loc.device)
            workers_current_time = demand.new_ones(batch_size, w_num).to(loc.device)
            cur_coord = w_loc
            demand = torch.cat((torch.zeros(batch_size, 1, device=loc.device), demand), 1)
            tasks_finish_time = torch.zeros((batch_size, n_loc), dtype=torch.float, device=loc.device)
            used_capacity = demand.new_zeros(batch_size, w_num).to(loc.device)
            # tour = demand.new_zeros(batch_size, w_num).to(loc.device)
            prev_a=torch.zeros(batch_size, w_num, dtype=torch.long, device=loc.device)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            )
            workers_done = (
                torch.zeros(
                    batch_size, 1, w_num,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (w_num + 63) // 64, dtype=torch.int64, device=loc.device)
             )
        else:
            workers_current_time = workers_decision_time.clone()
            depot_zeros = demand.new_zeros((batch_size, 1), dtype=torch.uint8)
            demand1 = demand.new_ones(batch_size, new_task)
            tasks_finish_time1=torch.zeros((batch_size, new_task), dtype=torch.float, device=loc.device)
            prev_state.tasks_finish_time.shape
            tasks_finish_time =torch.cat((prev_state.tasks_finish_time, tasks_finish_time1), -1).to(device=loc.device)
            visited_1 = (  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, new_task,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            )
            visited_1.shape
            if t_num == new_task:
                demand=torch.cat((depot_zeros, demand1), -1).to(device=loc.device)
                visited_ = torch.cat((depot_zeros[:, None, :], visited_1), -1).to( device=loc.device)
            else:
                demand = torch.cat((prev_state.demand, demand1), -1).to(
                    device=loc.device)
                prev_state.visited_.shape
                visited_ = torch.cat(
                    (prev_state.visited_, visited_1), -1).to(
                    device=loc.device)
            visited_.shape
            demand.shape
            # visited_.shape
            used_capacity1 = demand.new_zeros(batch_size, new_worker)
            prev_a1 = torch.zeros(batch_size, new_worker, dtype=torch.long, device=loc.device)
            lengths = torch.cat((prev_state.lengths, prev_a1), -1).to(
                    device=loc.device)
            cur_coord1 = w_loc
            if w_num==new_worker:
                used_capacity = used_capacity1
                prev_a = prev_a1
                cur_coord = cur_coord1
            else:
                used_capacity = torch.cat((prev_state.used_capacity, used_capacity1), -1).to(
                    device=loc.device)
                prev_a = torch.cat((prev_state.prev_a, prev_a1), -1)
                if new_worker==0:
                    cur_coord = prev_state.cur_coord
                else:
                    cur_coord = torch.cat(
                        (prev_state.cur_coord, cur_coord1[:, -new_worker:]), -2)
            workers_done = (
                torch.zeros(
                    batch_size, 1, w_num,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (w_num + 63) // 64, dtype=torch.int64, device=loc.device)
            ).squeeze(-2)
            exe_capacity = (used_capacity>=capacity).to(torch.int)
            exe_workers_deadline_time=(workers_deadline_time.squeeze(-1)<=time).to(torch.int)
            workers_done = (workers_done|exe_capacity|exe_workers_deadline_time)[:, None, : ].to(torch.uint8)

        return StateTOTP(
            coords=torch.cat((depot[:, None, :], loc), -2),  # [batch_size, graph_size, 2]]
            demand=demand,
            pay=pay,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            tasks_deadline_time=tasks_deadline_time,
            tasks_start_time=tasks_start_time,
            tasks_finish_time=tasks_finish_time,
            current_time=torch.zeros((batch_size, 1), dtype=torch.float, device=loc.device),
            next_decision_time=workers_decision_time,
            #robot_taking_decision=torch.zeros((batch_size, 1), dtype=torch.int64, device=loc.device),
            robot_taking_decision=robot_taking_decision,
            wor=torch.arange(w_num, dtype=torch.int64, device=loc.device)[:, None],
            speed=speed,
            # prev_a is current node
            prev_a=prev_a,
            used_capacity=used_capacity,
            capacity=capacity,
            workers_decision_time=workers_decision_time,
            workers_current_time=workers_current_time,
            workers_deadline_time=workers_deadline_time,
            visited_=visited_,
            workers_done=workers_done,
            cur_coord=cur_coord,
            robots_work_capacity=input['w_capacity'],
            robots_next_decision_time=workers_decision_time,
            robots_start_location=cur_coord,
            robots_current_destination_location=cur_coord,
            lengths=lengths,
            # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):
        assert self.all_finished()
        # coords: [batch_size, graph_size+1, 2]
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, wor, t, T):  # [batch_size, num_veh]

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        prev_a = selected  # [batch_size, num_wor]
        batch_size, _ = selected.size()
        workers_done = self.workers_done

        current_time = self.workers_decision_time

        # Add the length, coords:[batch_size, graph_size, 2]
        cur_coord = self.coords.gather(  # [batch_size, num_wor, 2]
            1,
            selected[:, :, None].expand(selected.size(0), len(self.wor), self.coords.size(-1))
        )
        b = cur_coord[torch.arange(batch_size), wor]
        cur_coord = torch.stack([self.cur_coord.contiguous()], 0).squeeze(0)
        cur_coord[torch.arange(batch_size), wor] = b

        time = ((cur_coord - self.cur_coord).norm(2, 2) / self.speed[torch.arange(batch_size), wor]).unsqueeze(-1)
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        for i in range (len(wor)):
            if time[i, wor[i]]==0:
                workers_done[i, 0, wor[i]] = 1
                if t>T:
                    self.workers_decision_time[i, wor[i]] += 100

        temp_time = time[torch.arange(batch_size), wor] + current_time[torch.arange(batch_size), wor]
        self.workers_decision_time[torch.arange(batch_size), wor] = temp_time

        aa1=selected[torch.arange(batch_size), wor]-1
        self.tasks_finish_time[torch.arange(batch_size), aa1]=(self.workers_decision_time[torch.arange(batch_size), wor]).squeeze(-1)
        # workers_decision_time = self.workers_decision_time
        workers_decision_time = self.workers_decision_time

        selected_demand_broad = self.demand[:, :, None].gather(  # [batch_size, num_wor]
            1,
            prev_a[torch.arange(batch_size), wor][:, None, None].expand(prev_a.size(0), len(self.wor),
                                                                        self.demand[:, :, None].size(-1))
        ).squeeze(2)

        selected_demand = torch.zeros_like(selected_demand_broad)
        selected_demand[torch.arange(batch_size), wor] = selected_demand_broad[torch.arange(batch_size), wor].clone()
        self.demand[torch.arange(batch_size), prev_a[torch.arange(batch_size), wor]] = 0

        used_capacity = self.used_capacity
        used_capacity[torch.arange(batch_size), wor] = (self.used_capacity[torch.arange(batch_size), wor] +
                                                        selected_demand[torch.arange(batch_size), wor]) \
                                                       # * (prev_a[torch.arange(batch_size), wor] != 0).float()  # [batch_size, num)_veh]

        # if t>T:
        is_worker_done = ((self.used_capacity[torch.arange(batch_size)] >= self.capacity[torch.arange(batch_size)]).to(torch.bool)
                          | (self.workers_deadline_time.squeeze(-1) <= self.workers_decision_time.squeeze(-1)).to(torch.bool))[:, None, :]
        # else:
        #     aa1 = (self.used_capacity[torch.arange(batch_size)] >= self.capacity[
        #         torch.arange(batch_size)]).to(torch.bool)
        #     aa2 = (self.workers_deadline_time.squeeze(-1) <= self.workers_decision_time.squeeze(-1)).to(torch.bool)
        #     aa3 = (t < self.workers_decision_time.squeeze(-1)).to(torch.bool)
        #     is_worker_done = (aa1 | aa2| aa3)[:, None, :]

        # workers_done = self.workers_done
        workers_done[is_worker_done] = 1

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[torch.arange(batch_size), wor][:, None, None].expand_as(
                self.visited_[:, :, 0:1]), 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a[torch.arange(batch_size), wor])

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
        aa=aa-(len - 1)
        bb=bb-len1
        valid_index = (aa-(len - 1) == 0).nonzero()
        valid_index1 = (bb == 0).nonzero()
        # aa = time[valid_index, wor[valid_index]]
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
    def get_mask(self, wor, t, T):
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
        visited_loc.shape

        cur_time=self.workers_decision_time[torch.arange(batch_size), wor]
        distance_matrix = (self.cur_coord[:, :, None, :] - self.coords[:, None, 1:, :]).norm(p=2, dim=-1).to(device=visited_loc.device)

        time_matrix = torch.mul(distance_matrix, (1 / self.speed)).to(device=visited_loc.device)  # 时间除以路程=1/速度

        cur_time=(cur_time+time_matrix[torch.arange(batch_size), wor])-self.tasks_start_time.squeeze(-1)
        finnal_time = self.tasks_deadline_time.max().int()

        cur_time64 = torch.tensor(cur_time+1, dtype=torch.int64)
        cur_time64 = torch.where(cur_time64 < 0, 0, cur_time64)
        cur_time64 = torch.where(cur_time64 > int(finnal_time-1), int(finnal_time-1), cur_time64)

        bb3=self.pay.gather(1, (cur_time64[:, None, :]).expand(cur_time.size(0), 1, self.pay.size(-1))).squeeze(1)
        bb3.shape
        positive_pay = (bb3-0*(distance_matrix[torch.arange(batch_size), wor]) <= 0)[:, None, :]
        positive_pay.shape
        arrived_before_task=((time_matrix[torch.arange(batch_size), wor]+self.workers_decision_time[torch.arange(batch_size), wor] )
                        <= (self.tasks_start_time.squeeze(-1)))[:, None, :]

        exceeds_task_deadline=((time_matrix[torch.arange(batch_size), wor] + self.workers_decision_time[torch.arange(batch_size), wor])
        >=self.tasks_deadline_time.squeeze(-1))[:, None, :]
        exceeds_task_deadline.shape

        exceeds_time = ((time_matrix[torch.arange(batch_size), wor]+self.workers_decision_time[torch.arange(batch_size), wor] )
                        >= (self.workers_deadline_time[torch.arange(batch_size), wor]) )[:, None, :] \
                       | exceeds_task_deadline | arrived_before_task

        exceeds_time.shape
        # Nodes that cannot be visited are already visited or too much demand to be served now

        exceeds_cap = (self.demand[self.ids, 1:] + (self.used_capacity[torch.arange(batch_size), wor].unsqueeze(-1))[
            ..., None].expand_as(self.demand[self.ids, 1:]) >
                       (self.capacity[torch.arange(batch_size), wor]).unsqueeze(-1)[..., None].expand_as(
                           self.demand[self.ids, 1:]))
        exceeds_cap.shape

        mask_loc = visited_loc.to(torch.bool) | exceeds_cap | exceeds_time | positive_pay

        # Cannot visit the depot if just visited and still unserved nodes

        mask_depot = torch.as_tensor(torch.ones((mask_loc.size()[0], 1)).clone().detach(), dtype=torch.bool,
                                  device=mask_loc.device)  # [batch_size, 1]

        aa = mask_loc.to(torch.float)
        aa = torch.count_nonzero(aa[:, :, :].squeeze(1), dim=1).reshape(-1, 1)


        for k in range(batch_size):
            if aa[k] == task_size - 1:
                a = self.prev_a[k, wor[k]]
                if a > 0:
                    mask_loc[k, :, a - 1] = False
                else:
                    mask_depot [k, 0] = False

        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size]


    def construct_solutions(self, actions):
        return actions
