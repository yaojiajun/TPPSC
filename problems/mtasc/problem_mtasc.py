from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from problems.mtasc.state_mtasc import StateMTASC
from utils.beam_search import beam_search
import copy
from random import randint, sample, shuffle
import random

class MTASC(object):
    NAME = 'mtasc'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi, veh_list, tour, finish_time):  # pi is a solution sequence, [batch_size, num_veh, tour_len]

        SPEED = dataset['w_speed'].squeeze(-1)
        batch_size, graph_size = dataset['demand'].size()
        w_loc = dataset['w_loc'].squeeze(-1)
        w_score=dataset['w_score']

        t_pay=dataset['t_pay'].squeeze(-1)
        t_deadline=dataset['t_deadline'].squeeze(-1)
        num_veh = SPEED.size(-1)

        zero = torch.zeros(batch_size, 1, device=SPEED.device)
        tour1 = tour.transpose(1, 0).transpose(1, 2).transpose(0, 1)

        all_loc = torch.cat((dataset['w_loc'][:, :, None, :].transpose(1, 0), (dataset['t_loc']).expand(num_veh, -1, -1, -1)), -2)

        finnal_time = t_deadline.max().int()

        finish_time=torch.where(finish_time > 100, finish_time%100, finish_time)[:,None,:]
        finish_time = torch.tensor(finish_time+1, dtype=torch.int64)
        finnal_time = torch.tensor(finnal_time, dtype=torch.int64)
        finish_time = torch.where(finish_time < 0, 0, finish_time)
        finish_time = torch.where(finish_time > (finnal_time-1), (finnal_time-1), finish_time)

        pay=t_pay.gather(1,finish_time.expand(finish_time.size(0), 1 , t_pay.size(-1))).squeeze(1)
        all_pay = torch.cat((zero[:, :, None], pay[:, :, None]), 1).expand(num_veh, -1, -1, -1)

        cost_1 = all_pay.gather(2, tour1[..., None].expand(*tour1.size(), all_pay.size(-1)))

        aa = w_loc[:, :, :][:, None, :].transpose(2, 0).transpose(1, 2)
        dis_1 = torch.cat(
            (aa, all_loc.gather(2, tour1[..., None].expand(*tour1.size(), all_loc.size(-1)))),
            -2)

        worker_num, batch_size, rout_long, _ = dis_1.size()  # worker num, batch size, rout long

        flag = 2 * (dis_1[:, :, 1:] - dis_1[:, :, :-1]).norm(p=2, dim=-1)
        flag[flag != 0] = 1

        cost_2=w_score.transpose(0, 1).expand(-1, -1, rout_long-1)

        total_cost = ((0.5*cost_1[:, :].squeeze(-1) + cost_2*0.5- 20 * (dis_1[:, :, 1:] - dis_1[:, :, :-1]).norm(p=2,
                                                                                                 dim=-1)) * flag).sum(
            -1)

        return torch.sum(total_cost, dim=0), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        kwargs['is_dynamic'] = True
        return MTASCDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMTASC.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MTASC.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    task_num, _ = args["t_loc"].size()
    depot = args["depot"]
    t_loc = args["t_loc"]
    demand= args["demand"]
    t_start = args["t_start"]
    t_deadline = args["t_deadline"]
    t_pay= args["t_pay"]
    w_loc = args["w_loc"]
    w_capacity = args["w_capacity"]
    w_start= args["w_start"]
    w_deadline = args["w_deadline"]
    w_speed = args["w_speed"]
    w_riato= args["w_riato"]
    grid_size = 1


    return {
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        't_loc': torch.tensor(t_loc, dtype=torch.float),  # scale demand
        'demand': torch.tensor(demand, dtype=torch.float) / grid_size,
        't_start': torch.tensor(t_start, dtype=torch.float),
        't_deadline': torch.tensor(t_deadline, dtype=torch.float) / grid_size,
        't_pay': torch.tensor(t_pay, dtype=torch.float),  # scale demand
        'w_loc': torch.tensor(w_loc, dtype=torch.float) / grid_size,
        'w_capacity': torch.tensor(w_capacity, dtype=torch.float),
        'w_start': torch.tensor(w_start, dtype=torch.float) / grid_size,
        'w_deadline': torch.tensor(w_deadline, dtype=torch.float),  # scale demand
        'w_speed': torch.tensor(w_speed, dtype=torch.float) / grid_size,
        'w_riato': torch.tensor(w_riato, dtype=torch.float),
    }


class MTASCDataset(Dataset):

    def __init__(self, filename=None, size_t=50, size_w=10, num_samples=10000, offset=0, distribution=None, is_dynamic=False):
        super(MTASCDataset, self).__init__()

        self.data_set = []

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)  # (N, size+1, 2)

            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:
            if is_dynamic:

                max_n_agent = size_w
                data = []
                period=6
                for i in range(num_samples):
                    t_loc=torch.FloatTensor(size_t, 2).uniform_(0, 1)
                    if size_w>=10:
                        t_starttime = (torch.randint(1, int(size_w/2), (size_t, 1)).to(torch.float))
                    else:
                        t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
                    sorted, indices = torch.sort(t_starttime, 0)
                    t_starttime = sorted
                    t_deadline = t_starttime + period
                    t_pay=torch.randint(1, 100, (size_t, 1)).to(torch.float).squeeze(-1)
                    pay = []
                    for i in range(period):
                        if i<period/2:
                            pay.append(t_pay)
                        else:
                            pay.append(t_pay-t_pay*0.10*(i-period/2+1))
                    pay=torch.stack(pay, 0)
                    pay[pay<0]=0

                    finnal_time = period+int(size_w)
                    count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
                    prev_time = count_arrived.size(-1)
                    for i in range(prev_time):
                        zeros_prev = torch.zeros( i, count_arrived[i]).to(device=t_starttime.device)
                        zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                        p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                        if i > 0:
                            prev_p = torch.cat((prev_p, p), 1)
                        else:
                            prev_p = p
                    prev_p.shape
                    pay = prev_p.unsqueeze(-1)

                    if size_w >= 10:
                        agents_starttime = (torch.randint(1, int(size_w / 2), (max_n_agent, 1)).to(torch.float))
                    else:
                        agents_starttime = (torch.randint(1, int(size_w), (max_n_agent, 1)).to(torch.float))
                    sorted, indices = torch.sort(agents_starttime, 0)
                    agents_starttime = sorted
                    agents_deadline = agents_starttime + period

                    case_info = {
                        'depot': torch.FloatTensor(2).uniform_(0, 0),
                        't_loc': t_loc,
                        'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                        't_start': t_starttime,
                        't_deadline': t_deadline,
                        't_pay': pay,
                        # 'relation': relation,
                        'w_loc': torch.FloatTensor(max_n_agent, 2).uniform_(0, 1),
                        'w_capacity': torch.randint(1, 20, (max_n_agent, 1), dtype=torch.float,
                                                    device=agents_deadline.device).view(-1),
                        'w_start': agents_starttime,
                        'w_deadline': agents_deadline,
                        'w_speed':((torch.randint(1, 6, (max_n_agent, 1)).to(torch.float)) / 10),
                        'w_score': (torch.randint(1, 100, (max_n_agent, 1)).to(torch.float) )
                    }
                    data.append(case_info)
                self.data = data

        self.size_t = len(self.data)  # num_samples

    def __len__(self):
        return self.size_t

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data

    def get_dynamic_data(self, t_pay, finnal_time, count_arrived, count_dead, strength=0.1):
        pay1=[]
        for j in range(finnal_time):
            pay1.append([0])
        arrived_all_tasks = 0
        prev_arrived_tasks=0

        for i in range(count_arrived.size(0)):
            arrived_all_tasks+=count_arrived[i]
            cur_task = t_pay[prev_arrived_tasks:arrived_all_tasks]
            for j in range(finnal_time):
                if j<=i:
                    for k in range(cur_task.size(0)):
                        pay1[j].append(0)
                elif i < j <= i+2:
                    for k in range(cur_task.size(0)):
                        pay1[j].append( float(cur_task[k]*j*0.25))
                elif i+2 < j <= i + 4:
                    for k in range(cur_task.size(0)):
                        pay1[j].append( float(cur_task[k]))
                elif i + 4 < j <= i + 6:
                    for k in range(cur_task.size(0)):
                        pay1[j].append( float(cur_task[k]-cur_task[k]* ((j-3+i)) * 0.25))
                elif i + 6 < j:
                    for k in range(cur_task.size(0)):
                        pay1[j].append(0)
                # pay.append(pay1)
            prev_arrived_tasks=arrived_all_tasks.int()

        pay =  torch.tensor(pay1, dtype=torch.float)
        pay=pay[:, 1:].unsqueeze(-1)

        return pay

