import argparse
import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from utils.data_utils import save_dataset

 
def truncated_normal(graph_size, sigma):
    mu = 0.5
    lower, upper = 0, 1
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    return torch.stack([torch.from_numpy(X.rvs(graph_size)), torch.from_numpy(X.rvs(graph_size))], 1)

def generate_tpsc_data(size_t, size_w, type, num_samples):

    if 'c' == type: #1
        capacity=[2,5,10,15,20]
        speed=5
        period=6
        reward=100
        score=100
        value=capacity

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)
        w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))

        for i in range(num_samples):
            t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
            if size_w >= 10:
                t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
            else:
                t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
            sorted, indices = torch.sort(t_starttime, 0)
            t_starttime = sorted
            t_deadline = t_starttime + period

            init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
            pay = []
            for i in range(period):
                if i < period / 2:
                    pay.append(init_pay)
                else:
                    pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
            pay = torch.stack(pay, 0)
            pay[pay < 0] = 0

            finnal_time = t_deadline.max().int()
            count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
            prev_time = count_arrived.size(-1)
            for i in range(prev_time):
                zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                if i > 0:
                    prev_p = torch.cat((prev_p, p), 1)
                else:
                    prev_p = p
            prev_p.shape
            pay = prev_p.unsqueeze(-1)

            init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
            init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)
            init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)
            w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)

            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted
            agents_deadline = agents_starttime + period

            for capacity in value:  # capacity
                w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                           device=w_loc.device).view(-1)
                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" + "c"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'c_task_data{size_t}{capacity}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'c_worker_data{size_w}{capacity}.xlsx'
                df1.to_excel(file_path2, index=False)

    if 's'== type: #2
        capacity = 10
        speed = [1, 2, 3, 4, 5]
        period = 6
        reward=100
        score=100
        # size_t=100
        value = speed

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)
        w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))

        for i in range(num_samples):
            t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
            if size_w >= 10:
                t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
            else:
                t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
            sorted, indices = torch.sort(t_starttime, 0)
            t_starttime = sorted
            t_deadline = t_starttime + period

            init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
            pay = []
            for i in range(period):
                if i < period / 2:
                    pay.append(init_pay)
                else:
                    pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
            pay = torch.stack(pay, 0)
            pay[pay < 0] = 0

            finnal_time = t_deadline.max().int()
            count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
            prev_time = count_arrived.size(-1)
            for i in range(prev_time):
                zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                if i > 0:
                    prev_p = torch.cat((prev_p, p), 1)
                else:
                    prev_p = p
            prev_p.shape
            pay = prev_p.unsqueeze(-1)

            init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
            init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)
            init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)
            w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                       device=w_loc.device).view(-1)


            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted
            agents_deadline = agents_starttime + period

            for speed in value:  # capacity
                w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)

                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" +"s"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f's_task_data{size_t}{speed}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f's_worker_data{size_w}{speed}.xlsx'
                df1.to_excel(file_path2, index=False)

    if 'p' == type: #3
        capacity = 10
        speed = 5
        period = [2,4,6,8,10]
        reward = 100
        score=100
        # size_t=100
        value = period

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)
        w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))

        for i in range(num_samples):
            t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
            if size_w >= 10:
                t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
            else:
                t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
            sorted, indices = torch.sort(t_starttime, 0)
            t_starttime = sorted


            init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
            init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)

            w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                       device=w_loc.device).view(-1)
            w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)

            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted

            for period in value:  # period
                t_deadline = t_starttime + period

                pay = []
                for i in range(period):
                    if i < period / 2:
                        pay.append(init_pay)
                    else:
                        pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
                pay = torch.stack(pay, 0)
                pay[pay < 0] = 0
                finnal_time = t_deadline.max().int()
                count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
                prev_time = count_arrived.size(-1)
                for i in range(prev_time):
                    zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                    zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                    p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                    if i > 0:
                        prev_p = torch.cat((prev_p, p), 1)
                    else:
                        prev_p = p
                prev_p.shape
                pay = prev_p.unsqueeze(-1)

                init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)
                init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
                agents_deadline = agents_starttime + period

                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" +"p"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'p_task_data{size_t}{period}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'p_worker_data{size_w}{period}.xlsx'
                df1.to_excel(file_path2, index=False)

    if 'r' == type: #4
        capacity = 10
        speed = 5
        period = 6
        reward = [20,40,60,80,100]
        score=100
        # size_t=100
        value = reward

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)
        w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))

        for i in range(num_samples):
            t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
            if size_w >= 10:
                t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
            else:
                t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
            sorted, indices = torch.sort(t_starttime, 0)
            t_starttime = sorted
            t_deadline = t_starttime + period

            init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
            init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)
            init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)
            w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                       device=w_loc.device).view(-1)
            w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)

            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted
            agents_deadline = agents_starttime + period

            for reward in value:  # capacity

                init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
                pay = []
                for i in range(period):
                    if i < period / 2:
                        pay.append(init_pay)
                    else:
                        pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
                pay = torch.stack(pay, 0)
                pay[pay < 0] = 0

                finnal_time = t_deadline.max().int()
                count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
                prev_time = count_arrived.size(-1)
                for i in range(prev_time):
                    zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                    zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                    p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                    if i > 0:
                        prev_p = torch.cat((prev_p, p), 1)
                    else:
                        prev_p = p
                prev_p.shape
                pay = prev_p.unsqueeze(-1)

                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" +"r"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'r_task_data{size_t}{reward}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'r_worker_data{size_w}{reward}.xlsx'
                df1.to_excel(file_path2, index=False)


    if 'tn' == type:  #5 task_num
        capacity = 10
        speed = 5
        period = 6
        reward = 100
        size_t=[50,100]#[60,80,100,120,140]
        score=100
        value = size_t

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)
        w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))

        for i in range(num_samples):

            init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)
            w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)
            w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                       device=w_loc.device).view(-1)
            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted
            agents_deadline = agents_starttime + period

            for size_t in value:  # capacity

                t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
                if size_w >= 10:
                    t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
                else:
                    t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
                sorted, indices = torch.sort(t_starttime, 0)
                t_starttime = sorted
                t_deadline = t_starttime + period

                init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
                pay = []
                for i in range(period):
                    if i < period / 2:
                        pay.append(init_pay)
                    else:
                        pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
                pay = torch.stack(pay, 0)
                pay[pay < 0] = 0

                finnal_time = t_deadline.max().int()
                count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
                prev_time = count_arrived.size(-1)
                for i in range(prev_time):
                    zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                    zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                    p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                    if i > 0:
                        prev_p = torch.cat((prev_p, p), 1)
                    else:
                        prev_p = p
                prev_p.shape
                pay = prev_p.unsqueeze(-1)
                init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
                init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)

                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" + "n"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'n_task_data{size_w}{size_t}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'n_worker_data{size_w}{size_t}.xlsx'
                df1.to_excel(file_path2, index=False)

    if 'dis' == type:  #5 task_num
        capacity = 10
        speed = 5
        period = 6
        reward = 100
        size_t=100
        task_dis=[0.2,0.4,0.6,0.8,1]
        score=100
        value = task_dis

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)
        w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))

        for i in range(num_samples):

            init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)
            w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)
            w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                       device=w_loc.device).view(-1)
            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted
            agents_deadline = agents_starttime + period

            for task_dis in value:  # capacity

                t_loc = truncated_normal(size_t, task_dis)
                if size_w >= 10:
                    t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
                else:
                    t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
                sorted, indices = torch.sort(t_starttime, 0)
                t_starttime = sorted
                t_deadline = t_starttime + period

                init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
                pay = []
                for i in range(period):
                    if i < period / 2:
                        pay.append(init_pay)
                    else:
                        pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
                pay = torch.stack(pay, 0)
                pay[pay < 0] = 0

                finnal_time = t_deadline.max().int()
                count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
                prev_time = count_arrived.size(-1)
                for i in range(prev_time):
                    zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                    zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                    p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                    if i > 0:
                        prev_p = torch.cat((prev_p, p), 1)
                    else:
                        prev_p = p
                prev_p.shape
                pay = prev_p.unsqueeze(-1)
                init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
                init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)

                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" + "n"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+"_" + str(task_dis)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'n_task_data{size_t}{task_dis}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'n_worker_data{size_w}{task_dis}.xlsx'
                df1.to_excel(file_path2, index=False)


    if 'ws' == type: #6
        capacity=10
        speed=5
        period=6
        reward=100
        worker_score = [20,40,60,80,100]
        value=worker_score

        w_loc = torch.FloatTensor(size_w, 2).uniform_(0, 1)

        for i in range(num_samples):
            t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
            if size_w >= 10:
                t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
            else:
                t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
            sorted, indices = torch.sort(t_starttime, 0)
            t_starttime = sorted
            t_deadline = t_starttime + period

            init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
            pay = []
            for i in range(period):
                if i < period / 2:
                    pay.append(init_pay)
                else:
                    pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
            pay = torch.stack(pay, 0)
            pay[pay < 0] = 0

            finnal_time = t_deadline.max().int()
            count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
            prev_time = count_arrived.size(-1)
            for i in range(prev_time):
                zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                if i > 0:
                    prev_p = torch.cat((prev_p, p), 1)
                else:
                    prev_p = p
            prev_p.shape
            pay = prev_p.unsqueeze(-1)

            init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
            init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)
            init_duration_w = torch.randint(period, period + 1, (size_w, 1)).to(torch.int)
            init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)
            w_speed = ((torch.randint(1, speed + 1, (size_w, 1)).to(torch.float)) / 10)
            w_capacity = torch.randint(1, capacity + 1, (size_w, 1), dtype=torch.float,
                                       device=w_loc.device).view(-1)
            if size_w >= 10:
                agents_starttime = (torch.randint(1, int(size_w / 2), (size_w, 1)).to(torch.float))
            else:
                agents_starttime = (torch.randint(1, int(size_w), (size_w, 1)).to(torch.float))
            sorted, indices = torch.sort(agents_starttime, 0)
            agents_starttime = sorted
            agents_deadline = agents_starttime + period

            for score in value:  # capacity

                w_score = (torch.randint(1, score, (size_w, 1)).to(torch.float))
                case_info = {
                    'depot': torch.FloatTensor(2).uniform_(0, 0),
                    't_loc': t_loc,
                    'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
                    't_start': t_starttime,
                    't_deadline': t_deadline,
                    't_pay': pay,
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" + "ws"+ str(size_w) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'ws_task_data{size_t}{score}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(size_w):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'ws_worker_data{size_w}{score}.xlsx'
                df1.to_excel(file_path2, index=False)

    if 'wn' == type: #7
        capacity=10
        speed=5
        period=6
        reward=100
        score = 100
        worker_num=[6,8,10,12,14]
        value=worker_num

        for i in range(num_samples):
            t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
            if size_w >= 10:
                t_starttime = (torch.randint(1, int(size_w / 2), (size_t, 1)).to(torch.float))
            else:
                t_starttime = (torch.randint(1, int(size_w), (size_t, 1)).to(torch.float))
            sorted, indices = torch.sort(t_starttime, 0)
            t_starttime = sorted
            t_deadline = t_starttime + period

            init_pay = torch.randint(1, reward, (size_t, 1)).to(torch.float).squeeze(-1)
            pay = []
            for i in range(period):
                if i < period / 2:
                    pay.append(init_pay)
                else:
                    pay.append(init_pay - init_pay * 0.10 * (i - period / 2 + 1))
            pay = torch.stack(pay, 0)
            pay[pay < 0] = 0

            finnal_time = t_deadline.max().int()
            count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
            prev_time = count_arrived.size(-1)
            for i in range(prev_time):
                zeros_prev = torch.zeros(i, count_arrived[i]).to(device=t_starttime.device)
                zeros_sus = torch.zeros(finnal_time - period - i, count_arrived[i]).to(device=t_starttime.device)
                p = torch.cat((zeros_prev, pay[:, 0:count_arrived[i]], zeros_sus), 0)
                if i > 0:
                    prev_p = torch.cat((prev_p, p), 1)
                else:
                    prev_p = p
            prev_p.shape
            pay = prev_p.unsqueeze(-1)

            init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
            init_duration_t = torch.randint(period, period + 1, (size_t, 1)).to(torch.int)

            for num in value:  # capacity
                w_loc = torch.FloatTensor(num, 2).uniform_(0, 1)
                w_score = (torch.randint(1, score, (num, 1)).to(torch.float))
                init_duration_w = torch.randint(period, period + 1, (num, 1)).to(torch.int)
                init_dis = torch.randint(5, 6, (num, 1)).to(torch.int)
                w_speed = ((torch.randint(1, speed + 1, (num, 1)).to(torch.float)) / 10)
                w_capacity = torch.randint(1, capacity + 1, (num, 1), dtype=torch.float,
                                           device=w_loc.device).view(-1)
                if num >= 8:
                    agents_starttime = (torch.randint(1, int(num / 2), (num, 1)).to(torch.float))
                else:
                    agents_starttime = (torch.randint(1, int(num), (num, 1)).to(torch.float))
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
                    'w_loc': w_loc,
                    'w_capacity': w_capacity,
                    'w_start': agents_starttime,
                    'w_deadline': agents_deadline,
                    'w_speed': w_speed,
                    'w_score': w_score
                }

                data = []
                data.append(case_info)
                save_dataset(data,
                             datadir + "/" + "wn"+ str(num) + "_" + str(size_t) + "tasks_" + problem + "_" + str(period) + "_"
                             +  str(capacity) + "_" + str(speed) + "_" + str(reward) + "_" + str(score)+".pkl")
                #生成工人和任务信息写入到excel 供启发式算法使用
                t_loc1=t_loc[:,0]
                t_loc2 = t_loc[:, 1]
                t_id = []
                t_type = []
                for l in range(size_t):
                    t_id.append(l+1)
                    t_type.append('t')
                df = pd.DataFrame({'id': t_id,'begin_time': t_starttime.flatten(),'type': t_type, 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                                   'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                                   'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
                file_path1 = f'wn_task_data{size_t}{num}.xlsx'
                df.to_excel(file_path1, index=False)

                w_loc1=w_loc[:,0]
                w_loc2 = w_loc[:, 1]
                w_id = []
                w_type = []
                for l in range(num):
                    w_id.append(l+1)
                    w_type.append('w')
                df1 = pd.DataFrame({'id': w_id, 'begin_time': agents_starttime.flatten(), 'type': w_type, 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                                   'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                                    'duration': init_duration_w.flatten(), 'score': w_score.flatten() })
                file_path2 = f'wn_worker_data{size_w}{num}.xlsx'
                df1.to_excel(file_path2, index=False)


    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument('--tasks_size', type=int, default=100, help="The size of the problem graph")
    parser.add_argument('--workers_size', type=int, default=5, help="The size of the problem graph")
    parser.add_argument('--type', type=str, default='tn',
                        help="c:capacity s:speed p:period r:reward tn: task_num ws:worker_score wn:worker_num dis:distribution" )
    parser.add_argument("--name", type=str, default='tppsc', help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='tppsc',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=1, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=50,
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    problem = "sc"
    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    np.random.seed(opts.seed)

    dataset = generate_tpsc_data(opts.tasks_size,opts.workers_size, opts.type, 1) #c


