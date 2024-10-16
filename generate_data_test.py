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

def generate_mrta_data(size_t, size_w,num_samples):
    max_n_agent = size_w
    # data = []
    # t_loc=[]
    period = 6

    init_duration_w = torch.randint(6, 7, (size_w, 1)).to(torch.int)
    init_dis = torch.randint(5, 6, (size_w, 1)).to(torch.int)

    w_loc = torch.FloatTensor(max_n_agent, 2).uniform_(0, 1)
    w_capacity = torch.randint(1, 21, (max_n_agent, 1), dtype=torch.float,
                               device=w_loc.device).view(-1)
    w_speed = ((torch.randint(1, 6, (max_n_agent, 1)).to(torch.float)) / 10)
    w_riato = (torch.randint(1, 100, (max_n_agent, 1)).to(torch.float) / 100)
    if size_w >= 10:
        agents_starttime = (torch.randint(1, int(size_w / 2) + 1, (max_n_agent, 1)).to(torch.float))
    else:
        agents_starttime = (torch.randint(1, int(size_w) + 1, (max_n_agent, 1)).to(torch.float))
    sorted, indices = torch.sort(agents_starttime, 0)
    agents_starttime = sorted
    agents_deadline = agents_starttime + 6


    for i in range(num_samples):

        init_zore = torch.randint(0, 1, (size_t, 1)).to(torch.int)
        init_duration_t = torch.randint(6, 7, (size_t, 1)).to(torch.int)
        t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
        # t_loc=truncated_normal(size_t, 0.4)
        if size_w >= 10:
            t_starttime = (torch.randint(1, int(size_w / 2) + 1, (size_t, 1)).to(torch.float))
        else:
            t_starttime = (torch.randint(1, int(size_w)+1, (size_t, 1)).to(torch.float))
        sorted, indices = torch.sort(t_starttime, 0)
        t_starttime = sorted
        t_deadline = t_starttime + period
        init_pay = torch.randint(10, 30, (size_t, 1)).to(torch.float).squeeze(-1)
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
            zeros_sus = torch.zeros(finnal_time - 6 - i, count_arrived[i]).to(device=t_starttime.device)
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
            'w_riato': w_riato
        }
        #生成工人和任务信息写入到excel 供启发式算法使用
        t_loc1=t_loc[:,0]
        t_loc2 = t_loc[:, 1]
        df = pd.DataFrame({'begin_time': t_starttime.flatten(), 'loc1': t_loc1.flatten(),'loc2': t_loc2.flatten(),
                           'duration': init_duration_t.flatten(),  'pay': init_pay.flatten(),  'depend_pro': init_zore.flatten(),
                           'depend_sus': init_zore.flatten(), 'conflict': init_zore.flatten() })
        if size_t==40:
            file_path1 = 'task_data40.xlsx'
        if size_t == 80:
            file_path1 = 'task_data80.xlsx'
        if size_t == 120:
            file_path1 = 'task_data120.xlsx'
        if size_t==160:
            file_path1 = 'task_data160.xlsx'
        df.to_excel(file_path1, index=False)

        w_loc1=w_loc[:,0]
        w_loc2 = w_loc[:, 1]
        df1 = pd.DataFrame({'begin_time': agents_starttime.flatten(), 'loc1': w_loc1.flatten(),'loc2': w_loc2.flatten(),
                           'dis': init_dis.flatten(), 'capacity': w_capacity.flatten(), 'speed': w_speed.flatten(),
                            'duration': init_duration_w.flatten(), 'riato': w_riato.flatten() })
        file_path2 = 'worker_data5.xlsx'
        df1.to_excel(file_path2, index=False)
        data=[]
        data.append(case_info)
        if size_t==40:
            save_dataset(data, datadir + "/" + "5-40my" +"test" + "_tasks_" + problem + ".pkl")
        if size_t == 80:
            save_dataset(data, datadir + "/" + "5-80my" +"test" + "_tasks_" + problem + ".pkl")
        if size_t == 120:
            save_dataset(data, datadir + "/" + "5-120my" +"test" + "_tasks_" + problem + ".pkl")
        if size_t==160:
            save_dataset(data, datadir + "/" + "5-160my" +"test" + "_tasks_" + problem + ".pkl")
        size_t+=40

    # data.append(case_info)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument('--tasks_size', type=int, default=40, help="The size of the problem graph")
    parser.add_argument('--workers_size', type=int, default=5, help="The size of the problem graph")
    parser.add_argument("--name", type=str, default='mrta', help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='mrta',
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

    dataset = generate_mrta_data(opts.tasks_size,opts.workers_size, 4)


