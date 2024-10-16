import argparse
import os
import torch
import numpy as np
from utils.data_utils import save_dataset


def generate_mrta_data(size_t, size_w,num_samples):
    max_n_agent = size_w
    data = []
    period = 6
    for i in range(num_samples):
        t_loc = torch.FloatTensor(size_t, 2).uniform_(0, 1)
        t_starttime = (torch.randint(1, int(size_w/2)+1, (size_t, 1)).to(torch.float))
        sorted, indices = torch.sort(t_starttime, 0)
        t_starttime = sorted
        t_deadline = t_starttime + period
        t_pay = torch.randint(10, 20, (size_t, 1)).to(torch.float).squeeze(-1)
        pay = []
        for i in range(period):
            if i < period / 2:
                pay.append(t_pay)
            else:
                pay.append(t_pay - t_pay * 0.15 * (i - period / 2 + 1))
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

        agents_starttime = (torch.randint(1, int(size_w/2)+1, (max_n_agent, 1)).to(torch.float))
        sorted, indices = torch.sort(agents_starttime, 0)
        agents_starttime = sorted
        agents_deadline = agents_starttime + 6

        case_info = {
            'depot': torch.FloatTensor(2).uniform_(0, 0),
            't_loc': t_loc,
            'demand': (torch.FloatTensor(size_t).uniform_(0, 0).int() + 1).float(),
            't_start': t_starttime,
            't_deadline': t_deadline,
            't_pay': pay,
            # 'relation': relation,
            'w_loc': torch.FloatTensor(max_n_agent, 2).uniform_(0, 1),
            'w_capacity': torch.randint(1, 10, (max_n_agent, 1), dtype=torch.float,
                                        device=agents_deadline.device).view(-1),
            'w_start': agents_starttime,
            'w_deadline': agents_deadline,
            'w_speed': ((torch.randint(2, 3, (max_n_agent, 1)).to(torch.float)) / 10),
            'w_riato': (torch.randint(1, 100, (max_n_agent, 1)).to(torch.float) / 100)
        }
        data.append(case_info)
    # data.append(case_info)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument('--tasks_size', type=int, default=50, help="The size of the problem graph")
    parser.add_argument('--workers_size', type=int, default=5, help="The size of the problem graph")
    parser.add_argument("--name", type=str, default='mrta', help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='mrta',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=50,
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=4321, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    problem = "sc"
    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    np.random.seed(opts.seed)

    dataset = generate_mrta_data(opts.tasks_size,opts.workers_size, opts.dataset_size)

    save_dataset(dataset, datadir + "/" + "my" + "validation" +"_tasks_" + problem + ".pkl")

