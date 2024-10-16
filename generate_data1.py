import argparse
import os
import torch
import numpy as np
from utils.data_utils import save_dataset


def generate_mrta_data():
    with open('data/sc/data_00.txt', encoding='utf-8') as file:
        data = []
        agents_starttime=[]
        agents_location=[]
        agents_deadline=[]
        agents_capacity=[]
        agents_speed=[]
        agents_riato=[]
        t_starttime=[]
        t_location=[]
        t_deadline=[]
        t_depend=[]
        t_depend_des=[]
        t_conflict=[]
        t_pay =[]

        txt_info=next(file).strip("\n").split()
        worker_num=int(txt_info[0])
        task_num = int(txt_info[1])
        for line in file:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
            if data_line[2]=='w':
                agents_starttime.append([int(data_line[1])])
                agents_location.append([float(data_line[3]),float(data_line[4])])
                agents_deadline.append([float(data_line[8]) + int(data_line[1])])
                agents_capacity.append(int(data_line[6]))
                agents_speed.append([float(data_line[7])])
                agents_riato.append([float(data_line[8])])
            else:
                t_starttime.append([int(data_line[1])])
                t_location.append([float(data_line[3]),float(data_line[4])])
                t_deadline.append([int(data_line[5]) + int(data_line[1])])
                t_pay.append([float(data_line[6])])
                t_depend.append(data_line[7])
                t_depend_des.append(data_line[8])
                t_conflict.append(data_line[9])


        t_pay=torch.tensor(t_pay, dtype=torch.float).squeeze(-1)
        t_starttime=torch.tensor(t_starttime, dtype=torch.float)
        t_deadline=torch.tensor(t_deadline, dtype=torch.float)

        finnal_time = t_deadline.max().int()
        count_arrived = torch.bincount(t_starttime.squeeze(-1).short())
        count_dead = torch.bincount(t_deadline.squeeze(-1).short())
        prev_time = count_arrived.size(-1)

        pay1=[]
        for j in range(finnal_time):
            pay1.append([0])
        pay = []
        arrived_all_tasks = 0
        prev_arrived_tasks=0

        for i in range(count_arrived.size(0)):
            arrived_all_tasks+=count_arrived[i]
            cur_task = t_pay[prev_arrived_tasks:arrived_all_tasks]
            for j in range(finnal_time):
                if j<=i-1:
                    for k in range(cur_task.size(0)):
                        pay1[j].append(0)
                elif i-1 < j <= i+2:
                    for k in range(cur_task.size(0)):
                        pay1[j].append(cur_task[k])
                elif i+2 < j <= i + 5:
                    for k in range(cur_task.size(0)):
                        pay1[j].append( float(cur_task[k]-0.2*cur_task[k]*(j-i-2)))
                elif i + 5 < j:
                    for k in range(cur_task.size(0)):
                        pay1[j].append(0)
                # pay.append(pay1)
            prev_arrived_tasks=arrived_all_tasks.int()

        pay =  torch.tensor(pay1, dtype=torch.float)
        pay=pay[:, 1:]
        pay.shape


        case_info = {
            'depot': torch.FloatTensor(2).uniform_(0, 0),
            't_loc': torch.tensor(t_location, dtype=torch.float),
            'demand': (torch.FloatTensor(task_num).uniform_(0, 0).int() + 1).float(),
            't_start': torch.tensor(t_starttime, dtype=torch.float),
            't_deadline': torch.tensor(t_deadline, dtype=torch.float),
            't_pay': pay,
            # 'relation': torch.tensor(relation, dtype=torch.int32),
            'w_loc': torch.tensor(agents_location, dtype=torch.float),
            'w_capacity': torch.tensor(agents_capacity, dtype=torch.int),
            'w_start': torch.tensor(agents_starttime, dtype=torch.float),
            'w_deadline': torch.tensor(agents_deadline, dtype=torch.float),
            'w_speed': ((torch.randint(2, 3, (worker_num, 1)).to(torch.float)) / 10),
            'w_riato': torch.tensor(agents_riato, dtype=torch.float)
        }
        data.append(case_info)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, default='mrta', help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='mrta',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=100, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=2500,
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

    dataset = generate_mrta_data()

    save_dataset(dataset, datadir + "/" + "1my" + "_tasks_" + problem + ".pkl")

