#  A Reinforced Spatio-temporal Learning Framework for Multi-Task Allocation in Spatial Crowdsourcing

This repository contains code for implementing a reinforced spatio-temporal learning framework for Multi-Task Allocation in Spatial Crowdsourcing (MTASC). Training with REINFORCE with greedy rollout baseline. Environment: The MTASC environment can be initialized using the MTASC class in problems/mtasc/problem_mtasc.py. All state space variables and functions required to update the state space are implemented using the StateTPPSC class in problems/mtasc/state_mtasc.py. training.py implements the REINFORCE algorithm. 
## Usage

##  Dependencies
Python>=3.8  
NumPy  
SciPy  
PyTorch>=1.7  
tqdm  
tensorboard_logger  

## How to use the code:
**Training**:  
    To start training run the run.py file. The options.py can be used for different environement settings such as number of tasks, number of workers and other parameters.
    The trained models for each epoch will be stored in a directory named 'outputs'
    We recommend using a GPU for training.

**Evaluation**:  
    The datasets for testing can be found inside the directory named 'data'. To evaluate a model, you can add the --eval-only flag to run.py, or use eval.py, which will additionally measure timing and save the results.

**Generating data**:  
    More test data with varying number of tasks and workers can be generated using the script generate_data_normal.py, generate_syn_data.py and generate_real_data.py.

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).
