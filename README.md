#  Task Planning in Preference-aware Spatial Crowdsourcing with Transformer based Reinforcement Learning

This repository contains code for implementing a Transformer-based reinforcement learning framework for learning to solve task planning (TOTP) in spatial crowdsourcing with bilateral preferences. Training with REINFORCE with greedy rollout baseline. Environment: The TPPSC environment can be initialized using the TPPSC class in problems/tppsc/problem_tppsc.py. All state space variables and functions required to update the state space are implemented using the StateTPPSC class in problems/tppsc/state_tppsc.py. training.py implements the REINFORCE algorithm.

##  Dependencies
Python>=3.8

NumPy

SciPy

PyTorch>=1.7

tqdm

tensorboard_logger

## How to use the code:
*Training:
    To start training run the run.py file. The options.py can be used for different environement settings such as number of tasks, number of workers and other parameters.
    The trained models for each epoch will be stored in a directory named 'outputs'
    We recommend using a GPU for training.

*Evaluation:
    The datasets for testing can be found inside the directory named 'data'. To evaluate a model, you can add the --eval-only flag to run.py, or use eval.py, which will additionally measure timing and save the results.

*Generating data:
    More test data with varying number of tasks and workers can be generated using the script generate_data_normal.py, generate_syn_data.py and generate_real_data.py.

##  Contact
If you have any questions or concerns, please raise an issue or email: 201810102795@mail.scut.edu.cn

This repository includes adaptions of the following repositories as baselines:

https://github.com/wouterkool/attention-learn-to-route

https://github.com/Demon0312/HCVRP_DRL

https://github.com/adamslab-ub/CapAM-MRTA

https://github.com/BUAA-BDA/SpatialCrowdsourcing-GOMA#usage-of-the-algorithms
