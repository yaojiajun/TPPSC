#  Two-sided Online Task Planning in Spatial Crowdsourcing with Graph Transformer based Reinforcement Learning
Graph Transformer based model for learning to solve the Two-sided Online Task Planning in Spatial Crowdsourcing (TOTP) with max-sum objective. Training with REINFORCE with greedy rollout baseline.

##  Dependencies
* Python>=3.8
* NumPy
* SciPy
* PyTorch>=1.7
* tqdm
* tensorboard_logger

## How to use the code
Training:
    To start training run the run.py file. The options.py can be used for different environement settings such as number of tasks, number of workers and other parameters.
    The trained models for each epoch will be stored in a directory named 'outputs'
    We recommend using a GPU for training.

Evaluation:
    The datasets for testing can be found inside the directory named 'data'. To evaluate a model, you can add the --eval-only flag to run.py, or use eval.py, which will additionally measure timing and save the results.

Test data generation:
    More test data with varying number of tasks and workers can be generated using the script generate_data_normal.py, generate_data_rand.py and generate_data_real.py.

##  Contact
If you have any questions or concerns, please raise an issue or email: 201810102795@mail.scut.edu.cn

This repository includes adaptions of the following repositories as baselines:
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/Demon0312/HCVRP_DRL
* https://github.com/adamslab-ub/CapAM-MRTA
* https://github.com/BUAA-BDA/SpatialCrowdsourcing-GOMA#usage-of-the-algorithms
