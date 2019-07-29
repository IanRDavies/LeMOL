# Learning to Model Opponent Learning for Opponent Modelling 

This is an initial release of the base models for work brining ideas from meta-learning into multi-agent reinforcement learning to aid opponent model. This repo is a work in progress and will be periodically updated as bugs are fixed and new architectures (currently in testing) are released.

This is the code for implementing what we term LeMOL algorithms. 
The code is modified from https://github.com/dadadidodi/m3ddpg

For Multi-Agent Particle Environments (MPE) installation, please refer to https://github.com/openai/multiagent-particle-envs

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple_push``

- The code is currently set up for two-player environments only.

### Command-line Options

#### Environment Options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max_episode_len` maximum length of each episode for the environment (default: `25`)

- `--num_episodes` total number of training episodes (default: `10000`)

- `--num_adversaries`: number of adversaries in the environment (default: `0`)

- `--good_policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"maddpg"`; options: {`"mmmaddpg"`, `"maddpg"`, `"lemol"`})

- `--bad_policy`: algorithm used for the adversary policies in the environment
(default: `"lemol"`; options: {`"mmmaddpg"`, `"maddpg"`, `"lemol"`})

- `--discrete_actions`: flags that the game should run with discrete (i.e. one-hot) actions.
(default: `False`)

##### UAV Environment Under Development

#### Core Training Parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch_size`: batch size (default: `1024`)

- `--num_units`: number of units in the MLP (default: `64`)

- `--adv_eps`: adversarial rate against competitors (for M3DDPG only, default: `1e-3`)

- `--adv_eps_s`: adversarial rate against collaborators (for M3DDPG only, default: `1e-5`)

- `--agent_update_freq`: The number of time steps between agents updating their policies and Q networks (default: `1`)

- `--polyak`: The value used for slowly updating target networks. Updates are such that the parameters are `target_params_new = (1 - polyak) * target_params_old + polyak * params` (default: `1e-4`)

- `--replay_buffer_capacity`: The size of the replay buffer in time steps. Once this capacity is reached new experience overwrites old (default: `1e6`)

#### LeMOL Parameters

- `--omlr`: Learning rate for the opponent model (default: `1e-3`)

- `--lstm_hidden_dim`: The number of units in the LSTM used to model opponent learning (default: `64`)

- `--lemol_save_path`: The path in which LeMOL's experience is saved to later be used for opponent model training

- `--chunk_length`: The number of time steps to pass to the opponent modelling LSTM at a time during training (the point at which back propagation is truncated, default: `64`)

- `--num_traj_to_train`: The number of experience trajectories to train the opponent model on in each iteration (default: `8`)

- `--traj_per_iter`: The number of trajectories to play between a pair of agents before LeMOL's opponent model is trained (default: `3`)

- `--om_training_iters`: The number of iterations of opponent model training to perform after each set of play outs for the current pair of agents (default: `5`)

- `--n_opp_cycles`: The number of cycles which where one cycle consists of a play out trajectory followed by opponent model training (default: `10`)