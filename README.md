# Learning to Model Opponent Learning for Opponent Modelling 

This is the release of models and baselines for work bringing ideas from meta-learning into multi-agent reinforcement learning to aid opponent modelling.
This repo is periodically reviewed and updated as bugs are fixed and new architectures (currently in testing) are released.

This is the code for implementing what we term Learning to Model Opponent Learning (LeMOL) algorithms. 
The code is modified from https://github.com/dadadidodi/m3ddpg

For Multi-Agent Particle Environments (MPE) installation, please refer to https://github.com/openai/multiagent-particle-envs

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py``

- The code is currently set up for two-player environments only.

### Command-line Options

#### General Environment Options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple_push"`)

#### Particle Environment Options

- `--max_episode_len`: maximum length of each episode for the environment (default: `25`)

- `--num_episodes`: total number of training episodes (default: `11024`)

- `--num_adversaries`: number of adversaries in the environment (default: `1`)

- `--good_policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"maddpg"`; options: {`"mmmaddpg"`, `"maddpg"`, `"ddpg"`, `"ppo"`, `"lemol"`})

- `--bad_policy`: algorithm used for the adversary policies in the environment
(default: `"lemol"`; options: {`"mmmaddpg"`, `"maddpg"`, `"ddpg"`, `"ppo"`, `"lemol"`})

- `--discrete_actions`: flags that the game should run with discrete (i.e. one-hot) actions.
(default: `False`)

##### UAV Environment Options

- `--uav_grid_height`: number of rows in the UAV gridworld (default: `3`)

- `--uav_grid_width`: number of columns in the UAV gridworld (default: `3`)

- `--uav_seeker_noise`: probability of UAV being misinformed when listening (default: `0.3`)

- `--uav_listen_imp`: reduction in observation noise when UAV agents choose to stop and listen (default: `0.1`)

- `--uav_reward_scale`: scaling of binary rewards for UAV environment (default: `1.0`)

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

- `--episode_embedding_dim`: The dimension of the episode representation used to update the opponent model in LeMOL-EP. Must be divisible by 2 (default: `128`)

- `--block_processing`: Turns on episode processing for LeMOL

- `--recurrent_om_prediction`: Provides LeMOL with a recurrent in-episode opponent model

- `--in_ep_lstm_dim`: The dimension of the state of the within-episode LSTM for LeMOL-EP (default: `32`)

- `--decentralise_lemol_obs`: Provides Q function for LeMOL with LeMOL agent's observation only

- `--fully_decentralise_lemol`: Switches LeMOL training to become fully decentralised

### PPO Parameters

- `--ppo_lambda`: Lambda value for GAE-Lambda in PPO (default `0.97`)

- `--ppo_pi_lr`: Learning rate for PPO policy optimiser (default: `3e-4`)

- `--ppo_vf_lr`: Learning rate for PPO value function optimiser (default: `1e-3`)

- `--ppo_target_kl`: The KL divergence between old and new policies after an update as used in early stopping (default: `0.01`)

- `--ppo_clip_ratio`: The clipping point which determines how far the new policy can stray from the old (default: `0.2`)

- `--ppo_train_cycles`: The maximum number of optimisation cycles to run for PPO within each training step (default: `80`)

- `--verbose_ppo`: Turns on verbose logging for PPO