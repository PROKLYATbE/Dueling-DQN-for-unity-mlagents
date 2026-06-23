# Dueling-DQN-for-unity-mlagents
Custom implementation of a Dueling DQN trainer for Unity ML-Agents.

## 1. Requirement

Before installation, prepare the working environment:

- Windows 10 or newer;
- Python 3.10;
- pip 23.0 or newer;
- Unity with the ML-Agents package installed;
- the installed Python package mlagents;

## 2. Creating a Virtual Environment

https://docs.unity3d.com/Packages/com.unity.ml-agents%404.0/manual/Using-Virtual-Environment.html

## 3. Installing Unity ML-Agents

https://docs.unity3d.com/Packages/com.unity.ml-agents%404.0/manual/Installation.html

## 4. Installing the Dueling DQN Plugin

Download the repository with the Dueling DQN trainer source code.

You can clone the repository using Git:

```bibtex
git clone https://github.com/PROKLYATbE/Dueling-DQN-for-unity-mlagents.git
```

Or download it manually from GitHub as a ZIP archive and extract it to any convenient folder.

Before installing the plugin, activate the Python virtual environment:

```bibtex
NameOfYourVirtualEnv\Scripts\activate
```

Then install the plugin using pip:

```bibtex
pip install PathToInstalledPlugin
```

After installation, the dueling_dqn trainer should become available in Unity ML-Agents.

## 5. YAML Configuration

The Dueling DQN trainer is configured through a standard Unity ML-Agents YAML configuration file.

> **Important:** this trainer supports only **discrete action spaces**.  
> Continuous actions are not supported. In Unity, the agent must use `Discrete` actions in the `Behavior Parameters` component.

The behavior name in the YAML file must match the `Behavior Name` field in the Unity `Behavior Parameters` component.

For example, if the YAML file contains:

```bibtex
behaviors:
  BasicDQN:
```

then the Unity `Behavior Name` must also be:

```bibtex
BasicDQN
```

### Main Parameters

- `trainer_type` — specifies the trainer type. For this plugin, it must be set to `dueling_dqn`.
- `max_steps` — maximum number of environment steps used for training.
- `time_horizon` — number of steps collected before the trajectory is processed by the trainer.
- `summary_freq` — frequency of writing training statistics for TensorBoard.
- `keep_checkpoints` — number of saved model checkpoints.

### Hyperparameters

- `learning_rate` — learning rate used by the optimizer.
- `learning_rate_schedule` — learning rate schedule. The current configuration uses a constant learning rate.
- `batch_size` — number of samples used in one training update.
- `buffer_size` — maximum number of transitions stored in the replay buffer.
- `tau` — coefficient for soft updating the target network.
- `steps_per_update` — controls how often the network is updated during training.
- `exploration_schedule` — exploration schedule for epsilon-greedy action selection.
- `exploration_initial_eps` — initial epsilon value used at the beginning of training.
- `exploration_final_eps` — final epsilon value used after exploration decay.
- `exploration_steps` — number of steps over which epsilon is decreased from the initial value to the final value.

### Network Settings

- `normalize` — enables or disables observation normalization.
- `hidden_units` — number of neurons in each hidden layer.
- `num_layers` — number of hidden layers in the neural network.
- `vis_encode_type` — visual encoder type used by ML-Agents.

### Reward Signal

- `gamma` — discount factor for future rewards.
- `strength` — multiplier for the extrinsic reward signal.