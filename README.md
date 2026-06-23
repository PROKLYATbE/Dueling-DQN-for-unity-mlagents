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

## Common Trainer Configurations

| Setting | Description |
|---|---|
| `trainer_type` | Type of trainer used for training. For this plugin, the value must be `dueling_dqn`. |


## Dueling DQN Hyperparameters

| Setting | Default Value | Description |
|---|---|
| `hyperparameters -> learning_rate` | `0.0003` | Learning rate used by the optimizer during neural network updates. |
| `hyperparameters -> learning_rate_schedule` | `128` | Schedule for changing the learning rate during training. The `constant` value keeps the learning rate unchanged. |
| `hyperparameters -> batch_size` | `50000` | Number of samples taken from the replay buffer for one training update. |
| `hyperparameters -> buffer_size` | `0` | Maximum number of transitions stored in the replay buffer. |
| `hyperparameters -> tau` | `0.005` | Coefficient used for soft updating the target network. A smaller value makes target updates smoother. |
| `hyperparameters -> steps_per_update` | `1` | Controls how often the neural network is updated relative to the number of environment steps. |
| `hyperparameters -> exploration_schedule` | `false` | Schedule used for changing epsilon during epsilon-greedy exploration. |
| `hyperparameters -> exploration_initial_eps` | `linear` | Initial epsilon value at the beginning of training. A higher value increases random exploration. |
| `hyperparameters -> exploration_final_eps` | `0.1` | Final epsilon value after exploration decay. |
| `hyperparameters -> exploration_steps` | `0.05` | Number of steps over which epsilon decreases from `exploration_initial_eps` to `exploration_final_eps`. |

For common Unity ML-Agents trainer settings, network settings, reward signals, and other standard YAML parameters, refer to the official ML-Agents documentation:

https://docs.unity3d.com/Packages/com.unity.ml-agents%404.0/manual/Training-Configuration-File.html