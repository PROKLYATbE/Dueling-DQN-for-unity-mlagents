from setuptools import setup, find_packages

setup(
    name="mlagents-dueling-dqn-plugin",
    version="0.0.1",
    description="Dueling DQN trainer plugin for Unity ML-Agents",
    packages=find_packages(),
    entry_points={
        "mlagents.trainer_type": [
            "dueling_dqn=mlagents_dueling_dqn_plugin.dueling_dqn.dueling_dqn_trainer:get_type_and_setting",
        ]
    },
)
