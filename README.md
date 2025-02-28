# ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs

## [Paper](https://arxiv.org/abs/2502.17529)
This is the official implementation of the paper ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs.

## Introduction
we introduce LLMs into multi-lane convoy control for the first time to enhance the convoy's adaptive capabilities and cooperation in complex scenarios. The main role of the LLM is to provide effective decision-making support for each vehicle in the convoy through efficient reasoning and generation capabilities. At the same time, we integrate a staggered formation control strategy based on locally dynamic distributed graphs, ensuring that the convoy maintains formation while exhibiting both flexibility and stability.

![overview](Asserts/overview.png)
The overall framework of the multi-lane convoy formation control method. It contains a total of five modules: environment, reasoning, shared memory, trajectory planning, and control. The reasoning module obtains the perception results from the environment and generates the target lanes and target speeds of the vehicles, the trajectory planning module obtains the target values and generates the trajectories of each vehicle in the convoy, and finally the control module outputs the acceleration and steering angle commands, which are then applied to the environment.

![reasoningModule](Asserts/ReasoningModule.png)
A case of the Reasoning module process. This simple obstacle avoidance scenario illustrates how the reasoning module collects information from the ego vehicle, environment vehicles, and neighbors, then generates a scene description for decision-making by the large model. In the figure, veh7 changes lanes to the right due to a slow vehicle ahead, while veh3 outputs an IDLE decision to follow the neighboring vehicle in the same lane.

# Gettting Started

## Install dependencies
- Create a new conda virtual env
'''
conda env create -f environment.yml
conda activate Convoy
'''
