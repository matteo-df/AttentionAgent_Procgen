# Self-Attention Agent Plays Dodgeball

In this work, the [Neuroevolution of Self-Interpretable Agents](https://attentionagent.github.io/) idea is implemented to play Dodgeball, one the [OpenAI PROCGEN](https://github.com/openai/procgen) games. The goal is to show that agents capable of seeing only a portion of the pixel inputs from the environment can solve the tasks with simpler models, and also achieve better generalization.

![Attention_Agent_Dodgeball](https://github.com/zeeke22/AttentionAgent_Procgen/blob/main/log/gif/poscol_rew20.gif)

Our agent receives visual input as a stream of 96x96px RGB images. Each image frame is passed through a self-attention bottleneck module, responsible for selecting K=10 patches (highlighted in the gif above).

The agent segment the 96x96px image inputs of each game step into N patches and then creates an Attention matrix. For each patch, it obtains a vector representing the corresponding importance, in order to select K patches of the highest importance.
Features from these K patches (such as location) are then used by a controller (LSTM module) to produce the agent action.
The parameters of the self-attention module and the LSTM are trained using [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](https://en.wikipedia.org/wiki/CMA-ES), through the [pycma](https://github.com/CMA-ES/pycma) package.

## Requirements

* **Packages**: run the command `pip3 install -r requirements.txt` to install the required packages.
* **Modified Dodgeball Game**: in this work it was used a version of the Dodgeball game without walls. To reproduce the result it is necessary to install from source Procgen following [the instructions on the official Github Repo](https://github.com/openai/procgen#install-from-source) and modify the `procgen/src/games/dodgeball.cpp`, setting `num_iterations = 0;` in the `game_reset()` method.


## Evaluate pre-trained models

Two pre-trained models are present in the repository `pretrained/`:
* pos: the agent knows only the center position of the 'important' patches
* poscol:  the agent knows the center position and the color of a random pixel of the 'important' patches

```
# Evaluate the 'pos' agent results (such as number of enemies defeated or win rate) for 100 episodes.
python3 test_agent.py agent=pos n_episodes=100

# Show on screen the 'poscol' agent playing dodgeball for 20 episodes.
python3 test_agent.py agent=poscol n_episodes=20 render=True

# Save a gif in /log/gif/ of the agent playing the best episode out of 10, highlighting at each step the top k attention patches .
python3 test_agent.py agent=pos save_gif_with_attention_patches=True
```

## Training

To train on a local machine, run the following command:
```
# Train 'pos' configuration locally.
python3 train_agent.py agent=pos
```
A training checkpoint is saved every 50 steps (overwriting the previous one) in `log/cmaes_checkpoint/`. Restarting a train with the same `agent_name` in the config file will load automatically the most recent checkpoint and continue the training.

Modifying the `.yaml` files it is possible to change the training parameters, such as the CMAES population size, the number of hidden layers of the LSTM, and so on.  

Training info are logged using [Weights and Biases](https://wandb.ai/), configured in `conf/log/log.yaml`. To use other logging framework, the code must be changed manually.
