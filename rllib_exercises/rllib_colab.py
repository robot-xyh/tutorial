#!/usr/bin/env python
# coding: utf-8

# # Install Dependencies
# 
# If you are running on Google Colab, you need to install the necessary dependencies before beginning the exercise.

# In[ ]:


print('NOTE: Intentionally crashing session to use the newly installed library.\n')

get_ipython().system('pip uninstall -y pyarrow')
get_ipython().system('pip install ray[debug]==0.7.5')
get_ipython().system('pip install bs4')

get_ipython().system('git clone https://github.com/ray-project/tutorial || true')
from tutorial.rllib_exercises import test_exercises

print("Successfully installed all the dependencies!")

# A hack to force the runtime to restart, needed to include the above dependencies.
import os
os._exit(0)


# # RL Exercise - Markov Decision Processes
# 
# **GOAL:** The goal of the exercise is to introduce the Markov Decision Process abstraction and to show how to use Markov Decision Processes in Python.
# 
# **The key abstraction in reinforcement learning is the Markov decision process (MDP).** An MDP models sequential interactions with an external environment. It consists of the following:
# - a **state space**
# - a set of **actions**
# - a **transition function** which describes the probability of being in a state $s'$ at time $t+1$ given that the MDP was in state $s$ at time $t$ and action $a$ was taken
# - a **reward function**, which determines the reward received at time $t$
# - a **discount factor** $\gamma$
# 
# More details are available [here](https://en.wikipedia.org/wiki/Markov_decision_process).
# 
# **NOTE:** Reinforcement learning algorithms are often applied to problems that don't strictly fit into the MDP framework. In particular, situations in which the state of the environment is not fully observed lead to violations of the MDP assumption. Nevertheless, RL algorithms can be applied anyway.
# 
# ## Policies
# 
# A **policy** is a function that takes in a **state** and returns an **action**. A policy may be stochastic (i.e., it may sample from a probability distribution) or it can be deterministic.
# 
# The **goal of reinforcement learning** is to learn a **policy** for maximizing the cumulative reward in an MDP. That is, we wish to find a policy $\pi$ which solves the following optimization problem
# 
# \begin{equation}
# \arg\max_{\pi} \sum_{t=1}^T \gamma^t R_t(\pi),
# \end{equation}
# 
# where $T$ is the number of steps taken in the MDP (this is a random variable and may depend on $\pi$) and $R_t$ is the reward received at time $t$ (also a random variable which depends on $\pi$).
# 
# A number of algorithms are available for solving reinforcement learning problems. Several of the most widely known are [value iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration), [policy iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Policy_iteration), and [Q learning](https://en.wikipedia.org/wiki/Q-learning).
# 
# ## RL in Python
# 
# The `gym` Python module provides MDP interfaces to a variety of simulators. For example, the CartPole environment interfaces with a simple simulator which simulates the physics of balancing a pole on a cart. The CartPole problem is described at https://gym.openai.com/envs/CartPole-v0. This example fits into the MDP framework as follows.
# - The **state** consists of the position and velocity of the cart as well as the angle and angular velocity of the pole that is balancing on the cart.
# - The **actions** are to decrease or increase the cart's velocity by one unit.
# - The **transition function** is deterministic and is determined by simulating physical laws.
# - The **reward function** is a constant 1 as long as the pole is upright, and 0 once the pole has fallen over. Therefore, maximizing the reward means balancing the pole for as long as possible.
# - The **discount factor** in this case can be taken to be 1.
# 
# More information about the `gym` Python module is available at https://gym.openai.com/.

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


# The code below illustrates how to create and manipulate MDPs in Python. An MDP can be created by calling `gym.make`. Gym environments are identified by names like `CartPole-v0`. A **catalog of built-in environments** can be found at https://gym.openai.com/envs.

# In[ ]:


env = gym.make('CartPole-v0')
print('Created env:', env)


# Reset the state of the MDP by calling `env.reset()`. This call returns the initial state of the MDP.

# In[ ]:


state = env.reset()
print('The starting state is:', state)


# The `env.step` method takes an action (in the case of the CartPole environment, the appropriate actions are 0 or 1, for moving left or right). It returns a tuple of four things:
# 1. the new state of the environment
# 2. a reward
# 3. a boolean indicating whether the simulation has finished
# 4. a dictionary of miscellaneous extra information

# In[ ]:


# Simulate taking an action in the environment. Appropriate actions for
# the CartPole environment are 0 and 1 (for moving left and right).
action = 0
state, reward, done, info = env.step(action)
print(state, reward, done, info)


# A **rollout** is a simulation of a policy in an environment. It alternates between choosing actions based (using some policy) and taking those actions in the environment.
# 
# The code below performs a rollout in a given environment. It takes **random actions** until the simulation has finished and returns the cumulative reward.

# In[ ]:


def random_rollout(env):
    state = env.reset()
    
    done = False
    cumulative_reward = 0

    # Keep looping as long as the simulation has not finished.
    while not done:
        # Choose a random action (either 0 or 1).
        action = np.random.choice([0, 1])
        
        # Take the action in the environment.
        state, reward, done, _ = env.step(action)
        
        # Update the cumulative reward.
        cumulative_reward += reward
    
    # Return the cumulative reward.
    return cumulative_reward
    
reward = random_rollout(env)
print(reward)
reward = random_rollout(env)
print(reward)


# **EXERCISE:** Finish implementing the `rollout_policy` function below, which should take an environment *and* a policy. The *policy* is a function that takes in a *state* and returns an *action*. The main difference is that instead of choosing a **random action**, the action should be chosen **with the policy** (as a function of the state).

# In[ ]:


def rollout_policy(env, policy):
    state = env.reset()
    
    done = False
    cumulative_reward = 0

    # EXERCISE: Fill out this function by copying the 'random_rollout' function
    # and then modifying it to choose the action using the policy.
    raise NotImplementedError

    # Return the cumulative reward.
    return cumulative_reward

def sample_policy1(state):
    return 0 if state[0] < 0 else 1

def sample_policy2(state):
    return 1 if state[0] < 0 else 0

reward1 = np.mean([rollout_policy(env, sample_policy1) for _ in range(100)])
reward2 = np.mean([rollout_policy(env, sample_policy2) for _ in range(100)])

print('The first sample policy got an average reward of {}.'.format(reward1))
print('The second sample policy got an average reward of {}.'.format(reward2))

assert 5 < reward1 < 15, ('Make sure that rollout_policy computes the action '
                          'by applying the policy to the state.')
assert 25 < reward2 < 35, ('Make sure that rollout_policy computes the action '
                           'by applying the policy to the state.')


# # RLlib Exercise 1 - Proximal Policy Optimization
# 
# **GOAL:** The goal of this exercise is to demonstrate how to use the proximal policy optimization (PPO) algorithm.
# 
# To understand how to use **RLlib**, see the documentation at http://rllib.io.
# 
# PPO is described in detail in https://arxiv.org/abs/1707.06347. It is a variant of Trust Region Policy Optimization (TRPO) described in https://arxiv.org/abs/1502.05477
# 
# PPO works in two phases. In one phase, a large number of rollouts are performed (in parallel). The rollouts are then aggregated on the driver and a surrogate optimization objective is defined based on those rollouts. We then use SGD to find the policy that maximizes that objective with a penalty term for diverging too much from the current policy.
# 
# ![ppo](https://raw.githubusercontent.com/ucbrise/risecamp/risecamp2018/ray/tutorial/rllib_exercises/ppo.png)
# 
# **NOTE:** The SGD optimization step is best performed in a data-parallel manner over multiple GPUs. This is exposed through the `num_gpus` field of the `config` dictionary (for this to work, you must be using a machine that has GPUs).

# In[ ]:


# Be sure to install the latest version of RLlib.
get_ipython().system(' pip install -U ray[rllib]')


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print


# In[ ]:


# Start up Ray. This must be done before we instantiate any RL agents.
ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False)


# Instantiate a PPOTrainer object. We pass in a config object that specifies how the network and training procedure should be configured. Some of the parameters are the following.
# 
# - `num_workers` is the number of actors that the agent will create. This determines the degree of parallelism that will be used.
# - `num_sgd_iter` is the number of epochs of SGD (passes through the data) that will be used to optimize the PPO surrogate objective at each iteration of PPO.
# - `sgd_minibatch_size` is the SGD batch size that will be used to optimize the PPO surrogate objective.
# - `model` contains a dictionary of parameters describing the neural net used to parameterize the policy. The `fcnet_hiddens` parameter is a list of the sizes of the hidden layers.

# In[ ]:


config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed

agent = PPOTrainer(config, 'CartPole-v0')


# Train the policy on the `CartPole-v0` environment for 2 steps. The CartPole problem is described at https://gym.openai.com/envs/CartPole-v0.
# 
# **EXERCISE:** Inspect how well the policy is doing by looking for the lines that say something like
# 
# ```
# episode_len_mean: 22.262569832402235
# episode_reward_mean: 22.262569832402235
# ```
# 
# This indicates how much reward the policy is receiving and how many time steps of the environment the policy ran. The maximum possible reward for this problem is 200. The reward and trajectory length are very close because the agent receives a reward of one for every time step that it survives (however, that is specific to this environment).

# In[ ]:


for i in range(2):
    result = agent.train()
    print(pretty_print(result))


# **EXERCISE:** The current network and training configuration are too large and heavy-duty for a simple problem like CartPole. Modify the configuration to use a smaller network and to speed up the optimization of the surrogate objective (fewer SGD iterations and a larger batch size should help).

# In[ ]:


config = DEFAULT_CONFIG.copy()
config['num_workers'] = 3
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 0

agent = PPOTrainer(config, 'CartPole-v0')


# **EXERCISE:** Train the agent and try to get a reward of 200. If it's training too slowly you may need to modify the config above to use fewer hidden units, a larger `sgd_minibatch_size`, a smaller `num_sgd_iter`, or a larger `num_workers`.
# 
# This should take around 20 or 30 training iterations.

# In[ ]:


for i in range(2):
    result = agent.train()
    print(pretty_print(result))


# Checkpoint the current model. The call to `agent.save()` returns the path to the checkpointed model and can be used later to restore the model.

# In[ ]:


checkpoint_path = agent.save()
print(checkpoint_path)


# Now let's use the trained policy to make predictions.
# 
# **NOTE:** Here we are loading the trained policy in the same process, but in practice, this would often be done in a different process (probably on a different machine).

# In[ ]:


trained_config = config.copy()

test_agent = PPOTrainer(trained_config, 'CartPole-v0')
test_agent.restore(checkpoint_path)


# Now use the trained policy to act in an environment. The key line is the call to `test_agent.compute_action(state)` which uses the trained policy to choose an action.
# 
# **EXERCISE:** Verify that the reward received roughly matches up with the reward printed in the training logs.

# In[ ]:


env = gym.make('CartPole-v0')
state = env.reset()
done = False
cumulative_reward = 0

while not done:
    action = test_agent.compute_action(state)
    state, reward, done, _ = env.step(action)
    cumulative_reward += reward

print(cumulative_reward)


# # RLlib Exercise 2 - Custom Environments and Reward Shaping
# 
# **GOAL:** The goal of this exercise is to demonstrate how to adapt your own problem to use RLlib.
# 
# To understand how to use **RLlib**, see the documentation at http://rllib.io.
# 
# RLlib is not only easy to use in simulated benchmarks but also in the real-world. Here, we will cover two important concepts: how to create your own Markov Decision Process abstraction, and how to shape the reward of your environment so make your agent more effective. 

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import spaces
import numpy as np
from tutorial.rllib_exercises import test_exercises

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

ray.init(ignore_reinit_error=True, log_to_driver=False)


# ## 1. Different Spaces
# 
# The first thing to do when formulating an RL problem is to specify the dimensions of your observation space and action space. Abstractions for these are provided in ``gym``. 
# 
# ### **Exercise 1:** Match different actions to their corresponding space.
# 
# The purpose of this exercise is to familiarize you with different Gym spaces. For example:
# 
#     discrete = spaces.Discrete(10)
#     print("Random sample of this space: ", [discrete.sample() for i in range(4)])
# 
# Use `help(spaces)` or `help([specific space])` (i.e., `help(spaces.Discrete)`) for more info.

# In[ ]:


action_space_map = {
    "discrete_10": spaces.Discrete(10),
    "box_1": spaces.Box(0, 1, shape=(1,)),
    "box_3x1": spaces.Box(-2, 2, shape=(3, 1)),
    "multi_discrete": spaces.MultiDiscrete([ 5, 2, 2, 4 ])
}

action_space_jumble = {
    "discrete_10": 1,
    "CHANGE_ME": np.array([0, 0, 0, 2]),
    "CHANGE_ME": np.array([[-1.2657754], [-1.6528835], [ 0.5982418]]),
    "CHANGE_ME": np.array([0.89089584]),
}


for space_id, state in action_space_jumble.items():
    assert action_space_map[space_id].contains(state), (
        "Looks like {} to {} is matched incorrectly.".format(space_id, state))
    
print("Success!")


# ## **Exercise 2**: Setting up a custom environment with rewards
# 
# We'll setup an `n-Chain` environment, which presents moves along a linear chain of states, with two actions:
# 
#      (0) forward, which moves along the chain but returns no reward
#      (1) backward, which returns to the beginning and has a small reward
# 
# The end of the chain, however, presents a large reward, and by moving 'forward', at the end of the chain this large reward can be repeated.
# 
# #### Step 1: Implement ``ChainEnv._setup_spaces``
# 
# We'll use a `spaces.Discrete` action space and observation space. Implement `ChainEnv._setup_spaces` so that `self.action_space` and `self.obseration_space` are proper gym spaces.
#   
# 1. Observation space is an integer in ``[0 to n-1]``.
# 2. Action space is an integer in ``[0, 1]``.
# 
# For example:
# 
# ```python
#     self.action_space = spaces.Discrete(2)
#     self.observation_space = ...
# ```
# 
# You should see a message indicating tests passing when done correctly!
# 
# #### Step 2: Implement a reward function.
# 
# When `env.step` is called, it returns a tuple of ``(state, reward, done, info)``. Right now, the reward is always 0. 
# 
# Implement it so that 
# 
# 1. ``action == 1`` will return `self.small_reward`.
# 2. ``action == 0`` will return 0 if `self.state < self.n - 1`.
# 3. ``action == 0`` will return `self.large_reward` if `self.state == self.n - 1`.
# 
# You should see a message indicating tests passing when done correctly. 

# In[ ]:


class ChainEnv(gym.Env):
    
    def __init__(self, env_config = None):
        env_config = env_config or {}
        self.n = env_config.get("n", 20)
        self.small_reward = env_config.get("small", 2)  # payout for 'backwards' action
        self.large_reward = env_config.get("large", 10)  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self._horizon = self.n
        self._counter = 0  # For terminating the episode
        self._setup_spaces()
    
    def _setup_spaces(self):
        ##############
        # TODO: Implement this so that it passes tests
        self.action_space = None
        self.observation_space = None
        ##############

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # 'backwards': go back to the beginning, get small reward
            ##############
            # TODO 2: Implement this so that it passes tests
            reward = -1
            ##############
            self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            ##############
            # TODO 2: Implement this so that it passes tests
            reward = -1
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            ##############
            # TODO 2: Implement this so that it passes tests
            reward = -1
            ##############
        self._counter += 1
        done = self._counter >= self._horizon
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        self._counter = 0
        return self.state
    
# Tests here:
test_exercises.test_chain_env_spaces(ChainEnv)
test_exercises.test_chain_env_reward(ChainEnv)


# ### Now let's train a policy on the environment and evaluate this policy on our environment.
# 
# You'll see that despite an extremely high reward, the policy has barely explored the state space.

# In[ ]:


trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 400
trainer_config["sgd_minibatch_size"] = 64
trainer_config["num_sgd_iter"] = 10


# In[ ]:


trainer = PPOTrainer(trainer_config, ChainEnv);
for i in range(20):
    print("Training iteration {}...".format(i))
    trainer.train()


# In[ ]:


env = ChainEnv({})
state = env.reset()

done = False
max_state = -1
cumulative_reward = 0

while not done:
    action = trainer.compute_action(state)
    state, reward, done, results = env.step(action)
    max_state = max(max_state, state)
    cumulative_reward += reward

print("Cumulative reward you've received is: {}. Congratulations!".format(cumulative_reward))
print("Max state you've visited is: {}. This is out of {} states.".format(max_state, env.n))


# ## Exercise 3: Shaping the reward to encourage proper behavior.
# 
# You'll see that despite an extremely high reward, the policy has barely explored the state space. This is often the situation - where the reward designed to encourage a particular solution is suboptimal, and the behavior created is unintended.
# 
# #### Modify `ShapedChainEnv.step` to provide a reward that encourages the policy to traverse the chain (not just stick to 0). Do not change the behavior of the environment (the action -> state behavior should be the same).
# 
# You can change the reward to be whatever you wish.

# In[ ]:


class ShapedChainEnv(ChainEnv):
    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # 'backwards': go back to the beginning
            reward = -1
            self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = -1
            self.state += 1
        else:  # 'forwards': stay at the end of the chain
            reward = -1
        self._counter += 1
        done = self._counter >= self._horizon
        return self.state, reward, done, {}
    
test_exercises.test_chain_env_behavior(ShapedChainEnv)


# ### Evaluate `ShapedChainEnv` by running the cell below.
# 
# This trains PPO on the new env and counts the number of states seen.

# In[ ]:


trainer = PPOTrainer(trainer_config, ShapedChainEnv);
for i in range(20):
    print("Training iteration {}...".format(i))
    trainer.train()

env = ShapedChainEnv({})

max_states = []

for i in range(5):
    state = env.reset()
    done = False
    max_state = -1
    cumulative_reward = 0
    while not done:
        action = trainer.compute_action(state)
        state, reward, done, results = env.step(action)
        max_state = max(max_state, state)
        cumulative_reward += reward
    max_states += [max_state]

print("Cumulative reward you've received is: {}!".format(cumulative_reward))
print("Max state you've visited is: {}. This is out of {} states.".format(np.mean(max_state), env.n))
assert (env.n - np.mean(max_state)) / env.n < 0.2, "This policy did not traverse many states."


# In[ ]:




