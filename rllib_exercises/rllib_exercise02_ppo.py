from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print


# Start up Ray. This must be done before we instantiate any RL agents.

# In[ ]:


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


# ## Visualize results with TensorBoard
# 
# **EXERCISE**: Finally, you can visualize your training results using TensorBoard. To do this, open a new terminal in Jupyter lab using the "+" button, and run:
#     
# `$ tensorboard --logdir=~/ray_results --host=0.0.0.0`
# 
# And open your browser to the address printed (or change the current URL to go to port 6006). Check the "episode_reward_mean" learning curve of the PPO agent. Toggle the horizontal axis between both the "STEPS" and "RELATIVE" view to compare efficiency in number of timesteps vs real time time.
# 
# Note that TensorBoard will not work in Binder.
