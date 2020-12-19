import gym
import torch
from cart_agent import CartAgent
from itertools import count
from torch.distributions import Bernoulli

# hyper parameter
batch_size = 5
num_epochs = 500
gamma = 0.99
learning_rate = 0.01

env = gym.make('CartPole-v0')
# space action
# print(env.action_space)
# print(env.observation_space)

cartAgent = CartAgent(lr=learning_rate, gamma=gamma)

for epoch in range(num_epochs):
    next_state = env.reset()
    env.render(mode='rgb_array')

    for t in count():
        state = torch.from_numpy(next_state).float()
        probs = cartAgent.act(state)
        m = Bernoulli(probs)
        action = m.sample()
        action = action.data.numpy().astype(int).item()
        next_state, reward, done, _ = env.step(action)
        env.render(mode='rgb_array')

        if done:
            reward = 0

        cartAgent.memorize(state, action, reward)

        if done:
            print('Episode {}: durations {}'.format(epoch, t))
            break

    if epoch > 0 and epoch % batch_size == 0:
        cartAgent.learn()