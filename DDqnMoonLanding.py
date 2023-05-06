import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import cv2

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 100000
batch_size = 64

env = gym.make('LunarLander-v2',render_mode='rgb_array')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

class DQNAgent:
    def __init__(self):
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 4)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.tau = 1e-3

        self.qnetwork_local = QNetwork()
        self.qnetwork_target = QNetwork()
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=0.0005)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward/100.0, next_state, done))

        if len(self.memory) > self.batch_size*10:
            for t in range(10):
                experiences = random.sample(self.memory, k=self.batch_size)
                self.learn(experiences)
               

    def act(self, state, eps):
        # print(state)
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randint(0, 3)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float()

        Q_argmax = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_argmax)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        # Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # print(self.qnetwork_local(states))
        # print(actions.unsqueeze(1))
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))

        loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

agent = DQNAgent()

def double_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        state,_ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done,  truncated, info  = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if i_episode % 500 == 0 and i_episode != 0:
                img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow("test", img)
                cv2.waitKey(30)
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {}\tEPS: {:.2f}'.format(i_episode, np.mean(scores_window),eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = double_dqn()
env.close()

plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

