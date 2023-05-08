import gym
import random
import torch
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt


import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()

print('CUDA 사용 가능 여부 :', USE_CUDA)
print('현재 사용 device :', DEVICE)
print('CUDA Index :', torch.cuda.current_device())
print('GPU 이름 :', torch.cuda.get_device_name())
print('GPU 개수 :', torch.cuda.device_count())



env = gym.make('LunarLander-v2',render_mode='rgb_array')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

class DQNAgent:
    def __init__(self):
        # self.fc1 = torch.nn.Linear(8, 64).to(DEVICE)
        # self.fc2 = torch.nn.Linear(64, 128).to(DEVICE)
        # self.fc3 = torch.nn.Linear(128, 4).to(DEVICE)

        self.batch_size = 64
        self.learning_rate=0.0005
        self.gamma = 0.98
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.tau = 0.01

        self.update_step = 4
        self.steps=0

        self.qnetwork_local = QNetwork().to(DEVICE)
        self.qnetwork_target = QNetwork().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.memory = ReplayBuffer(4,100000,self.batch_size)


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward/100.0, next_state, done)
        # print(self.steps)
        if len(self.memory) > self.batch_size:
            for i in range(4):
                experiences = self.memory.sample()
                self.learn(experiences)
               

    def act(self, state, eps):
        # print(state)
        state = torch.from_numpy(state).float().to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        # 다시학습모드로
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randint(0, 3)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # print("actions")
        # print(actions)
        # print("states")
        # print(states)
        # print("rewards")
        # print(rewards)
        # print("next_states")
        # print(next_states)
        # print("dones")
        # print(dones)

       
        # with torch.cuda.device(DEVICE):
        # predicted_actions=torch.argmax(self.qnetwork_local(next_states),1)
        # predicted_actions=predicted_actions.reshape(-1,1)
        # print("predicted_actions")
        # print(predicted_actions)
        Q_argmax = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # print("Q_argmax")
        # print(Q_argmax)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_argmax)
        # Q_targets_next = Q_targets_next.detach()
        # print("Q_targets_next")
        # print(Q_targets_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        # Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # print(self.qnetwork_local(states))
        # print(actions.unsqueeze(1))
        Q_actions=actions.unsqueeze(1).reshape(self.batch_size,1)
        # print("actions")
        # print(actions)
        # print(Q_actions)
        Q_expected = self.qnetwork_local(states).gather(1, Q_actions)
        # Q_expected = rewards + (self.gamma * Q_expected * (1 - dones))
        # print("Q_expected")
        # print(Q_expected)
        # print("Q_targets")
        # print(Q_targets)

        loss = torch.nn.functional.mse_loss(Q_targets,Q_expected).to(DEVICE)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps = (self.steps + 1) % self.update_step
        if self.steps == 0 :
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
            

    def soft_update(self, local_model, target_model, tau):
        # print("전")
        # for name, param in target_model.named_parameters():
        #     print(name)
        #     print(param.data)
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        # print("후")
        # for name, param in target_model.named_parameters():
        #     print(name)
        #     print(param.data)

class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

agent = DQNAgent()

def double_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.9995):
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
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth').to(DEVICE)
            break
    return scores

scores = double_dqn()
env.close()

plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

