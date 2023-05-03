import gym
# colleections library는 replay buffer에 쓰일 deque를 import하기 위함임
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2

LEARNING_RATE = 0.0005
GAMMA = 0.95
BUFFER_LIMIT = 100000
BATCH_SIZE = 64


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 3)  # 출력 0 또는3 랜덤값
        else:
            return out.argmax().item()


class QTGnet(nn.Module):
    def __init__(self):
        super(QTGnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 3)  # 출력 0 또는3 랜덤값
        else:
            return out.argmax().item()


def train(q1, q_target, memory, optimizer1, q2, optimizer2):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(BATCH_SIZE)
        print(s)
        q_out = q1(s)
        q_a = q_out.gather(1, a.long())
        print("------")
        print(s_prime)
        a_prime = q2(s_prime)
        print("------")
        print(a_prime)
        max_q_prime = q_target.forward(a_prime).max(1)[0].unsqueeze(1)
        target = r + GAMMA * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()


def main():
    env = gym.make('LunarLander-v2',  render_mode='rgb_array')

    q1 = Qnet()
    q2 = Qnet()
    q_target = QTGnet()
    q2.load_state_dict(q1.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0

    optimizer1 = optim.Adam(q1.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.Adam(q2.parameters(), lr=LEARNING_RATE)

    for epiIndex in range(10000):
        epsilon = max(0.01, 0.5 - 0.01*(epiIndex/200))
        s, _ = env.reset()
        done = False
        time = 1000
        for t in range(time):
            a = q1.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if epiIndex % 500 == 0 and epiIndex != 0:
                img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow("test", img)
                cv2.waitKey(30)
            if done:
                break
        if memory.size() > 5000:
            train(q1, q_target, memory, optimizer1, q2, optimizer2)
        if epiIndex % print_interval == 0 and epiIndex != 0:
            q_target.load_state_dict(q1.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                epiIndex, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
