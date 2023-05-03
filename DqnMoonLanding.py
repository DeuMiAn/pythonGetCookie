import gym
# colleections library는 replay buffer에 쓰일 deque를 import하기 위함임
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 100000
batch_size = 64


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):

        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # mini_batch안에 들어있는 튜블형식에 데이터를 s, a, r, s_prime, done_mask 따로따로 분리함
            # s_prime은 s에서 action한 값 즉 다음값
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


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a.long())
        max_q_prime = q_target.forward(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('LunarLander-v2',  render_mode='rgb_array')

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for epiIndex in range(10000):
        epsilon = max(0.01, 0.5 - 0.01*(epiIndex/200))
        s, _ = env.reset()
        done = False
        time = 0
        for t in range(1000):
            time += 1
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
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
            train(q, q_target, memory, optimizer)
        if epiIndex % print_interval == 0 and epiIndex != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                epiIndex, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    env.close()

    print(s)

    print('a')
    print(a)


if __name__ == '__main__':
    main()
