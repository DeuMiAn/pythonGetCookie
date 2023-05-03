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
batch_size = 32


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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)


class Qnet(nn.Module):
    def __init__(self) -> None:
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 240)
        self.fc3 = nn.Linear(240, 4)

    def forward(self, x):
        # 활성함수 relu 사용 0~무한
        x = F.relu(self.fc1(x))
        # 활성함수 relu 사용 0~무한
        x = F.relu(self.fc2(x))
        # 마지막에는 활성함수 X
        # 원래 마지막에는 활성함수 안넣음
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # 입실론 그리디 구현을위한 함수
        #
        print(obs)
        out = self.forward(obs)
        # 코인이라는 변수에 랜덤값(0~1) 넣고
        coin = random.random()
        if coin < epsilon:
            print("나는 랜덤하지롱")
            return random.randint(0, 3)  # 출력 0 또는3 랜덤값
        else:
            # print(out)
            # print(out.argmax().item())
            print("나는 운명적이다")
            return out.argmax().item()


def main():
    env = gym.make('LunarLander-v2',  render_mode='human')

    # q 네트워크
    q = Qnet()
    s, _ = env.reset()

    # 학습될수록 epsilon값 줄어들음
    epsilon = 0.5

    print(s)
    a = q.sample_action(torch.from_numpy(s).float(), epsilon)
    print('a')
    print(a)


if __name__ == '__main__':
    main()
