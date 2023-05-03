import gym
# colleections library는 replay buffer에 쓰일 deque를 import하기 위함임
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2


# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000  # replay buffer에 최대크기지정 문제마다 버퍼의 크기가 다름 DQN논문에서는 100만을 선언함
batch_size = 32  # replay buffer에서 샘플링할때 보통 필요함 최적이 32임


# replay buffer 를 구현한 클래스
class ReplayBuffer():
    def __init__(self):
        # collections.deque에서 넣고 뺴고를 함
        # 이때 buffer_limit가 최대크기를 제한해서 최대크기 이상값이면 자동으로 오래된데이터 삭제됨
        self.buffer = collections.deque(maxlen=buffer_limit)

    # put메소드 리플리데이터 넣는함수
    def put(self, transition):
        self.buffer.append(transition)

    # 메모리 샘플링해주는 함수
    def sample(self, n):
        # 랜덤하게 버퍼해서 샘플링해라
        # 이떄 random.sample(sequence, k)
        # sequence: 리스트, 집합, range() 등 random의 범위가 될 sequence 입력
        # k: 반환될 리스트의 크기 입력
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

    def size(self):
        return len(self.buffer)


# 토치 nn.Module 이라는 클래스 상속받음
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # 딥러닝 레이어는 크게 3개
        # 입력 레이어 4개를 입력받고
        self.fc1 = nn.Linear(4, 128)
        # 이후 2개의 은닉 레이어가 존재
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

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
        out = self.forward(obs)
        # 코인이라는 변수에 랜덤값(0~1) 넣고
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)  # 출력 0 또는1 랜덤값
        else:
            # print(out)
            # print(out.argmax().item())
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # 최종적으로
        # s는 32개임 [32,4]
        # q(s)출력은 [32,2]
        q_out = q(s)
        print("q_out")
        print(q_out)
        # a는 액션들만 모아놓은것으로 [32,1] 형태임

        # print("a테스트")
        # print(a)

        # gather에서 1차원에서 골라라
        # 예시 a[0][1] 이 있으면 a[][*]여기서 고르는거임
        q_a = q_out.gather(1, a)
        print("a")
        print(a)
        print("q_a")
        print(q_a)

        # q타켓 호출함
        # q타겟 네트워크에 지금까지 state를 넣음 이때 s_prime [32,2]
        # max_q_prime = q_target(s_prime)
        # max(1)로 max값만뽑으면 [32]로 바뀜
        # max_q_prime=max_q_prime.max(1)[0]
        # unsqueeze로 차원을 늘림 [32,1]로 바뀜
        # max_q_prime=max_q_prime.unsqueeze(1)
        # 즉 요약하면 아래 코드 한줄임
        max_q_prime = q_target.forward(s_prime).max(1)[0].unsqueeze(1)
        # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        # 결론은 위 형태로 한것은 아래 차원을 맞춰서 계산하기위함
        # done_mask는 게임이 끝나면 gamma * max_q_prime을 0으로 만들어줌
        # done_mask=[32,1]
        target = r + gamma * max_q_prime * done_mask
        print("target")
        print(target)

        # 기존 q네트워크와  target Q네트워크에 간극을 줄이기 위한 로스함수
        loss = F.smooth_l1_loss(q_a, target)

        # 경상하강법 모드 비워줌 경사를 0으로 초기화하여 이전에 계산된 경사값이 영향을 미치지 않도록 합니다.
        optimizer.zero_grad()

        # loss.backward하는순간
        # 현재 배치에서 계산된 손실 함수에 대한 그라디언트(gradient)를 계산합니다.
        # 이 그라디언트는 모델의 모든 가중치에 대한 손실 함수의 기울기입니다.
        loss.backward()

        # 가중치 업데이트됨
        # 계산된 그라디언트를 사용하여 가중치를 업데이트합니다. 이때 학습률(learning rate)이 곱해져서 가중치가 얼마나 업데이트될지 결정됩니다.
        optimizer.step()

        # 결론적으로 32배치가 10번 업데이트하니까 총 320개의 배치가 업데이트됨


def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    # q네트워크 선언
    q = Qnet()
    # 타켓네트워크 선언
    q_target = Qnet()
    # q네트워크에 있는 dict를 복제해서 타켓에 붙여넣음
    # state_dict는 model의 weight 정보를 dict형태로 있음
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 10
    score = 0.0
    # 딥러닝 경상하강법 관련 손실최소화 알고리즘
    # 이때 q 네트워크만 최적화하겠음
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        # 학습될수록 epsilon값 줄어들음
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)
                      )  # Linear annealing from 8% to 1%
        s, _ = env.reset()
        # print("s")
        # print(s)
        done = False

        while not done:
            # 아까 만들어둔 q네트워크에 샘플액션으로 동작 함수실행
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            # 게임이 끝날때이후로 값을 거르기위해 거름망
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if n_epi % 100 == 0 and n_epi != 0:
                img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow("test", img)
                cv2.waitKey(30)
            if done:
                break

        # 메모리가 2000(충분히 쌓이면) 훈련함수시작(학습)
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
        # 에피소드 진행이 print_interval횟수가 진행될때마다 타켓네트워크 업데이트함
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
