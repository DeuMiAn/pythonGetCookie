

# env = gym.make("FrozenLake-v1", render_mode = "human")
def requestFrozenLakeGame():
    import numpy as np
    from collections import defaultdict
    import gym
    from gym.envs.registration import register
    env = gym.make("FrozenLake-v1")
    env.reset()
    # env.close()
    totalWin = 0  # 정답찾는 횟수를 체크하기위한 부분
    N = defaultdict(float)
    Q = defaultdict(float)
    total_reward = defaultdict(float)
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            Q[(s, a)] = np.random.uniform(0, 1)

    def epsilon_greedy(state, Q, epsilon=0.3):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax([Q[(state, x)]
                               for x in range(env.action_space.n)])
        return(action)

    def generate_episode(Q, init_state, epsilon=0.8):
        episode = []
        time_steps = 0
        state = init_state
        while True:
            time_steps += 1
            action = epsilon_greedy(state, Q, epsilon=epsilon)
            next_state, reward, terminal, *_ = env.step(action)
            episode.append((state, action, reward))

            if terminal:
                if reward > 0:
                    global totalWin  # 정답찾는 횟수를 체크하기위한 부분
                    totalWin += 1
                    print("Reached the goal with reward = ",
                          reward)  # 정답찾는 횟수를 체크하기위한 부분
                    # response = input("Do you want to continue? (yes/no): ")
                    # if response == "no":
                    #      break
                break
            else:
                state = next_state
        return(episode)

    state = env.reset()
    state = state[0]

    epsilon = 0.79  # 0.78에서 점자적으로 감소할것
    for i in range(100000):
        print("{iteration}-th Episode".format(iteration=i))
        env.reset()
        episode = generate_episode(Q, init_state=state, epsilon=epsilon)
        state_action = [(s, a) for (s, a, r) in episode]
        rewards = [r for (s, a, r) in episode]

        for t, (state, action, reward) in enumerate(episode):
            if not (state, action) in state_action[:t]:
                gamma = [0.9 ** x for x in range(0, len(rewards[t:]))]
                G = sum([rr * g for rr, g in zip(rewards[t:], gamma)])
                total_reward[(state, action)] += G
                N[(state, action)] += 1
                Q[(state, action)] = total_reward[(
                    state, action)] / N[(state, action)]
        # epsilon 값 감소
        epsilon = max(0.067, epsilon - 0.00065)
        print(totalWin)


def decode_state(state):
    """상태 값을 (택시 행, 택시 열, 승객 위치, 목적지 위치) 형태로 분리합니다."""
    taxi_location = state // 500  # 택시 위치 (0 ~ 24)
    passenger_location = (state % 500) // 20  # 승객 위치 (0 ~ 4)
    destination = (state % 500) % 20 // 4  # 목적지 위치 (0 ~ 3)
    taxi_row, taxi_col = taxi_location // 5, taxi_location % 5  # 택시 위치 (행, 열)
    print("taxi_row: "+str(taxi_row))
    print("taxi_col: "+str(taxi_col))
    print("passenger_location: "+str(passenger_location))
    print("destination: "+str(destination))


def python_reinforcement_learning1():
    import numpy as np
    import gym
    import random

    f = open("qtable.txt", 'w')

    # create Taxi environment
    env = gym.make('Taxi-v3', render_mode='ansi')

    # 현재 상태에 따른 보상 테이블 초기화
    state_size = env.observation_space.n
    action_size = env.action_space.n
    print(state_size)
    print(action_size)
    input()
    qtable = np.zeros((state_size, action_size))

    # 하이퍼 매개 변수
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # 훈련 변수들
    num_episodes = 1000
    max_steps = 99

    # 훈련시작
    for episode in range(num_episodes):

        # 환경을 재설정
        state = env.reset()
        print(state)
        decode_state(state[0])
        input()
        state = state[0]
        terminal = False

        for s in range(max_steps):

            # 탐사 시작~ 절충안 찾기
            if random.uniform(0, 1) < epsilon:
                # 탐험
                action = env.action_space.sample()
            else:
                # 안전빵
                action = np.argmax(qtable[state, :])

            # 행동을 취하고 보상을 준수합니다
            next_state, reward, terminal, info, *_ = env.step(action)
            print()

            # Q 알고리즘
            # 현재에 기대되는 값은/      올드 값/                 학습속도/        보상/   거리간 보상 가중치장치/ 어디로 갔을떄 거기서 기대할수있는 가장 큰값    예전값
            qtable[state, action] = qtable[state, action] + learning_rate * \
                (reward + discount_rate *
                 np.max(qtable[next_state, :])-qtable[state, action])

            # 상태 업데이트
            state = next_state
            print(env.render())

            # if terminal, finish episode
            if terminal == True:
                break

        # 엡실론 감소(모험심 감소함)
        epsilon = np.exp(-decay_rate*episode)
        f.write("\n")
        f.write("%d \n" % episode)
        f.write("\n")
        for qValue in qtable:
            data = ' | '.join(map(str, qValue))
            f.write("%s\n" % data)

    f.close()
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    #
    # 초기상태 얻기
    state = env.reset()
    state = state[0]
    terminal = False
    rewards = 0
    for s in range(max_steps+1):
        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))
        action = np.argmax(qtable[state, :])
        next_state, reward, terminal, *_ = env.step(action)
        rewards += reward
        print(env.render())
        print(f"score: {rewards}")
        state = next_state

        if terminal == True:
            break

    # 택시 환경의 이 인스턴스를 종료합니다
    env.close()


python_reinforcement_learning1()
