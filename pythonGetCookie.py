import gym
import numpy as np
from collections import defaultdict

env = gym.make("FrozenLake-v1", render_mode = "human")
# env = gym.make("FrozenLake-v1")
env.reset()
# env.close()
totalWin=0 #정답찾는 횟수를 체크하기위한 부분
N = defaultdict(float)
Q = defaultdict(float)
total_reward = defaultdict(float)
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = np.random.uniform(0, 1)
        

def epsilon_greedy(state, Q, epsilon = 0.3):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax([Q[(state, x)] for x in range(env.action_space.n)])
    return(action)


def generate_episode(Q, init_state, epsilon = 0.8):
    episode = []
    time_steps = 0
    state = init_state
    while True:
        time_steps += 1
        action = epsilon_greedy(state, Q, epsilon = epsilon)
        next_state, reward, terminal, *_ = env.step(action)
        episode.append((state, action, reward))
        
        if terminal:
            if reward>0:
                global totalWin #정답찾는 횟수를 체크하기위한 부분
                totalWin +=1 
                print("Reached the goal with reward = ", reward) #정답찾는 횟수를 체크하기위한 부분
                # response = input("Do you want to continue? (yes/no): ")
                # if response == "no":
                #      break
            break
        else:
            state = next_state
    return(episode)
    
state = env.reset()
state = state[0]

epsilon = 0.79 # 0.78에서 점자적으로 감소할것
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
            Q[(state, action)] = total_reward[(state, action)] / N[(state, action)]
    # epsilon 값 감소
    epsilon = max(0.067, epsilon - 0.00065)
    print(totalWin)

