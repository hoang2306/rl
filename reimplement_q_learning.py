import gymnasium as gym
import numpy as np
from pprint import pprint
from tqdm import tqdm 

# create env 
env = gym.make('CartPole-v1') 

# discretize value in observation space for q-learning 
def discretize(value, bins):
    return np.digitize(value, bins)

# test 
# bins = np.linspace(-2.4, 2.4, 6)
# print(f'bins: {bins}') # [-2.4  -1.44 -0.48  0.48  1.44  2.4 ]
# print(discretize(0.1, bins)) # 

discrete_observation_space = (20, 20, 20, 20)

bins = []
for i in range(4):
    if i == 0 or i == 2:
        start_point = env.observation_space.low[i]
        end_point = env.observation_space.high[i]
    else: 
        start_point = -4 
        end_point = 4

    # remember: ignore start point & end point
    a = np.linspace(start_point, end_point, discrete_observation_space[i], endpoint=False) 
    a = np.delete(a, 0)
    bins.append(a)
    # print(f'bins[{i}]: {a}')

def calculate_discrete_state(state):
    discrete_state = []
    for i in range(4):
        discrete_state.append(discretize(state[i], bins[i]))
    return tuple(discrete_state)
    """
    if use discrete_state instead of tuple(discrete_state), 
    then the q_table shape will be (4, 20, 20, 20, 20, 2)
    because q_table[discrete_state] is slice operator 
    
    else q_table[tuple(...)] is indexing operator 
    """
    # return discrete_state

def pick_sample(s, epsilon):
    # epsilon greedy
    if np.random.random() < epsilon:
        # return env.action_space.sample()
        return np.random.randint(env.action_space.n)
    else:
        return np.argmax(q_table[s])

# q-learning 
q_table = np.zeros(discrete_observation_space + (env.action_space.n,))
# print(f'q table shape: {q_table.shape}')

# config 
gamma = 0.99 
alpha = 0.1
epsilon = 1
epsilon_decay = epsilon/4000
episodes = 6000

reward_history = []
act_history = []

# debug tuple in calculate_discrete_state
# s, _ = env.reset()
# s = calculate_discrete_state(s)
# print(f'discrete state: {s}')
# a = pick_sample(s, epsilon)
# print(f'picked action: {a}')
# s, r, term, trunc, _ = env.step(a)
# s = calculate_discrete_state(s)
# print(f's: {s}')
# print(f'q_table[s]: {q_table[s].shape}')

use_tqdm = False 
pbar = tqdm(range(episodes), desc='training') if use_tqdm else range(episodes)
for i in pbar:
    s, _ = env.reset()
    # discrete state
    s = calculate_discrete_state(s)
    done = False
    total_reward = 0
    cnt_act = 0 
    while not done:
        cnt_act += 1 
        a = pick_sample(s, epsilon)
        state, reward, term, trunc, _ = env.step(a)
        done = term or trunc
        next_state = calculate_discrete_state(state)

        # update q_table
        max_q_value = np.max(q_table[next_state])
        """
        stupid code :D 
        q_table[next_state][a] += alpha * (reward + gamma * max_q_value - q_table[next_state][a]) ????
        """
        q_table[s][a] += alpha * (reward + gamma * max_q_value - q_table[s][a])

        s = next_state
        total_reward += reward

    reward_history.append(total_reward)
    act_history.append(cnt_act)
    # print(f'episode: {i}, total action: {cnt_act}')
    print(f'episode: {i}, total_reward: {total_reward}', end='\r')

    if epsilon >= epsilon_decay:
        epsilon -= epsilon_decay

print(f'max reward in one episode: {max(reward_history)}')
print(f'avg act: {np.mean(act_history[:6000])}')

    
 