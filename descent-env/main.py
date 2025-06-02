import numpy as np
from descent_env import DescentEnv
import random 

env = DescentEnv()

ALT_MIN = 2000
ALT_MAX = 4000
ALT_MEAN = 1500
ALT_STD = 3000
VZ_MEAN = 0
VZ_STD = 5
RWY_DIS_MEAN = 100
RWY_DIS_STD = 200
altitude_space = np.linspace((ALT_MIN - ALT_MEAN)/ALT_STD, (ALT_MAX - ALT_MEAN)/ALT_STD, 100)
vertical_velocity_space = np.linspace(-10, 10, 100) 
target_altitude_space = np.linspace((ALT_MIN - ALT_MEAN)/ALT_STD, (ALT_MAX - ALT_MEAN)/ALT_STD, 100)
runway_distance_space = np.linspace(-2, 2, 100)

def get_state(obs):
    alt = obs['altitude'][0]
    vz = obs['vz'][0]
    target_alt = obs['target_altitude'][0]
    runway_dist = obs['runway_distance'][0]
    alt_idx = np.digitize(alt, altitude_space) - 1
    vz_idx = np.digitize(vz, vertical_velocity_space) - 1
    target_alt_idx = np.digitize(target_alt, target_altitude_space) - 1
    runway_dist_idx = np.digitize(runway_dist, runway_distance_space) - 1
    return alt_idx, vz_idx, target_alt_idx, runway_dist_idx


actions = list(np.linspace(-1, 1, 20))

def get_sample_action():
    return random.choice(actions)

Q = np.zeros((len(altitude_space), len(vertical_velocity_space), len(target_altitude_space), len(runway_distance_space), len(actions)))

def optimal_policy(state, Q):
    action = actions[np.argmax(Q[state])]
    return action

def epsilon_greedy_policy(state, Q, epsilon=0.1):
    explore = np.random.binomial(1, epsilon)
    if explore:
        action = get_sample_action()
    else:
        action = optimal_policy(state, Q)
        
    return action




i = 0
total_reward = 0
step_reward = 0
step_count = 0
max_steps = 1

obs, _ = env.reset()
done = False

while i < 100:
    obs, _ = env.reset()
    done = False
    print(f"Episode {i + 1}")
    while not done:
        p = random.uniform(0, 1)
        state = get_state(obs)
        if p < 0.3:
            action = get_sample_action()
        else:
            action = optimal_policy(state, Q)
        next_obs, reward, done, _, _ = env.step(np.array([action]))
        next_state = get_state(next_obs)
        action_idx = actions.index(action)
        Q[state][action_idx] = Q[state][action_idx] + 0.9 * (reward + 0.9 * np.max(Q[next_state]) - Q[state][action_idx])
        obs = next_obs
        total_reward += reward
    print(f"Total reward for episode {i + 1}: {total_reward}")
    total_reward = 0
    i+= 1       

env = DescentEnv(render_mode='human')     
obs, _ = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    state = get_state(obs)
    action = optimal_policy(state, Q)
    obs, reward, done, _, _ = env.step(np.array([action]))
    total_reward += reward
    steps += 1
    env.render()

env.close()
print(f"Total reward (Q final): {total_reward}")
print(f"Steps: {steps}")    
    

        
            
    