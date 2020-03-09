import gym
import numpy as np
import time, pickle, os
env = gym.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))
start_time = time.time()

def choose_action(state):
    action = 0
    if np.random.uniform(0,1)<epsilon:
        action = env.action_space.sample()

    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma*Q[state2, action2]
    Q[state,action] = Q[state, action] + lr_rate*(target-predict)

for episode in range(total_episodes):

    t = 0
    state = env.reset()
    action = choose_action(state)

    while t<max_steps:
        # env.render()

        state2, reward, done, info = env.step(action)
        action2 = choose_action(state2)
        learn(state, state2, reward, action, action2)
        state = state2
        action  =action2
        t+=1

        if done:
            break
print(Q)
print("--- %s seconds ---" % (time.time() - start_time))

