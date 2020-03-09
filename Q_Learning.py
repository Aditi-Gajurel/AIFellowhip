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
    """
    This function choose an action for a given state
    with the use of epsilon greedy mechanism.

    Parameters
    ----------
    state
        The provided state where the agent is.


    Returns
    -------
    action
        Agent that the policy suggests


    """
    action = 0
    if np.random.uniform(0,1)<epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action

def learn(state, state2, reward, action):
    """
    This function learns Q value from the Q learning technique.

    Parameters
    ----------
    state
        The state from where the transition occurs
    state2
        The state to which we arrive after taking the action
    action
        action specified by choose_action following epsilon-greedy algorithm
    reward
        immediate reward obtained after choosing an action form the state


    Returns
    -------
    Q[state, action]

    """
    predict = Q[state, action]
    target = reward + gamma*np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate*(target-predict)


for episode in range(total_episodes):
    state = env.reset()
    t = 0
    while t<max_steps:
        env.render()
        action = choose_action(state)
        state2, reward, done, info = env.step(action)
        learn(state, state2, reward, action)

        state = state2

        t +=1

        if done:
            break
print(Q)
print("--- %s seconds ---" % (time.time() - start_time))


