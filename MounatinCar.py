import gym
import numpy as np
env = gym.make("MountainCar-v0")
learning_rate = 0.1
episodes = 25000
discount = 0.99
epsilon = 0.5
start_epsilon_decay = 1
end_epsilon_decay = episodes//2
epsilon_decay = epsilon/(end_epsilon_decay - start_epsilon_decay)
display = 2000
sizes = [25]*len(env.observation_space.high)
win_size = (env.observation_space.high - env.observation_space.low)/sizes
q_table = np.random.uniform(low=-2, high=0, size=(sizes + [env.action_space.n]))


def discritize_states(state):
    d_state = (state - env.observation_space.low)/win_size
    return tuple(d_state.astype(np.int))
for i in range(episodes):
    #print("episode no.:", i)
    if i % display == 0:
        print(i)
        render = True
    else:
        render = False
    a = discritize_states(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[a])
        next_state, reward, done, _ = env.step(action)
        next_discrete_state = discritize_states(next_state)
        if render:
            env.render()
        if not done:
            future_q = np.max(q_table[next_discrete_state])
            current_q = q_table[a + (action,)]
            new_q = (1-learning_rate)*current_q + learning_rate*(reward + discount*future_q)
            q_table[a + (action, )] = new_q
        elif next_state[0] >= env.goal_position:
            q_table[a + (action, )] = 0
            print("Reached the flag at episode:", i)
        a = next_discrete_state
    if end_epsilon_decay >= i >= start_epsilon_decay:
        epsilon -= epsilon_decay
env.close()

