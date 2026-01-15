import numpy as np
import random

class SimpleQLearning:


    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, 
                 epsilon_decay=0.9998, epsilon_min=0.1):
        self.q_table = np.zeros((64, 4))  # 64 states, 4 actiuni (FrozenLake 8x8)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def select_action(self, state, training=True):

        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_episode(self, env):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            self.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        self.decay_epsilon()
        return total_reward

    def evaluate(self, env, n_episodes=100):
        successes = 0
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action = self.select_action(state, training=False)
                state, reward, done, truncated, _ = env.step(action)
                if done and reward == 1:
                    successes += 1
        return (successes / n_episodes) * 100


class SimpleMARLSystem:

    
    def __init__(self):
        self.agent1 = SimpleQLearning()
        self.agent2 = SimpleQLearning()
        
    def train_episode(self, env):
        (s1, s2), _ = env.reset()
        done = False
        
        while not done:
            a1 = self.agent1.select_action(s1)
            a2 = self.agent2.select_action(s2)
            
            (ns1, ns2), (r1, r2), done, truncated, info = env.step((a1, a2))
            
            self.agent1.update(s1, a1, r1, ns1, done)
            self.agent2.update(s2, a2, r2, ns2, done)
            
            s1, s2 = ns1, ns2
            if done or truncated:
                break
                
        self.agent1.decay_epsilon()
        self.agent2.decay_epsilon()

    def evaluate(self, env, n_episodes=100):
        successes = 0
        for _ in range(n_episodes):
            (s1, s2), _ = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 200:
                a1 = self.agent1.select_action(s1, training=False)
                a2 = self.agent2.select_action(s2, training=False)
                
                (ns1, ns2), (r1, r2), done, truncated, info = env.step((a1, a2))
                s1, s2 = ns1, ns2
                steps += 1
                
                if done and info.get('success', False):
                    successes += 1
                    
        return (successes / n_episodes) * 100