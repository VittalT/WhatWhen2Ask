import gym
import homegrid
import numpy as np
import random
from collections import defaultdict
import json

TEST = True

class QLearningAgent:
    def __init__(self, env_name="homegrid-task", alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500):
        # Initialize environment and hyperparameters
        self.env = gym.make(env_name, disable_env_checker=True)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes  # Total training episodes
        self.Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))  # Q-table

    def preprocess_state(self, obs, info):
        """Convert state into a hashable format for the Q-table."""
        # print(state["image"])
        flattened_image = tuple(obs["image"].flatten())
        # if self.task != self.env.task:
        #     print(f"task mismatch {self.task} {self.env.task}")
        return flattened_image + (self.env.task,)
    
    def preprocess2_state(self, obs, info):
        """Convert state into a hashable format for the Q-table."""
        # print(info["symbolic_state"])
        symbolic_state = info["symbolic_state"]
        state_str = json.dumps(
            symbolic_state,
            sort_keys=True, 
            default=lambda o: o.item() if isinstance(o, np.generic) else o
        )
        return hash(state_str)
    
    def preprocess3_state(self, obs, info):
        """
        Convert symbolic state into a flat representation for Q-learning.
        """
        symbolic_state = info["symbolic_state"]
        agent = symbolic_state["agent"]
        objects = symbolic_state["objects"]

        # Flatten agent information
        state_vector = [
            agent["pos"][0], agent["pos"][1],  # Agent's position (x, y)
            agent["dir"],                      # Agent's direction
            1 if agent["carrying"] else 0      # Carrying status (1 if carrying, else 0)
        ]

        # Add information about objects (position, room, state)
        for obj in objects:
            state_vector.extend([
                obj["pos"][0] if obj["pos"] != (-1, -1) else -1,  # Object x-pos
                obj["pos"][1] if obj["pos"] != (-1, -1) else -1,  # Object y-pos
                1 if obj["state"] == "open" else 0,               # State: open (1) or closed (0)
            ])

        # Add front object (encoded as 1-hot for simplicity)
        front_obj = symbolic_state["front_obj"]
        state_vector.append(1 if front_obj else 0)  # Front object presence

        return tuple(state_vector)  # Return as a hashable tuple
    
    def choose_action(self, state):
        """Choose an action using the epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.Q_table[state])  # Exploit

    def train(self):
        """Train the agent using the Q-learning algorithm."""
        print(f"Q_table length: {len(self.Q_table)}")

        for episode in range(self.episodes):
            # print("\n\nepisode ", episode)

            obs, info = self.env.reset()
            # self.task = self.env.task
            # print(obs.keys())
            state = self.preprocess_state(obs, info)
            total_reward = 0

            for step in range(self.env.max_steps if TEST else 0):
                # Choose action
                action = self.choose_action(state)
                # print(action)

                # Take action in the environment
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                next_state = self.preprocess_state(next_obs, next_info)
                total_reward += reward

                # Update Q-table using the Bellman equation
                self.Q_table[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state][action]
                )

                # Move to the next state
                state = next_state

                # Break if the episode is over
                if terminated or truncated:
                    break

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}")

    def test(self, episodes=500, render=False):
        """Test the agent using the learned Q-table."""
        print(f"Q_table length: {len(self.Q_table)}")

        total_rewards = []
        for episode in range(episodes):
            obs, info = self.env.reset()
            # self.task = self.env.task
            state = self.preprocess_state(obs, info)
            total_reward = 0

            for step in range(self.env.max_steps if TEST else 0):
                # Choose the best action based on the Q-table
                action = np.argmax(self.Q_table[state])
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                next_state = self.preprocess_state(next_obs, next_info)
                total_reward += reward
                state = next_state

                if render:
                    self.env.render()  # Render the environment

                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{episodes}: Reward = {total_reward}")
    
        average_reward = sum(total_rewards) / episodes
        print(f"Average Total Reward over {episodes} episodes: {average_reward:.2f}")
        return average_reward
        
    def uncertainty_score(self, state):
        """
        Compute the uncertainty score based on the entropy of the Q-value distribution.
        :param state: The current state (hashed or processed).
        :return: Uncertainty score (0 to 1).
        """
        q_values = self.Q_table[state]  # Get Q-values for the state
        print(q_values)
        
        # Convert Q-values to probabilities using softmax
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        
        # Compute entropy of the probability distribution
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))  # Add small value to avoid log(0)
        
        # Normalize entropy to a 0-1 scale
        max_entropy = np.log(len(q_values))  # Maximum entropy occurs when all actions are equally likely
        uncertainty = entropy / max_entropy  # Scale entropy to the range [0, 1]
        
        return uncertainty

    def save_q_table(self, filepath):
        """Save the Q-table to a file."""
        np.save(filepath, dict(self.Q_table))
        print(f"Q-table saved to {filepath} with length {len(self.Q_table)}")

    def load_q_table(self, filepath):
        """Load a Q-table from a file."""
        loaded_q_table = np.load(filepath, allow_pickle=True).item()
        self.Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), loaded_q_table)
        print(f"Q-table loaded from {filepath} with length {len(self.Q_table)}")



if __name__ == "__main__":

    agent = QLearningAgent(env_name="homegrid-task", alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500 if TEST else 1)
    agent.load_q_table("q_table.npy")
    sample_state = random.choice(list(agent.Q_table.keys()))
    # print(sample_state)
    print(agent.uncertainty_score(sample_state))

# if __name__ == "__main__":

#     agent = QLearningAgent(env_name="homegrid-task", alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500 if TEST else 1)
#     # agent.load_q_table("q_table.npy")

#     print("Training the agent...")
#     agent.train()

#     print("Testing the agent...")
#     agent.test(render=False, episodes=500)

#     # Save and load Q-table if needed
#     agent.save_q_table("q_table.npy")
#     agent.load_q_table("q_table.npy")