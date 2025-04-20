import gym
import homegrid
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from homegrid.GPT import GPT4Helper
from pprint import pprint
import matplotlib.pyplot as plt

def visualize_image(image):
    plt.imshow(image)  # Matplotlib expects (H, W, 3) in RGB
    plt.axis("off")  # Hide axes for clarity
    plt.show()

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class DQNAgent:
    def __init__(self, env_name="homegrid-task", episodes=500):
        # Initialize environment and hyperparameters
        self.env = gym.make(env_name, disable_env_checker=True)
        self.alpha = 0.001
        self.gamma = 0.99
        self.epsilon = 1
        self.batch_size = 64
        self.episodes = episodes
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.01
        self.max_encoded_task_size = 100  # Length of the encoded task vector
        self.max_hint_size = 100  # Length of the hint vector
        self.llm_cost = 0.05
        self.num_llm_calls = 0
        self.max_llm_calls = 0

        # Replay buffer
        self.replay_buffer = {}  # Dictionary to store per-task experiences
        self.max_replay_buffer_size = 10000

        # Initialize DQNetwork
        image_shape = self.env.observation_space['image'].shape
        flattened_image_size = np.prod(image_shape)
        input_dim = flattened_image_size + self.max_encoded_task_size + self.max_hint_size
        self.output_dim = self.env.action_space.n
        self.model = DQNetwork(input_dim, self.output_dim)
        self.target_model = DQNetwork(input_dim, self.output_dim)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # Initialize LLMHelper
        self.llm_helper = GPT4Helper(model="gpt-4")
        self.hint_threshold = 0.7  # Uncertainty threshold to query LLM

    def preprocess_state(self, obs, hint=None):
        """Convert state into a flat representation, including an LLM hint."""
        # Flatten the image observation
        flattened_image = obs["image"].flatten() / 255.0  # Normalize to [0,1]

        # Encode the task string
        task_vector = self.encode_str(self.env.task, self.max_encoded_task_size)

        # Encode the hint (default to zeros if no hint is provided)
        if hint is None:
            hint_vector = np.zeros(self.max_hint_size, dtype=np.float32)
        else:
            hint_vector = self.encode_str(hint, self.max_hint_size)

        # Concatenate all components
        return np.concatenate([flattened_image, task_vector, hint_vector])
    
    def encode_str(self, str, max_size):
        """Convert the hint string into a numerical vector."""
        encoded_vector = np.zeros(max_size, dtype=np.float32)
        for i, char in enumerate(str):
            if i >= max_size:
                break # Truncate if the string is longer than max_encoded_task_size
            # Use a hash to convert the character into a numerical value
            encoded_vector[i] = (ord(char) * (i + 1)) % 1000 / 1000.0 # Scale to [0, 1]
        return encoded_vector

    def choose_action(self, obs, info):
        """Choose an action using the epsilon-greedy policy, incorporating LLM hints if needed."""
        # Calculate the agent's uncertainty
        agent_uncertainty = self.uncertainty_score(obs)

        hint = None
        cost = 0
        if agent_uncertainty > self.hint_threshold and self.num_llm_calls < self.max_llm_calls:
            # Query the LLM for a hint
            # print(info['symbolic_state'])
            cost = self.llm_cost * np.exp(self.num_llm_calls)
            hint, hint_uncertainty = self.llm_helper.query_llm(state=info['symbolic_state'], task=self.env.task)
            self.num_llm_calls += 1
            print(f"Queried LLM: Hint = {hint}, Hint Uncertainty = {hint_uncertainty:.2f}")

        # Preprocess the state with the hint
        full_state = self.preprocess_state(obs, hint)

        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return self.env.action_space.sample(), cost  # Explore
        full_state_tensor = torch.FloatTensor(full_state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(full_state_tensor)
        return torch.argmax(q_values).item(), cost  # Exploit


    def update_target_network(self):
        """Update target network weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self):
        """Train the model using a batch from the replay buffer."""
        if len(self.replay_buffer) == 0:
            return  # No tasks stored yet
        
        # Pick a random task to train on
        task_id = random.choice(list(self.replay_buffer.keys()))

        # # Ensure enough samples exist in the selected task buffer
        # if len(self.replay_buffer[task_id]) < self.batch_size:
        #     return  # Not enough samples, wait until the buffer is filled

        # Sample experiences from the selected task's replay buffer
        min_batch_size = min(self.batch_size, len(self.replay_buffer[task_id]))
        if min_batch_size < 1:
            return
        batch = random.sample(self.replay_buffer[task_id], min_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Assuming the following are lists or numpy arrays
        states = np.array(states, dtype=np.float32)            # Convert to a numpy array if not already
        actions = np.array(actions, dtype=np.int64)            # Ensure actions are integers
        rewards = np.array(rewards, dtype=np.float32)          # Convert rewards to float32
        next_states = np.array(next_states, dtype=np.float32)  # Convert to a numpy array if not already
        dones = np.array(dones, dtype=np.float32)              # Convert dones to float32

        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Add an extra dimension for actions
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)


        # Compute target Q-values
        with torch.no_grad():
            target_q_values = rewards + self.gamma * (1 - dones) * torch.max(self.target_model(next_states), dim=1)[0]

        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions).view(-1)
        if current_q_values.shape != target_q_values.shape:
            print("batch size", min_batch_size)
            print("current ", current_q_values.shape)
            print("target ", target_q_values.shape)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def uncertainty_score(self, obs):
        """
        Compute the uncertainty score based on the entropy of the Q-value distribution.
        :param state: The current state (preprocessed as a flat vector).
        :return: Uncertainty score (0 to 1).
        """
        state = self.preprocess_state(obs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        with torch.no_grad():
            q_values = self.model(state_tensor).numpy().flatten()  # Get Q-values for the state

        # Convert Q-values to probabilities using softmax
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))

        # Compute entropy of the probability distribution
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))  # Add small value to avoid log(0)

        # Normalize entropy to a 0-1 scale
        max_entropy = np.log(len(q_values))  # Maximum entropy occurs when all actions are equally likely
        uncertainty = entropy / max_entropy  # Scale entropy to the range [0, 1]

        return uncertainty

    def train(self, episodes=None):
        if episodes is None:
            episodes = self.episodes
        """Train the agent using the DQN algorithm."""
        for episode in range(episodes):
            obs, info = self.env.reset()
            self.num_llm_calls = 0
            # print("obs", obs.keys())
            # print("info", info.keys())
            state = self.preprocess_state(obs)
            total_reward = 0

            task_id = self.env.task  # Get the current task identifier

            # Ensure there's a deque for this task
            if task_id not in self.replay_buffer:
                self.replay_buffer[task_id] = deque(maxlen=self.max_replay_buffer_size)


            if episode % 100 == 0:
                sample_state = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(sample_state)
                print(f"Sample Q-Values at episode {episode}: {q_values}")
                print("Action space:", self.env.action_space.n)

            for step in range(self.env.max_steps):
                # Choose action
                action, cost = self.choose_action(obs, info)

                # Take action in the environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                reward -= cost
                next_state = self.preprocess_state(obs)

                # Store transition in replay buffer
                self.replay_buffer[task_id].append((state, action, reward, next_state, terminated))

                # Update state
                state = next_state
                total_reward += reward

                # Train the model
                self.train_step()

                if terminated or truncated:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update target network periodically
            if episode % 5 == 0:
                self.update_target_network()

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

    def test(self, episodes=None, render=False):
        """Test the agent using the learned policy."""
        if episodes is None:
            episodes = self.episodes
        total_rewards = []
        for episode in range(episodes):
            obs, info = self.env.reset()
            pprint(obs)
            visualize_image(obs['image'])
            pprint(info)
            
            self.num_llm_calls = 0
            total_reward = 0

            for step in range(self.env.max_steps):
                action, cost = self.choose_action(obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
                reward -= cost
                total_reward += reward

                if render:
                    self.env.render()

                if terminated or truncated:
                    break

            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{episodes}: Reward = {total_reward}")

        average_reward = sum(total_rewards) / episodes
        print(f"Average Total Reward over {episodes} episodes: {average_reward:.2f}")
        return average_reward

if __name__ == "__main__":
    agent = DQNAgent(env_name="homegrid-task", episodes=0)

    print("Training the agent...")
    agent.train(episodes=1)

    print("Testing the agent...")
    agent.test(episodes=1, render=False)