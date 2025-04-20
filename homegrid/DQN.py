import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from homegrid.BLIP import BLIP2Helper
from pprint import pprint
import matplotlib.pyplot as plt
import spacy
import os
import json
from PIL import Image

# Load FastText word embeddings from spaCy (300-dimensional)
nlp = spacy.load("en_core_web_md")
EMBED_DIM = nlp("hello").vector.shape[0]  # 300


def get_fasttext_embedding(text):
    """
    Convert a text (word or phrase) to a 300-dimensional FastText embedding.
    """
    return torch.tensor(nlp(text).vector, dtype=torch.float32)


def visualize_image(image):
    plt.imshow(image)  # expects (H, W, 3) in RGB
    plt.axis("off")
    plt.show()


class DQN(nn.Module):
    def __init__(self, observation_shape, num_actions, embed_dim=EMBED_DIM):
        """
        Multi-modal DQN that processes:
          - An observation (e.g. image) via a CNN.
          - A context vector arranged as follows:
              * First 300 dimensions: task embedding.
              * Remaining 904 dimensions: direction (4 dims), carrying embedding (300 dims),
                front object embedding (300 dims), and hint embedding (300 dims).
          The network processes the task and non-task parts of the context via separate branches.
        """
        super(DQN, self).__init__()

        # CNN branch for image observation (input shape: (H, W, C) -> (batch, C, H, W))
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Determine CNN output size using a dummy input.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape).permute(0, 3, 1, 2)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # Define dimensions:
        # Task branch: first 300 dims.
        task_dim = embed_dim  # 300
        # Non-task branch: remaining context.
        non_task_dim = (
            4 + 3 * embed_dim
        )  # one-hot (4) + carrying (300) + front object (300) + hint (300) = 904

        # Separate branch for the task embedding.
        self.task_net = nn.Sequential(nn.Linear(task_dim, 128), nn.ReLU())

        # Separate branch for the non-task context.
        self.non_task_net = nn.Sequential(nn.Linear(non_task_dim, 256), nn.ReLU())

        # Final fusion layer: fuse CNN features, task branch features, and non-task branch features.
        fused_input_size = cnn_out_size + 128 + 256
        self.fc = nn.Sequential(
            nn.Linear(fused_input_size, 256), nn.ReLU(), nn.Linear(256, num_actions)
        )

    def forward(self, observation, context):
        """
        Args:
            observation (Tensor): Image tensor of shape (batch, C, H, W).
            context (Tensor): Pre-concatenated context vector of shape (batch, 1204),
                              where the first 300 dims are for the task and the rest 904
                              dims are for direction, carrying, front object, and hint.
        Returns:
            q_values (Tensor): Q-value predictions of shape (batch, num_actions)
        """
        # Separate context into task and non-task parts.
        task_embedding = context[:, :300]
        non_task_context = context[:, 300:]

        # Process image observation.
        obs_features = self.cnn(observation)
        # Process task and non-task branches.
        task_features = self.task_net(task_embedding)
        non_task_features = self.non_task_net(non_task_context)

        # Fuse all features.
        fused = torch.cat([obs_features, task_features, non_task_features], dim=1)
        q_values = self.fc(fused)
        return q_values


class DQNAgent:
    def __init__(self, env_name="homegrid-task", episodes=500):
        # Initialize environment and hyperparameters.
        self.env = gym.make(env_name, disable_env_checker=True)
        self.alpha = 0.001
        self.gamma = 0.99
        self.epsilon = 1
        self.batch_size = 64
        self.episodes = episodes
        self.epsilon_decay = (
            0.9985  # faster decay can help transition from exploration to exploitation
        )
        self.epsilon_min = 0.01
        self.llm_cost = 0.01
        self.num_llm_calls = 0
        self.max_llm_calls = 0  # Adjust to allow LLM queries if needed
        self.current_hint = ""

        # Replay buffer (per task originally, but can be global if preferred)
        self.replay_buffer = {}  # remains per task in this example
        self.max_replay_buffer_size = 10000

        self.distance_weight = 0.01  # adjusted reward shaping weight

        # Get observation shape from the environment (e.g. (96, 96, 3))
        self.obs_shape = self.env.observation_space["image"].shape
        # Number of actions from environment.
        self.output_dim = self.env.action_space.n

        # The context vector will be 1204-dimensional in total:
        # 300 for task + 4 for direction + 300 for carrying + 300 for front object + 300 for hint.
        self.model = DQN(self.obs_shape, self.output_dim)
        self.target_model = DQN(self.obs_shape, self.output_dim)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # Initialize LLMHelper (assumed implemented in BLIP2Helper).
        self.llm_helper = BLIP2Helper()
        self.hint_threshold = 0.95

        # Checkpoint interval (save model every N episodes)
        self.checkpoint_interval = 250
        # Directory to save checkpoints
        self.checkpoint_dir = "checkpoints6"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def encode_str(self, text, encode_context=True):
        """
        Convert a string to a 300-dimensional FastText embedding.
        """
        if encode_context:
            return get_fasttext_embedding(text).unsqueeze(0)
        else:
            return text if text != "" else "none"

    def preprocess_state(self, obs, info, hint=None, encode_context=True):
        """
        Preprocess the state by concatenating the observation with context features.
        The context is arranged as:
          [task_embedding, direction_one_hot, carrying_embedding, front_obj_embedding, hint_embedding]
        """
        if not hint:
            hint = self.current_hint

        # Get task and symbolic state information.
        task_str = self.env.task  # ensure task is defined
        direction_int = info["symbolic_state"]["agent"]["dir"]
        carrying_str = info["symbolic_state"]["agent"]["carrying"] or ""
        front_obj_str = info["symbolic_state"]["front_obj"] or ""
        hint_str = hint

        if encode_context:
            observation = (
                torch.FloatTensor(obs["image"] / 255.0).permute(2, 0, 1).unsqueeze(0)
            )
            task_embed = self.encode_str(task_str, encode_context)  # (1, 300)
            direction_one_hot = torch.zeros(1, 4)
            direction_one_hot[0, direction_int] = 1.0
            carrying_embed = self.encode_str(carrying_str, encode_context)  # (1, 300)
            front_obj_embed = self.encode_str(front_obj_str, encode_context)  # (1, 300)
            hint_embed = self.encode_str(hint_str, encode_context)  # (1, 300)
            # Concatenate with task embedding at the start.
            context = torch.cat(
                [
                    task_embed,
                    direction_one_hot,
                    carrying_embed,
                    front_obj_embed,
                    hint_embed,
                ],
                dim=1,
            )
        else:
            # For non-encoded context, return a JSON string (for visualization if needed).
            observation = Image.fromarray(obs["image"])
            context_dict = {
                "task": task_str,
                "direction": direction_int,
                "carrying object": carrying_str,
                "front object": front_obj_str,
                "hint": hint_str,
            }
            context = json.dumps(context_dict)
        return observation, context

    def uncertainty_score(self, state):
        """
        Compute uncertainty based on the entropy of the softmax over Q-values.
        Assumes 'state' is a tuple (observation, context).
        """
        observation, context = state
        with torch.no_grad():
            q_values = self.model(observation, context)
            q_values = q_values.squeeze(0).cpu().numpy()
        q_values = q_values - np.max(q_values)  # for numerical stability
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
        max_entropy = np.log(len(q_values))
        return entropy / max_entropy

    def compute_distance(self, info):
        """
        Compute the distance between the agent and the target object (first mentioned in the task).
        """
        agent_pos = info["symbolic_state"]["agent"]["pos"]
        carried = info["symbolic_state"]["agent"]["carrying"]
        carried = carried.lower() if carried is not None else None
        task_text = self.env.task.lower()

        first_index = None
        first_distance = None
        for obj in info["symbolic_state"]["objects"] + self.env.rooms:
            obj_name = obj["name"].lower()
            if carried is not None and carried == obj_name:
                continue
            idx = task_text.find(obj_name)
            if idx != -1:
                obj_pos = obj["pos"]
                room_penalty = (
                    5 if info["symbolic_state"]["agent"]["room"] != obj["room"] else 0
                )
                distance = (
                    np.linalg.norm(
                        np.array(agent_pos, dtype=float)
                        - np.array(obj_pos, dtype=float)
                    )
                    + room_penalty
                )
                if first_index is None or idx < first_index:
                    first_index = idx
                    first_distance = distance
        return first_distance if first_distance is not None else 20.0

    def shaped_reward(self, base_reward, info, previous_distance, gamma=1):
        current_distance = self.compute_distance(info)
        potential_current = -current_distance
        potential_previous = -previous_distance
        distance_reward = gamma * potential_current - potential_previous
        return base_reward + self.distance_weight * distance_reward, current_distance

    def choose_action(self, state, obs, info):
        """
        Choose an action using an epsilon-greedy strategy.
        If uncertainty is high and LLM calls are allowed, query the LLM.
        """
        uncertainty = self.uncertainty_score(state)
        cost = 0
        if (
            uncertainty > self.hint_threshold
            and self.num_llm_calls < self.max_llm_calls
        ):
            cost = self.llm_cost * (2**self.num_llm_calls)
            observation, context = self.preprocess_state(
                obs, info, encode_context=False
            )
            hint, hint_uncertainty = self.llm_helper.query_llm(observation, context)
            self.num_llm_calls += 1
            print(
                f"Task: {self.env.task}\nHint: {hint}\nUncertainty: {hint_uncertainty:.2f}"
            )
            self.current_hint = hint
            state = self.preprocess_state(obs, info)
        if random.random() < self.epsilon:
            return self.env.action_space.sample(), cost, state
        with torch.no_grad():
            q_values = self.model(state[0], state[1])
        action = torch.argmax(q_values).item()
        return action, cost, state

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self):
        # Return if no experience is available.
        if not self.replay_buffer:
            return None

        # Using per-task replay buffer
        task_id = random.choice(list(self.replay_buffer.keys()))
        min_batch_size = min(self.batch_size, len(self.replay_buffer[task_id]))
        if min_batch_size < 1:
            return None
        batch = random.sample(self.replay_buffer[task_id], min_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        observation_batch = torch.cat([s[0] for s in states], dim=0)
        context_batch = torch.cat([s[1] for s in states], dim=0)
        next_observation_batch = torch.cat([s[0] for s in next_states], dim=0)
        next_context_batch = torch.cat([s[1] for s in next_states], dim=0)

        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            next_q_values = self.target_model(
                next_observation_batch, next_context_batch
            )
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values

        current_q_values = self.model(observation_batch, context_batch)
        current_q_values = current_q_values.gather(1, actions).squeeze(1)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, episodes=None):
        if episodes is None:
            episodes = self.episodes
        for episode in range(episodes):
            obs, info = self.env.reset()
            self.num_llm_calls = 0
            self.current_hint = ""
            state = self.preprocess_state(obs, info)
            total_reward = 0
            prev_distance = self.compute_distance(info)
            task_id = self.env.task
            if task_id not in self.replay_buffer:
                self.replay_buffer[task_id] = deque(maxlen=self.max_replay_buffer_size)
            if episode % 100 == -1:
                sample_state = self.preprocess_state(obs, info)
                observation, context = sample_state
                with torch.no_grad():
                    q_values = self.model(observation, context)
                print(f"Sample Q-Values at episode {episode}: {q_values}")
                print("Action space:", self.env.action_space.n)
            episode_loss = 0.0
            train_steps = 0
            for step in range(self.env.max_steps):
                action, cost, state = self.choose_action(state, obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
                if reward > 0:
                    self.current_hint = ""
                reward, prev_distance = self.shaped_reward(reward, info, prev_distance)
                reward -= cost
                next_state = self.preprocess_state(obs, info)
                # Update previous transition with current next_state if needed.
                transition = [state, action, reward, next_state, terminated]
                self.replay_buffer[task_id].append(transition)
                state = next_state
                total_reward += reward
                loss_val = self.train_step()
                if loss_val is not None:
                    episode_loss += loss_val
                    train_steps += 1
                if terminated or truncated:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % 5 == 0:
                self.update_target_network()
            avg_loss = episode_loss / train_steps if train_steps > 0 else 0
            print(
                f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.4f}, "
                f"Avg Training Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}"
            )

            # Save checkpoint.
            if (episode + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"model_checkpoint_{episode+1}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

    def test(self, episodes=None, render=False):
        if episodes is None:
            episodes = self.episodes
        total_rewards = []
        for episode in range(episodes):
            obs, info = self.env.reset()
            self.num_llm_calls = 0
            self.current_hint = ""
            total_reward = 0
            state = self.preprocess_state(obs, info)
            for step in range(self.env.max_steps):
                action, cost, state = self.choose_action(state, obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
                if reward > 0:
                    self.current_hint = ""
                reward -= cost
                total_reward += reward
                if render:
                    self.env.render()
                if terminated or truncated:
                    break
            total_rewards.append(total_reward)
            if episode % 100 == 0:
                print(
                    f"Test Episode {episode+1}/{episodes}, Reward: {total_reward:.4f}"
                )
        average_reward = sum(total_rewards) / episodes
        print(f"Average Test Reward over {episodes} episodes: {average_reward:.4f}")
        return average_reward
