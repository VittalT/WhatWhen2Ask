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
from datetime import datetime
import time

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

        # Define dimensions for context components
        # Task branch: first 300 dims
        task_dim = embed_dim  # 300
        # Non-task branch: remaining context
        non_task_dim = (
            4 + 3 * embed_dim
        )  # one-hot (4) + carrying (300) + front object (300) + hint (300) = 904

        # CNN branch - optimized for computational efficiency
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[2], 16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Determine CNN output size using a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape).permute(0, 3, 1, 2)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # Task branch - simplified for efficiency
        self.task_net = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Non-task branch - simplified for efficiency
        self.non_task_net = nn.Sequential(
            nn.Linear(non_task_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion network - simplified
        fused_input_size = cnn_out_size + 128 + 128
        self.fc = nn.Sequential(
            nn.Linear(fused_input_size, 128), nn.ReLU(), nn.Linear(128, num_actions)
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
    def __init__(self, env_name="homegrid-task", episodes=500, checkpoint_dir=None):
        assert checkpoint_dir is not None, "checkpoint_dir must be provided"
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize environment and hyperparameters
        self.env = gym.make(env_name, disable_env_checker=True)
        self.alpha = 0.0005  # Lower learning rate for more stable learning
        self.gamma = 0.99
        self.epsilon = 1.0
        self.batch_size = 128  # Larger batch size for better gradient estimates
        self.episodes = episodes
        self.epsilon_decay = 0.995  # Slower decay helps explore more thoroughly
        self.epsilon_min = 0.05  # Higher minimum exploration rate
        self.llm_cost = 0.01
        self.num_llm_calls = 0
        self.max_llm_calls = 0  # Disable LLM queries for pure DQN training
        self.current_hint = ""

        # Track episode number from previous training
        self.previous_episode = 0

        # Target network update frequency (update every N steps)
        self.target_update_freq = 500
        self.total_steps = 0

        # Memory parameters
        self.replay_buffer = {}
        self.max_replay_buffer_size = 2000  # Reduced for computational efficiency

        # Prioritized experience replay parameters
        self.use_per = True  # Can set to False if computationally expensive
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.001
        self.per_epsilon = 0.01

        # Reward shaping weight
        self.distance_weight = 0.05
        self.additional_weight = 1

        # Get observation shape from the environment (e.g. (96, 96, 3))
        self.obs_shape = self.env.observation_space["image"].shape
        # Number of actions from environment.
        self.output_dim = self.env.action_space.n

        # Initialize models and move them to the appropriate device
        self.model = DQN(self.obs_shape, self.output_dim).to(self.device)
        self.target_model = DQN(self.obs_shape, self.output_dim).to(self.device)
        self.update_target_network()

        # Optimizer with GPU support
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss(
            reduction="none"
        )  # Changed to 'none' for prioritized replay

        # Initialize LLMHelper (assumed implemented in BLIP2Helper)
        self.llm_helper = BLIP2Helper()
        self.hint_threshold = 0.95

        # Checkpoint interval (save model every N episodes)
        self.checkpoint_interval = 250
        # Directory to save checkpoints
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create separate folders for training and testing outputs
        self.training_dir = os.path.join(self.checkpoint_dir, "training")
        self.testing_dir = os.path.join(self.checkpoint_dir, "testing")
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.testing_dir, exist_ok=True)

        # Performance tracking
        self.best_avg_reward = float("-inf")
        self.no_improvement_count = 0

        # Save environment info for reproducibility
        self.env_info = {
            "env_name": env_name,
            "obs_shape": self.obs_shape,
            "action_space": self.output_dim,
            "device": str(self.device),
        }

        # Store potential components for visualization
        self.potential_components = {
            "pot_dist": 0,
            "pot_orientation": 0,
            "pot_carrying": 0,
            "pot_expl": 0,
            "pot_time": 0,
            "weighted_dist": 0,
            "weighted_orientation": 0,
            "weighted_carrying": 0,
            "weighted_expl": 0,
            "weighted_time": 0,
            "current_objective_idx": 0,
            "objectives": [],
        }

    def encode_str(self, text, encode_context=True):
        """
        Convert a string to a 300-dimensional FastText embedding and move to device.
        """
        if encode_context:
            # Get embedding and move to GPU if available
            return get_fasttext_embedding(text).unsqueeze(0).to(self.device)
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
            # Create observation tensor and move to device
            observation = (
                torch.FloatTensor(obs["image"] / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            # Create embeddings and context vectors on the device
            task_embed = self.encode_str(task_str, encode_context)  # (1, 300)

            direction_one_hot = torch.zeros(1, 4, device=self.device)
            direction_one_hot[0, direction_int] = 1.0

            carrying_embed = self.encode_str(carrying_str, encode_context)  # (1, 300)
            front_obj_embed = self.encode_str(front_obj_str, encode_context)  # (1, 300)
            hint_embed = self.encode_str(hint_str, encode_context)  # (1, 300)

            # Concatenate with task embedding at the start
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
            # For non-encoded context, return a JSON string (for visualization if needed)
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
        Uses GPU for computation then transfers back to CPU for numpy operations.
        """
        observation, context = state
        with torch.no_grad():
            q_values = self.model(observation, context)
            # Move to CPU for numpy operations
            q_values = q_values.squeeze(0).cpu().numpy()

        # Compute entropy (on CPU with numpy)
        q_values = q_values - np.max(q_values)  # for numerical stability
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
        max_entropy = np.log(len(q_values))

        return entropy / max_entropy

    def reset_episode(self, info):
        """Reset episode-specific variables and parse task objectives"""
        # Reset variables
        self.visited_rooms = {info["symbolic_state"]["agent"]["room"].lower()}
        self.visited_cells = {tuple(info["symbolic_state"]["agent"]["pos"])}
        self.current_step = 0
        self.previous_potential = None

    def _find_matching_objects(self, info):
        """
        Find all objects in the environment mentioned in the text description
        and sort them by their position in the text
        """
        matches = []
        text = self.env.task.lower()

        # Check all objects and rooms
        for obj in info["symbolic_state"]["objects"] + self.env.rooms:
            obj_name = obj["name"].lower()
            if obj_name in text:
                # Find position in the text
                idx = text.find(obj_name)
                matches.append((idx, obj))

        # Sort by position in text (earlier mentions first)
        matches.sort(key=lambda x: x[0])

        # Return just the objects in order
        return [obj for _, obj in matches]

    def compute_potential(self, info):
        """Compute a potential function for reward shaping based on objectives"""
        # Get agent's current state
        agent_pos = info["symbolic_state"]["agent"]["pos"]
        agent_room = info["symbolic_state"]["agent"]["room"].lower()
        agent_dir = info["symbolic_state"]["agent"]["dir"]
        carrying = info["symbolic_state"]["agent"]["carrying"]
        self.objectives = self._find_matching_objects(info)

        # Update visited rooms
        self.visited_rooms.add(agent_room)
        self.visited_cells.add(tuple(agent_pos))

        # Start with base potential
        potential = 0

        # Determine the current objective index
        current_objective_idx = 0

        # Check if carrying first objective (in a multi-objective task)
        carrying_first_obj = (
            carrying is not None
            and carrying.lower() == self.objectives[0]["name"].lower()
            and len(self.objectives) > 1
        )

        # If carrying the first object and there's more objectives, focus on the next one
        if carrying_first_obj:
            current_objective_idx = 1

        pot_dist, pot_orientation, pot_carrying = 0, 0, 0

        # If we have a valid objective at the current index
        if current_objective_idx < len(self.objectives):
            # Room penalty if in different room
            obj = self.objectives[current_objective_idx]
            obj_pos = obj["pos"]

            remaining_objs = self.objectives[current_objective_idx:]
            positions = [agent_pos] + [obj["pos"] for obj in remaining_objs]
            pot_dist = sum(
                np.linalg.norm(
                    np.array(positions[i], float) - np.array(positions[i + 1], float)
                )
                for i in range(len(positions) - 1)
            )

            # Orientation component
            dx = obj_pos[0] - agent_pos[0]
            dy = obj_pos[1] - agent_pos[1]
            agent_angles = [0, 90, 180, 270]  # 0=right, 1=down, 2=left, 3=up
            agent_angle = agent_angles[agent_dir]
            obj_angle = np.degrees(np.arctan2(dy, dx)) % 360
            signed_diff = abs(agent_angle - obj_angle)
            angle_diff = min(signed_diff, 360 - signed_diff)
            pot_orientation = max(0, 1 - angle_diff / 180)

            if carrying_first_obj:
                pot_carrying = 1

        # pot_rooms = len(self.visited_rooms)
        pot_expl = len(self.visited_cells)
        pot_time = self.current_step

        # Calculate weighted components
        weighted_dist = -0.05 * pot_dist  # negative because closer is better
        weighted_orientation = 0.1 * pot_orientation
        weighted_carrying = 0.5 * pot_carrying
        weighted_expl = 0.025 * pot_expl
        weighted_time = -0.025 * pot_time  # penalize as time goes

        # Store components for visualization
        self.potential_components = {
            "pot_dist": pot_dist,
            "pot_orientation": pot_orientation,
            "pot_carrying": pot_carrying,
            "pot_expl": pot_expl,
            "pot_time": pot_time,
            "weighted_dist": weighted_dist,
            "weighted_orientation": weighted_orientation,
            "weighted_carrying": weighted_carrying,
            "weighted_expl": weighted_expl,
            "weighted_time": weighted_time,
            "current_objective_idx": current_objective_idx,
            "objectives": self.objectives,
        }

        potential = (
            weighted_dist
            + weighted_orientation
            + weighted_carrying
            + weighted_expl
            + weighted_time
        )

        return potential

    def get_potential_components(self):
        """Return the previously calculated potential components"""
        return self.potential_components

    def shaped_reward(self, base_reward, info, gamma=0.99):
        """
        Pure potential-based reward shaping using Φ(s) potential function.
        Formula: F(s,s') = γΦ(s') - Φ(s)
        """
        # Calculate current potential
        current_potential = self.compute_potential(info)

        # First call in episode
        if self.previous_potential is None:
            self.previous_potential = current_potential
            return base_reward, base_reward

        # Calculate shaping
        shaping = gamma * current_potential - self.previous_potential

        # Update for next call
        self.previous_potential = current_potential

        # Return shaped reward and original reward
        return base_reward + shaping, base_reward

    def choose_action(self, state, obs, info, testing=False):
        """
        Choose an action using an epsilon-greedy strategy.
        If uncertainty is high and LLM calls are allowed, query the LLM.
        Uses GPU for tensor operations.

        Args:
            state: Preprocessed state tuple (observation, context)
            obs: Raw observation from environment
            info: Info dict from environment
            testing: If True, disable exploration and always choose greedy action
        """
        uncertainty = self.uncertainty_score(state)
        cost = 0

        # Check if we should use LLM hints
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

        # Epsilon-greedy action selection (skip if testing)
        if not testing and random.random() < self.epsilon:
            return self.env.action_space.sample(), cost, state

        # Greedy action selection using GPU
        with torch.no_grad():
            q_values = self.model(state[0], state[1])
            # Use tensor operations on GPU, only get final action as a CPU item
            action = torch.argmax(q_values, dim=1).item()

        return action, cost, state

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def add_experience(self, task_id, experience, error=None):
        """Add an experience to the replay buffer with priority based on TD error"""
        if task_id not in self.replay_buffer:
            self.replay_buffer[task_id] = {
                "experiences": deque(maxlen=self.max_replay_buffer_size),
                "priorities": deque(maxlen=self.max_replay_buffer_size),
            }

        # If error is not provided, give it maximum priority (for new experiences)
        if error is None:
            max_priority = 1.0
            if self.replay_buffer[task_id]["priorities"]:
                max_priority = max(self.replay_buffer[task_id]["priorities"])
            self.replay_buffer[task_id]["priorities"].append(max_priority)
        else:
            # Priority based on TD error
            priority = (abs(error) + self.per_epsilon) ** self.per_alpha
            self.replay_buffer[task_id]["priorities"].append(priority)

        self.replay_buffer[task_id]["experiences"].append(experience)

    def sample_batch(self, task_id):
        """Sample a batch of experiences using prioritized experience replay with error handling"""
        if task_id not in self.replay_buffer:
            return None, None, None

        if not self.replay_buffer[task_id]["experiences"]:
            return None, None, None

        buffer_size = len(self.replay_buffer[task_id]["experiences"])
        if buffer_size < 1:
            return None, None, None

        # Use smaller batch size if buffer is small
        actual_batch_size = min(self.batch_size, buffer_size)

        # PER sampling
        if self.use_per and len(self.replay_buffer[task_id]["priorities"]) == len(
            self.replay_buffer[task_id]["experiences"]
        ):
            priorities = np.array(self.replay_buffer[task_id]["priorities"])
            # Handle zero sum
            if np.sum(priorities) <= 0:
                # Fallback to uniform sampling
                indices = np.random.choice(
                    buffer_size, actual_batch_size, replace=False
                )
                weights = np.ones(actual_batch_size)
            else:
                probs = priorities / np.sum(priorities)
                indices = np.random.choice(
                    buffer_size, actual_batch_size, p=probs, replace=False
                )

                # Calculate importance sampling weights
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
                weights = (buffer_size * probs[indices]) ** (-self.per_beta)
                weights /= np.max(weights) if np.max(weights) > 0 else 1.0  # Normalize

            # Get selected experiences
            batch = [self.replay_buffer[task_id]["experiences"][i] for i in indices]
            return batch, indices, weights
            # except Exception as e:
            #     print(f"PER sampling error: {e}, using uniform sampling instead")
            #     # Fallback to uniform sampling on error
            #     batch = random.sample(
            #         list(self.replay_buffer[task_id]["experiences"]), actual_batch_size
            #     )
            #     return batch, None, None
        else:
            # Uniform sampling if PER is disabled or sizes don't match
            batch = random.sample(
                list(self.replay_buffer[task_id]["experiences"]), actual_batch_size
            )
            return batch, None, None

    def update_priorities(self, task_id, indices, td_errors):
        """Update priorities in the replay buffer based on TD errors"""
        if not self.use_per or indices is None:
            return

        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.per_epsilon) ** self.per_alpha
            self.replay_buffer[task_id]["priorities"][idx] = priority

    def train_step(self):
        # Return if no experience is available.
        if not self.replay_buffer:
            return None

        # Using per-task replay buffer
        task_id = random.choice(list(self.replay_buffer.keys()))

        # Sample batch using prioritized experience replay
        batch, indices, weights = self.sample_batch(task_id)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = zip(*batch)

        # Move all tensors to the device (GPU)
        observation_batch = torch.cat([s[0] for s in states], dim=0)
        context_batch = torch.cat([s[1] for s in states], dim=0)
        next_observation_batch = torch.cat([s[0] for s in next_states], dim=0)
        next_context_batch = torch.cat([s[1] for s in next_states], dim=0)

        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Convert importance sampling weights to tensor if using PER
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_model(
                next_observation_batch, next_context_batch
            )
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values

        current_q_values = self.model(observation_batch, context_batch)
        current_q_values = current_q_values.gather(1, actions).squeeze(1)

        # Calculate TD errors (used for updating priorities)
        td_errors = target_q_values - current_q_values

        # Apply weights to the loss if using PER
        if weights is not None:
            # Element-wise loss
            losses = self.loss_fn(current_q_values, target_q_values)
            # Apply importance sampling weights
            loss = torch.mean(weights * losses)
        else:
            # Regular MSE loss (mean is applied here)
            loss = torch.mean(self.loss_fn(current_q_values, target_q_values))

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities in replay buffer if using PER
        if self.use_per and indices is not None:
            # Get TD errors as numpy array - move back to CPU for numpy operations
            td_errors_np = td_errors.detach().cpu().numpy()
            self.update_priorities(task_id, indices, td_errors_np)

        return loss.item()

    def train(self, episodes=None):
        if episodes is None:
            episodes = self.episodes

        # For tracking performance trends
        rewards_history = []
        original_rewards_history = []  # Track original rewards instead of success rate

        # For tracking runtime
        training_start_time = time.time()

        # Early stopping variables
        best_reward = float("-inf")
        no_improvement_count = 0

        for episode in range(episodes):
            # Calculate actual episode number including previous training
            actual_episode = episode + self.previous_episode

            episode_start_time = time.time()
            obs, info = self.env.reset()

            # Reset episode-specific variables and parse objectives
            self.reset_episode(info)

            self.num_llm_calls = 0
            self.current_hint = ""

            state = self.preprocess_state(obs, info)
            total_reward = 0
            original_reward_sum = 0  # Track actual rewards separately
            task_id = self.env.task
            if task_id not in self.replay_buffer:
                self.replay_buffer[task_id] = {
                    "experiences": deque(maxlen=self.max_replay_buffer_size),
                    "priorities": deque(maxlen=self.max_replay_buffer_size),
                }

            # Verbose logging every 500 episodes or as needed
            verbose = episode % 500 == 0
            if verbose:
                print(
                    f"\nEpisode {actual_episode + 1}/{self.previous_episode + episodes}, Task: {task_id}"
                )
                print(f"Current epsilon: {self.epsilon:.3f}")

            episode_loss = 0.0
            train_steps = 0

            for step in range(self.env.max_steps):
                self.current_step += 1
                action, cost, state = self.choose_action(state, obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)

                self.total_steps += 1

                # Track if we got a success reward
                if reward > 0:
                    self.current_hint = ""
                    if verbose:
                        print(f"  Step {step}: Got reward {reward}! Success!")

                # Apply reward shaping but keep track of original reward
                shaped_reward, original_reward = self.shaped_reward(reward, info)
                shaped_reward -= cost
                original_reward -= cost

                next_state = self.preprocess_state(obs, info)

                # Create transition and add to replay buffer with maximum priority initially
                transition = [state, action, shaped_reward, next_state, terminated]
                self.add_experience(task_id, transition)

                state = next_state
                total_reward += shaped_reward
                original_reward_sum += original_reward

                # Train less frequently to speed up training (every 4 steps)
                if self.total_steps % 4 == 0:
                    loss_val = self.train_step()
                    if loss_val is not None:
                        episode_loss += loss_val
                        train_steps += 1

                # Update target network less frequently to speed up training
                if self.total_steps % self.target_update_freq == 0:
                    self.update_target_network()
                    if verbose:
                        print(f"  Updated target network at step {self.total_steps}")

                if terminated or truncated:
                    break

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Calculate average loss if we did any training steps
            avg_loss = episode_loss / max(1, train_steps)
            rewards_history.append(total_reward)
            original_rewards_history.append(original_reward_sum)

            # Calculate episode runtime
            episode_time = time.time() - episode_start_time
            total_time = time.time() - training_start_time
            avg_time_per_episode = total_time / (episode + 1)

            # Log less frequently to speed up training
            if (episode + 1) % 100 == 0 or episode == 0 or episode == episodes - 1:
                recent_rewards = rewards_history[-min(100, len(rewards_history)) :]
                recent_original_rewards = original_rewards_history[
                    -min(100, len(original_rewards_history)) :
                ]
                avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
                avg_recent_original_reward = sum(recent_original_rewards) / len(
                    recent_original_rewards
                )

                print(
                    f"Episode {actual_episode + 1}/{self.previous_episode + episodes}, "
                    f"Time: {episode_time:.2f}s (Avg: {avg_time_per_episode:.2f}s), "
                    f"Shaped Reward: {total_reward:.4f}, "
                    f"Original Reward: {original_reward_sum:.2f}, "
                    f"Avg Recent Reward: {avg_recent_reward:.4f}, "
                    f"Avg Recent Original Reward: {avg_recent_original_reward:.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

                # Early stopping check - use average reward instead of success rate
                if avg_recent_reward > best_reward:
                    best_reward = avg_recent_reward
                    no_improvement_count = 0

                    # Save best model if significantly better, check average reward
                    if (
                        len(rewards_history) >= 100
                        and avg_recent_reward > self.best_avg_reward
                    ):
                        self.best_avg_reward = avg_recent_reward
                        best_model_path = os.path.join(
                            self.training_dir, "best_model.pth"
                        )
                        model_data = {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epsilon": self.epsilon,
                            "episode": actual_episode
                            + 1,  # Store the next episode number
                            "total_steps": self.total_steps,
                            "avg_reward": avg_recent_reward,
                            "env_info": self.env_info,
                        }
                        torch.save(model_data, best_model_path)
                        print(
                            f"New best model saved with average reward: {avg_recent_reward:.4f}"
                        )
                else:
                    no_improvement_count += 1

                    # If no improvement for a while, reduce learning rate
                    if no_improvement_count >= 5:
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] *= 0.8
                        print(
                            f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6f}"
                        )
                        no_improvement_count = 0

            # Save checkpoint less frequently to speed up training
            if (episode + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.training_dir, f"model_checkpoint_{actual_episode+1}.pth"
                )
                checkpoint_data = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "episode": actual_episode + 1,  # Store the next episode number
                    "total_steps": self.total_steps,
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

                # Plot learning curve at checkpoints
                if len(rewards_history) > 0:
                    plt.figure(figsize=(12, 8))

                    # Plot both reward and original reward as moving averages
                    plt.subplot(2, 1, 1)

                    # Plot raw rewards in light color
                    plt.plot(rewards_history, "b-", alpha=0.3, label="Raw Rewards")

                    # Calculate moving average of rewards
                    if len(rewards_history) > 100:
                        window_size = 100
                        reward_moving_avg = [
                            sum(rewards_history[i : i + window_size]) / window_size
                            for i in range(len(rewards_history) - window_size + 1)
                        ]

                        plt.plot(
                            range(window_size - 1, episode + 1),
                            reward_moving_avg,
                            "b-",
                            linewidth=2,
                            label=f"Moving Avg (Window={window_size})",
                        )

                    plt.title(f"Learning Curve - Rewards (Episode {actual_episode+1})")
                    plt.ylabel("Shaped Reward")
                    plt.legend()

                    # Plot original rewards
                    plt.subplot(2, 1, 2)

                    # Plot raw original rewards in light color
                    plt.plot(
                        original_rewards_history,
                        "c-",
                        alpha=0.3,
                        label="Raw Original Rewards",
                    )

                    # Calculate moving average of original rewards
                    if len(original_rewards_history) > 100:
                        window_size = 100
                        original_reward_moving_avg = [
                            sum(original_rewards_history[i : i + window_size])
                            / window_size
                            for i in range(
                                len(original_rewards_history) - window_size + 1
                            )
                        ]

                        plt.plot(
                            range(window_size - 1, episode + 1),
                            original_reward_moving_avg,
                            "c-",
                            linewidth=2,
                            label=f"Moving Avg (Window={window_size})",
                        )

                        plt.title(f"Original Reward (Episode {actual_episode+1})")
                        plt.xlabel("Episode")
                        plt.ylabel("Original Reward")
                        plt.legend()

                    plt.tight_layout()
                    plot_path = os.path.join(
                        self.training_dir, f"learning_curve_{actual_episode+1}.png"
                    )
                    plt.savefig(plot_path)
                    plt.close()

    def test(self, episodes=None, render=False):
        """Test the agent's performance over multiple episodes using both original and shaped rewards"""
        if episodes is None:
            episodes = self.episodes

        total_rewards = []  # Original rewards
        shaped_rewards = []  # Shaped rewards
        steps_to_success = []
        task_success_rates = {}

        for episode in range(episodes):
            obs, info = self.env.reset()

            # Reset episode-specific variables and parse objectives
            self.reset_episode(info)

            self.num_llm_calls = 0
            self.current_hint = ""

            # Track current task
            current_task = self.env.task
            if current_task not in task_success_rates:
                task_success_rates[current_task] = {"attempts": 0, "successes": 0}
            task_success_rates[current_task]["attempts"] += 1

            total_reward = 0  # Original reward
            total_shaped_reward = 0  # Shaped reward
            state = self.preprocess_state(obs, info)

            for step in range(self.env.max_steps):
                self.current_step += 1
                action, cost, state = self.choose_action(state, obs, info, testing=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Track success without using shaped rewards
                if reward > 0:
                    self.current_hint = ""
                    steps_to_success.append(step)
                    # Track task-specific success
                    task_success_rates[current_task]["successes"] += 1

                # Get shaped reward for tracking
                shaped_reward, original_reward = self.shaped_reward(reward, info)

                # Subtract costs from both
                shaped_reward -= cost
                original_reward -= cost

                # Track both types of rewards
                total_reward += original_reward
                total_shaped_reward += shaped_reward

                if render:
                    self.env.render()

                if terminated or truncated:
                    break

            # Record results
            total_rewards.append(total_reward)
            shaped_rewards.append(total_shaped_reward)

            # Log progress periodically
            if episode % 1000 == 0 or episode == episodes - 1:
                success_rate = (
                    task_success_rates[current_task]["successes"] / (episode + 1) * 100
                )
                avg_steps = np.mean(steps_to_success) if steps_to_success else "N/A"
                print(
                    f"Test Episode {episode+1}/{episodes}, "
                    f"Original Reward: {total_reward:.4f}, "
                    f"Shaped Reward: {total_shaped_reward:.4f}, "
                    f"Success Rate: {success_rate:.1f}%, "
                    f"Avg Steps to Success: {avg_steps}"
                )

        # Calculate final metrics
        average_reward = sum(total_rewards) / max(1, len(total_rewards))
        average_shaped_reward = sum(shaped_rewards) / max(1, len(shaped_rewards))

        # Calculate success rate from task_success_rates
        success_count = sum(stats["successes"] for stats in task_success_rates.values())
        attempt_count = sum(stats["attempts"] for stats in task_success_rates.values())
        final_success_rate = (success_count / max(1, attempt_count)) * 100

        # Calculate average steps to success
        avg_steps_to_success = np.mean(steps_to_success) if steps_to_success else "N/A"

        print(f"\nTest Results over {len(total_rewards)} episodes:")
        print(f"  Average Original Reward: {average_reward:.4f}")
        print(f"  Average Shaped Reward: {average_shaped_reward:.4f}")
        print(f"  Success Rate: {final_success_rate:.1f}%")
        print(f"  Average Steps to Success: {avg_steps_to_success}")

        # Print task-specific success rates
        print("\nTask-specific success rates:")
        for task, stats in task_success_rates.items():
            task_success_rate = (
                (stats["successes"] / stats["attempts"]) * 100
                if stats["attempts"] > 0
                else 0
            )
            print(
                f"  {task}: {task_success_rate:.1f}% ({stats['successes']}/{stats['attempts']})"
            )

        # Save test results to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.testing_dir, f"test_results_{timestamp}.json")

        results = {
            "average_original_reward": float(average_reward),
            "average_shaped_reward": float(average_shaped_reward),
            "success_rate": float(final_success_rate),
            "average_steps_to_success": (
                float(avg_steps_to_success)
                if isinstance(avg_steps_to_success, (int, float))
                else None
            ),
            "task_success_rates": {
                task: {
                    "success_rate": (
                        (stats["successes"] / stats["attempts"]) * 100
                        if stats["attempts"] > 0
                        else 0
                    ),
                    "successes": stats["successes"],
                    "attempts": stats["attempts"],
                }
                for task, stats in task_success_rates.items()
            },
            "timestamp": timestamp,
            "episodes": episodes,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nTest results saved to: {results_file}")

        return average_reward, average_shaped_reward
