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
    def __init__(self, env_name="homegrid-task", episodes=500):
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize environment and hyperparameters
        self.env = gym.make(env_name, disable_env_checker=True)
        self.alpha = 0.0005  # Lower learning rate for more stable learning
        self.gamma = 0.99
        self.epsilon = 1.0

        # Memory optimization: Adjust batch size based on available GPU memory
        if torch.cuda.is_available():
            # Get GPU memory information
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Available GPU memory: {gpu_mem_gb:.1f} GB")

            # Scale batch size based on available memory
            if gpu_mem_gb > 20:  # High-end GPU
                self.batch_size = 128
            elif gpu_mem_gb > 10:  # Mid-range GPU
                self.batch_size = 96
            elif gpu_mem_gb > 6:  # Lower-end GPU
                self.batch_size = 64
            else:  # Very limited GPU memory
                self.batch_size = 32

            print(f"Using batch size: {self.batch_size}")
        else:
            self.batch_size = 128  # CPU default

        self.episodes = episodes
        self.epsilon_decay = 0.995  # Slower decay helps explore more thoroughly
        self.epsilon_min = 0.05  # Higher minimum exploration rate
        self.llm_cost = 0.01
        self.num_llm_calls = 0
        self.max_llm_calls = 0  # Disable LLM queries for pure DQN training
        self.current_hint = ""

        # Target network update frequency (update every N steps)
        self.target_update_freq = 500
        self.total_steps = 0

        # Memory parameters - reduce buffer size to save memory
        self.replay_buffer = {}
        # Memory optimization: Scale buffer size based on available memory
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory / (1024**3) < 12
        ):
            self.max_replay_buffer_size = 5000  # Smaller buffer for limited memory
        else:
            self.max_replay_buffer_size = 10000  # Standard size

        print(f"Using replay buffer size: {self.max_replay_buffer_size}")

        # Prioritized experience replay parameters
        self.use_per = True  # Can set to False if computationally expensive
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.001
        self.per_epsilon = 0.01

        # Reward shaping weight
        self.distance_weight = 0.05

        # Track previous room for room change bonus
        self.prev_room = None

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
        self.checkpoint_dir = "checkpoints8"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Performance tracking
        self.best_avg_reward = float("-inf")
        self.no_improvement_count = 0

        # Memory management - clean up torch cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB"
            )

        # Save environment info for reproducibility
        self.env_info = {
            "env_name": env_name,
            "obs_shape": self.obs_shape,
            "action_space": self.output_dim,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "replay_buffer_size": self.max_replay_buffer_size,
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

        Memory optimized version to prevent GPU memory leaks.
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
            # Create observation tensor and move to device - normalize in-place where possible
            # Convert to float32 to save memory (instead of float64)
            obs_array = obs["image"] / 255.0
            observation = (
                torch.FloatTensor(obs_array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            # Get embeddings using context-specific methods to save memory
            task_embed = self.encode_str(task_str, encode_context)  # (1, 300)

            # Create direction one-hot directly on device to avoid CPU->GPU transfer
            direction_one_hot = torch.zeros(1, 4, device=self.device)
            direction_one_hot[0, direction_int] = 1.0

            # Encode other embeddings
            carrying_embed = self.encode_str(carrying_str, encode_context)  # (1, 300)
            front_obj_embed = self.encode_str(front_obj_str, encode_context)  # (1, 300)
            hint_embed = self.encode_str(hint_str, encode_context)  # (1, 300)

            # Concatenate with task embedding at the start - do this once to avoid multiple concatenations
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

            # Optional memory cleanup for large tensors
            del task_embed, carrying_embed, front_obj_embed, hint_embed

            return observation, context
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

    def compute_distance(self, info):
        """
        Compute the distance between the agent and task-relevant objects with improved handling
        for different task structures.
        """
        agent_pos = info["symbolic_state"]["agent"]["pos"]
        agent_room = info["symbolic_state"]["agent"]["room"].lower()
        carried_obj = info["symbolic_state"]["agent"]["carrying"]
        carried_name = carried_obj.lower() if carried_obj is not None else None
        task_text = self.env.task.lower()

        # Try to extract source and destination objects from task
        task_parts = task_text.split(" to ")
        source_part = task_parts[0]
        dest_part = task_parts[1] if len(task_parts) > 1 else ""

        # Determine which objects we should be tracking distances to
        target_distances = {}

        # Process all objects in the environment
        for obj in info["symbolic_state"]["objects"] + self.env.rooms:
            obj_name = obj["name"].lower()
            obj_pos = obj["pos"]
            obj_room = (
                obj["room"].lower() if "room" in obj and obj["room"] is not None else ""
            )

            # Skip the object if we're already carrying it
            if carried_name is not None and carried_name == obj_name:
                continue

            # Calculate basic distance
            try:
                euclidean_dist = np.linalg.norm(
                    np.array(agent_pos, dtype=float) - np.array(obj_pos, dtype=float)
                )
            except:
                # Fallback if positions are invalid
                euclidean_dist = 10.0

            # Add room penalty if object is in a different room
            room_penalty = 5 if obj_room and obj_room != agent_room else 0
            total_dist = euclidean_dist + room_penalty

            # Check if this object is relevant to the task
            in_source = obj_name in source_part
            in_dest = obj_name in dest_part

            if in_source or in_dest:
                # Set distance with a priority flag: source objects first, then destination
                priority = 0 if in_source else 1
                target_distances[(priority, obj_name)] = total_dist

        # If we're carrying the source object and there's a destination, prioritize destination distances
        if carried_name is not None and carried_name in source_part and dest_part:
            # Find the closest destination-related object
            dest_distances = {k: v for k, v in target_distances.items() if k[0] == 1}
            if dest_distances:
                closest_dest = min(dest_distances.items(), key=lambda x: x[1])
                return closest_dest[1]

        # If we haven't returned yet, use the first mentioned relevant object
        if target_distances:
            # Sort by priority first (source=0, dest=1), then by order of mention in the task
            sorted_distances = sorted(
                target_distances.items(),
                key=lambda x: (x[0][0], task_text.find(x[0][1])),
            )
            return sorted_distances[0][1]

        # Fallback distance if no relevant objects found
        return 20.0

    def shaped_reward(self, base_reward, info, previous_distance, gamma=0.99):
        """Enhanced reward shaping optimized for sparse rewards environment"""
        # Get current distance to relevant objects
        current_distance = self.compute_distance(info)

        # Base distance-based shaping using potential-based approach (ensures consistency)
        potential_current = (
            -current_distance
        )  # Negative because smaller distance is better
        potential_previous = -previous_distance
        distance_reward = gamma * potential_current - potential_previous

        # Get task-related information
        task_text = self.env.task.lower()
        carrying = info["symbolic_state"]["agent"]["carrying"]
        carrying_str = carrying.lower() if carrying is not None else ""
        front_obj = info["symbolic_state"]["front_obj"] or ""
        front_obj_str = front_obj.lower()
        current_room = info["symbolic_state"]["agent"]["room"]

        # Create a more detailed shaping reward
        additional_reward = 0

        # Significant progress markers get bigger rewards

        # 1. Carrying task-relevant object
        if carrying is not None and carrying_str in task_text:
            # Check if this object is the first mentioned in the task (likely the most important)
            if (
                task_text.find(carrying_str) < task_text.find("to")
                and "to" in task_text
            ):
                additional_reward += 0.2  # Higher reward for the main object
            else:
                additional_reward += 0.1

        # 2. Facing task-relevant object
        if front_obj_str and front_obj_str in task_text:
            # If not carrying anything and facing key object, bigger reward
            if (
                carrying is None
                and task_text.find(front_obj_str) < task_text.find("to")
                and "to" in task_text
            ):
                additional_reward += 0.1
            else:
                additional_reward += 0.05

        # 3. Room exploration and navigation
        if self.prev_room is not None and self.prev_room != current_room:
            # Bigger reward for first room change to encourage exploration
            if self.total_steps < 100:
                additional_reward += 0.3
            else:
                additional_reward += 0.15

        # 4. Getting close to target locations (if task has "to" directive)
        if "to" in task_text:
            # Extract destination
            destination_part = task_text.split("to")[1].strip()
            if current_room.lower() in destination_part:
                additional_reward += 0.1  # Reward for being in target room

        # Ensure very positive base rewards remain dominant
        if base_reward > 0.5:
            shaped_reward = base_reward
        else:
            # Apply shaping only for non-success states
            shaped_reward = (
                base_reward
                + (self.distance_weight * distance_reward)
                + additional_reward
            )

        # Update previous room for next step
        self.prev_room = current_room

        return shaped_reward, current_distance

    def choose_action(self, state, obs, info):
        """
        Choose an action using an epsilon-greedy strategy.
        If uncertainty is high and LLM calls are allowed, query the LLM.
        Uses GPU for tensor operations.
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

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
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
                try:
                    max_priority = max(self.replay_buffer[task_id]["priorities"])
                except:
                    max_priority = 1.0
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
            try:
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
                    weights /= (
                        np.max(weights) if np.max(weights) > 0 else 1.0
                    )  # Normalize

                # Get selected experiences
                batch = [self.replay_buffer[task_id]["experiences"][i] for i in indices]
                return batch, indices, weights
            except Exception as e:
                print(f"PER sampling error: {e}, using uniform sampling instead")
                # Fallback to uniform sampling on error
                batch = random.sample(
                    list(self.replay_buffer[task_id]["experiences"]), actual_batch_size
                )
                return batch, None, None
        else:
            # Uniform sampling if PER is disabled or sizes don't match
            try:
                batch = random.sample(
                    list(self.replay_buffer[task_id]["experiences"]), actual_batch_size
                )
                return batch, None, None
            except:
                print(f"Sampling error, buffer size: {buffer_size}")
                return None, None, None

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

        # Use with torch.no_grad() for target calculation to save memory
        with torch.no_grad():
            next_q_values = self.target_model(
                next_observation_batch, next_context_batch
            )
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values
            # Detach to free memory
            target_q_values = target_q_values.detach()
            # Clean up intermediate tensors
            del next_q_values, max_next_q_values

        # Compute current Q-values
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
            # Clean up
            del losses
        else:
            # Regular MSE loss (mean is applied here)
            loss = torch.mean(self.loss_fn(current_q_values, target_q_values))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Get loss value before cleanup
        loss_value = loss.item()

        # Update priorities in replay buffer if using PER
        if self.use_per and indices is not None:
            # Get TD errors as numpy array - move back to CPU for numpy operations
            td_errors_np = td_errors.detach().cpu().numpy()
            self.update_priorities(task_id, indices, td_errors_np)
            del td_errors_np

        # Clean up all tensors to free memory
        del observation_batch, context_batch, next_observation_batch, next_context_batch
        del actions, rewards, dones, target_q_values, current_q_values, td_errors
        if weights is not None:
            del weights
        del loss

        # Force garbage collection occasionally
        if self.total_steps % 1000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss_value

    def train(self, episodes=None):
        if episodes is None:
            episodes = self.episodes

        # For tracking performance trends
        rewards_history = []

        # Early stopping variables
        best_reward = float("-inf")
        no_improvement_count = 0

        # Memory management: Add periodic memory cleanup counter
        memory_cleanup_freq = 10  # Clean memory every N episodes

        try:
            for episode in range(episodes):
                obs, info = self.env.reset()
                self.num_llm_calls = 0
                self.current_hint = ""
                # Reset previous room tracking
                self.prev_room = info["symbolic_state"]["agent"]["room"]

                state = self.preprocess_state(obs, info)
                total_reward = 0
                prev_distance = self.compute_distance(info)
                task_id = self.env.task
                if task_id not in self.replay_buffer:
                    self.replay_buffer[task_id] = {
                        "experiences": deque(maxlen=self.max_replay_buffer_size),
                        "priorities": deque(maxlen=self.max_replay_buffer_size),
                    }

                # Verbose logging every 500 episodes or as needed
                verbose = episode % 500 == 0
                if verbose:
                    print(f"\nEpisode {episode + 1}/{episodes}, Task: {task_id}")
                    print(f"Current epsilon: {self.epsilon:.3f}")

                episode_loss = 0.0
                train_steps = 0

                for step in range(self.env.max_steps):
                    action, cost, state = self.choose_action(state, obs, info)
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    self.total_steps += 1

                    if reward > 0:
                        self.current_hint = ""
                        if verbose:
                            print(f"  Step {step}: Got reward {reward}! Success!")

                    # Apply reward shaping
                    reward, prev_distance = self.shaped_reward(
                        reward, info, prev_distance
                    )
                    reward -= cost

                    next_state = self.preprocess_state(obs, info)

                    # Create transition and add to replay buffer with maximum priority initially
                    transition = [state, action, reward, next_state, terminated]
                    self.add_experience(task_id, transition)

                    state = next_state
                    total_reward += reward

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
                            print(
                                f"  Updated target network at step {self.total_steps}"
                            )

                    if terminated or truncated:
                        break

                # Decay epsilon after each episode
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                # Calculate average loss if we did any training steps
                avg_loss = episode_loss / max(1, train_steps)
                rewards_history.append(total_reward)

                # Log less frequently to speed up training
                if (episode + 1) % 100 == 0:
                    recent_rewards = rewards_history[-min(10, len(rewards_history)) :]
                    avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
                    print(
                        f"Episode {episode + 1}/{episodes}, "
                        f"Total Reward: {total_reward:.4f}, "
                        f"Avg Recent Reward: {avg_recent_reward:.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Epsilon: {self.epsilon:.3f}"
                    )

                    # Early stopping check
                    if avg_recent_reward > best_reward:
                        best_reward = avg_recent_reward
                        no_improvement_count = 0

                        # Save best model if significantly better
                        if (
                            len(rewards_history) >= 10
                            and avg_recent_reward > self.best_avg_reward
                        ):
                            self.best_avg_reward = avg_recent_reward
                            best_model_path = os.path.join(
                                self.checkpoint_dir, "best_model.pth"
                            )
                            torch.save(
                                {
                                    "model_state_dict": self.model.state_dict(),
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "epsilon": self.epsilon,
                                    "episode": episode,
                                    "total_steps": self.total_steps,
                                    "avg_reward": avg_recent_reward,
                                    "env_info": self.env_info,
                                },
                                best_model_path,
                            )
                            print(
                                f"New best model saved with avg reward: {avg_recent_reward:.4f}"
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
                        self.checkpoint_dir, f"model_checkpoint_{episode+1}.pth"
                    )
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epsilon": self.epsilon,
                            "episode": episode,
                            "total_steps": self.total_steps,
                        },
                        checkpoint_path,
                    )
                    print(f"Saved checkpoint: {checkpoint_path}")

                    # Plot learning curve at checkpoints
                    if len(rewards_history) > 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(rewards_history)
                        plt.title(f"Learning Curve (Episode {episode+1})")
                        plt.xlabel("Episode")
                        plt.ylabel("Total Reward")
                        plt.savefig(
                            os.path.join(
                                self.checkpoint_dir, f"learning_curve_{episode+1}.png"
                            )
                        )
                        plt.close()

                # Periodically clean up GPU memory to prevent leaks
                if torch.cuda.is_available() and episode % memory_cleanup_freq == 0:
                    torch.cuda.empty_cache()
                    # Print memory stats every 500 episodes
                    if verbose:
                        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                        max_allocated = torch.cuda.max_memory_allocated() / (
                            1024 * 1024
                        )
                        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                        print(
                            f"  GPU Memory: Current={allocated:.1f}MB, Peak={max_allocated:.1f}MB, Reserved={reserved:.1f}MB"
                        )

        except Exception as e:
            print(f"Training interrupted: {e}")
            # Save emergency checkpoint if training fails
            emergency_path = os.path.join(self.checkpoint_dir, "emergency_model.pth")
            try:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epsilon": self.epsilon,
                        "episode": episode if "episode" in locals() else 0,
                        "total_steps": self.total_steps,
                    },
                    emergency_path,
                )
                print(f"Emergency checkpoint saved: {emergency_path}")
            except:
                print("Failed to save emergency checkpoint")

            # Always try to clean up GPU memory when exiting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test(self, episodes=None, render=False):
        """Test the agent's performance over multiple episodes"""
        if episodes is None:
            episodes = self.episodes

        total_rewards = []
        success_count = 0

        for episode in range(episodes):
            obs, info = self.env.reset()
            self.num_llm_calls = 0
            self.current_hint = ""
            # Initialize previous room tracking for this episode
            self.prev_room = info["symbolic_state"]["agent"]["room"]

            total_reward = 0
            episode_success = False
            state = self.preprocess_state(obs, info)

            for step in range(self.env.max_steps):
                action, cost, state = self.choose_action(state, obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Track success
                if reward > 0:
                    self.current_hint = ""
                    episode_success = True

                reward -= cost
                total_reward += reward

                # Update previous room
                self.prev_room = info["symbolic_state"]["agent"]["room"]

                if render:
                    self.env.render()

                if terminated or truncated:
                    break

            # Record results
            total_rewards.append(total_reward)
            if episode_success:
                success_count += 1

            # Log progress periodically
            if episode % 1000 == 0 or episode == episodes - 1:
                success_rate = success_count / (episode + 1) * 100
                print(
                    f"Test Episode {episode+1}/{episodes}, "
                    f"Reward: {total_reward:.4f}, "
                    f"Success Rate: {success_rate:.1f}%"
                )

        # Calculate final metrics
        average_reward = sum(total_rewards) / max(1, len(total_rewards))
        final_success_rate = success_count / max(1, len(total_rewards)) * 100

        print(f"\nTest Results over {len(total_rewards)} episodes:")
        print(f"  Average Reward: {average_reward:.4f}")
        print(f"  Success Rate: {final_success_rate:.1f}%")

        return average_reward
