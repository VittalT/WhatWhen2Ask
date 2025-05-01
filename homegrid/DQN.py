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
    def __init__(self, num_actions, task_embed_dim=384, hint_embed_dim=385):
        super().__init__()

        # === Agent State Processing ===
        # Position: (x, y) normalized by (12, 10)
        # Direction: One-hot encoded (4 dims) normalized by 4
        # Inventory: 5 dims (one hot + none)
        self.agent_state_proj = nn.Sequential(
            nn.Linear(2 + 4 + 5, 64),  # 2 (pos) + 4 (dir) + 5 (inv)
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # === Context Processing ===
        # Task: all-MiniLM-L6-v2 Sentence transformer (384)
        # LLM hint: all-MiniLM-L6-v2 Sentence transformer + 1 for flag (385)
        self.context_proj = nn.Sequential(
            nn.Linear(task_embed_dim + hint_embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # === Observation Processing ===
        # Egocentric object map: (3, 3, 11) - 3x3 view, 10 object state pairs + 1 (blocked)
        self.ego_map_cnn = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # === Memory Processing ===
        # Location-based semantic map: (W, H, 12)
        # 11 object state pairs: 4 objects + 3 bins * 2 states + 1 blocked + 1 seen
        self.global_map_cnn = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Object map: (10 x 3) - 10 object state pairs with (x, y, present)
        self.object_map_proj = nn.Sequential(
            nn.Linear(10 * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Past visited: (10 x 2) - Last 10 visited cell locations
        self.visited_proj = nn.Sequential(
            nn.Linear(10 * 2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Past actions: (5 x 11) - Last 5 actions + 1 if invalid
        self.past_action_proj = nn.Sequential(
            nn.Linear(5 * 11, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # === Final Fusion Network ===
        # Combine all features with attention
        total_input_dim = 32 + 128 + 64 + 64 + 32 + 16 + 16  # sum of all branches
        self.attention = nn.MultiheadAttention(embed_dim=total_input_dim, num_heads=8)

        self.final_fusion = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Policy head for action selection
        self.policy_head = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.LayerNorm(num_actions),
        )

    def forward(
        self,
        agent_state,
        context,
        ego_map,
        global_map,
        object_map,
        visited,
        past_actions,
    ):
        """
        Forward pass through the DQN network.

        Args:
            agent_state: Tensor of shape (batch_size, 11) - position, direction, inventory
            context: Tensor of shape (batch_size, 769) - task + hint embeddings
            ego_map: Tensor of shape (batch_size, 11, 3, 3) - egocentric view
            global_map: Tensor of shape (batch_size, 12, H, W) - full semantic map
            object_map: Tensor of shape (batch_size, 30) - flattened object states
            visited: Tensor of shape (batch_size, 20) - past visited locations
            past_actions: Tensor of shape (batch_size, 55) - action history

        Returns:
            action_logits: Tensor of shape (batch_size, num_actions)
        """
        # Process each input stream
        agent_feats = self.agent_state_proj(agent_state)
        context_feats = self.context_proj(context)
        ego_feats = self.ego_map_cnn(ego_map)
        global_feats = self.global_map_cnn(global_map)
        object_feats = self.object_map_proj(object_map)
        visited_feats = self.visited_proj(visited)
        action_feats = self.past_action_proj(past_actions)

        # Concatenate all features
        combined_feats = torch.cat(
            [
                agent_feats,
                context_feats,
                ego_feats,
                global_feats,
                object_feats,
                visited_feats,
                action_feats,
            ],
            dim=-1,
        )

        # Apply self-attention
        attended_feats, _ = self.attention(
            combined_feats.unsqueeze(0),
            combined_feats.unsqueeze(0),
            combined_feats.unsqueeze(0),
        )
        attended_feats = attended_feats.squeeze(0)

        # Final fusion and policy
        fused_feats = self.final_fusion(attended_feats)
        action_logits = self.policy_head(fused_feats)

        return action_logits


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
        self.batch_size = 32  # Larger batch size for better gradient estimates
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
        self.max_replay_buffer_size = 250  # Reduced for computational efficiency

        # Prioritized experience replay parameters
        self.use_per = True  # Can set to False if computationally expensive
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.001
        self.per_epsilon = 0.01
        self.agent_view_size = 3

        # Number of actions from environment
        self.output_dim = self.env.action_space.n

        # Initialize models and move them to the appropriate device
        # Using task_embed_dim=384 for MiniLM and hint_embed_dim=385 for MiniLM+flag
        self.model = DQN(self.output_dim, task_embed_dim=384, hint_embed_dim=385).to(
            self.device
        )
        self.target_model = DQN(
            self.output_dim, task_embed_dim=384, hint_embed_dim=385
        ).to(self.device)
        self.update_target_network()

        # Optimizer with GPU support
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss(reduction="none")  # For prioritized replay

        # Initialize LLMHelper
        self.llm_helper = None
        self.hint_threshold = 0.95

        # Checkpoint interval
        self.checkpoint_interval = 250
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
            "action_space": self.output_dim,
            "device": str(self.device),
        }

        # Initialize state tracking
        self.all_visited_cells = set()
        self.visited_cells = deque(maxlen=10)  # Track last 10 visited cells
        self.past_actions = deque(maxlen=5)  # Track last 5 actions
        self.visited_rooms = set()

        # For potential-based reward shaping
        self.previous_potential = None
        self.objectives = []
        self.potential_components = {}

    # Add this method to track visited cells
    def update_visited_cells(self, pos):
        """Track the last 10 visited cell positions"""
        # Convert numpy values to Python native types if needed
        if hasattr(pos[0], "item"):
            pos = (pos[0].item(), pos[1].item())

        self.visited_cells.append(pos)
        self.all_visited_cells.add(pos)

    # Add this method to track past actions
    def update_past_actions(self, action):
        """Track the last 5 actions"""
        if action is not None:
            self.past_actions.append(action)

    def update_visited_rooms(self, room):
        """Track the last 5 visited rooms"""
        self.visited_rooms.add(room)

    def update_state(self, obs, info, action=None):
        """Update the state of the agent"""
        self.state = self.encode_state(obs, info)
        current_pos = info["symbolic_state"]["agent"]["pos"]
        agent_room = info["symbolic_state"]["agent"]["room"].lower()
        self.update_visited_cells(current_pos)
        self.update_visited_rooms(agent_room)
        if action is not None:
            self.update_past_actions(action)

    def reset_episode(self, obs, info):
        """Reset episode-specific variables and parse task objectives"""
        # Reset visited locations and past actions
        self.all_visited_cells = set()
        self.visited_cells = deque(maxlen=10)
        self.past_actions = deque(maxlen=5)
        self.update_state(obs, info)

        # Reset other variables
        self.visited_rooms = {info["symbolic_state"]["agent"]["room"].lower()}
        self.current_step = 0
        self.previous_potential = None

        # Parse objectives if needed
        self.objectives = self._find_matching_objects(info)

    def encode_str(self, text, encode_context=True):
        """
        Convert a string to a 300-dimensional FastText embedding and move to device.
        """
        if encode_context:
            # Get embedding and move to GPU if available
            return get_fasttext_embedding(text).unsqueeze(0).to(self.device)
        else:
            return text if text != "" else "none"

    def object_to_channel(self, obj):
        obj_name = obj["name"].lower()
        channel_map = {
            "bottle": 0,
            "fruit": 1,
            "papers": 2,
            "plates": 3,
            "trash bin": 4,
            "recycling bin": 6,
            "compost bin": 8,
        }

        channel = channel_map[obj_name]
        # Add state offset for bins (open/closed)
        if "bin" in obj_name and obj["state"].lower() == "closed":
            channel += 1  # Use next channel for closed state
        return channel

    def encode_state(self, obs, info):
        """
        Converts the symbolic state from environment info into tensors for the DQN model.

        Args:
            info: Dict containing symbolic state information from environment

        Returns:
            Tuple of tensors needed for DQN forward pass
        """

        # Use the image observation for visual features if needed
        image = (
            torch.FloatTensor(obs["image"] / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        sym_state = info["symbolic_state"]
        agent_info = sym_state["agent"]
        objects = sym_state["objects"]

        # === Agent State ===
        # Position: (x, y) - normalized by grid size (12, 10)
        pos_x, pos_y = agent_info["pos"]
        pos_x_norm = float(pos_x) / 12.0  # Normalize x by grid width
        pos_y_norm = float(pos_y) / 10.0  # Normalize y by grid height

        # Direction: one-hot encode (0-3) and normalize by 4
        dir_int = agent_info["dir"]
        direction = torch.zeros(4)
        direction[dir_int] = 1.0

        # Inventory: 5 dimensions (bottle, fruit, papers, plates, none)
        inventory = torch.zeros(5)
        carrying = agent_info["carrying"]
        if carrying is None:
            inventory[4] = 1.0  # None/empty slot
        else:
            # Map object name to index (0-3)
            obj_map = {"bottle": 0, "fruit": 1, "papers": 2, "plates": 3}
            obj_idx = obj_map.get(carrying, 4)  # Default to 4 if unknown
            inventory[obj_idx] = 1.0

        # Combine agent state
        agent_state = (
            torch.cat([torch.tensor([pos_x_norm, pos_y_norm]), direction, inventory])
            .unsqueeze(0)
            .to(self.device)
        )

        # === Context: Task + Hint ===
        # We'll use the existing encode_str method for these
        task_embed = self.encode_str(self.env.task)
        hint_embed = self.encode_str(self.current_hint)
        context = torch.cat([task_embed, hint_embed], dim=1).to(self.device)

        # === Egocentric Object Map ===
        # 3x3 view around agent with 11 channels
        # (10 object types + 1 blocked)
        ego_map = torch.zeros(11, self.agent_view_size, self.agent_view_size).to(
            self.device
        )

        # Populate egocentric map by checking objects within 1 cell of agent
        # Simple implementation for now, can be enhanced with actual observation data
        for obj in objects:
            obj_pos = obj["pos"]
            obj_x, obj_y = obj_pos
            # Check if object is within the 3x3 grid centered on agent
            rel_x = obj_x - pos_x + self.agent_view_size // 2  # +1 to center
            rel_y = obj_y - pos_y + self.agent_view_size // 2  # +1 to center

            if 0 <= rel_x < self.agent_view_size and 0 <= rel_y < self.agent_view_size:
                # Map object to channel
                obj_channel = self.object_to_channel(obj)
                ego_map[obj_channel, rel_y, rel_x] = 1.0

        # Set blocked channel (last channel) - placeholder for walls/obstacles
        # In a real implementation, this would be extracted from actual observation data
        ego_map[10, :, :] = 0.0

        # === Global Semantic Map ===
        # Full map with 12 channels
        # 11 object state pairs + 1 for seen cells
        W, H = 12, 10  # Grid dimensions
        global_map = torch.zeros(12, H, W).to(self.device)

        # Populate global map with objects from symbolic state
        for obj in objects:
            x, y = obj["pos"]
            channel = self.object_to_channel(obj)
            global_map[channel, y - 1, x - 1] = 1.0

        # Add "seen" cells to the last channel
        for visited_pos in self.all_visited_cells:
            x, y = visited_pos
            for d_x in range(-self.agent_view_size // 2, self.agent_view_size // 2 + 1):
                for d_y in range(
                    -self.agent_view_size // 2, self.agent_view_size // 2 + 1
                ):
                    nx, ny = x + d_x, y + d_y
                    if 0 < nx <= W and 0 < ny <= H:
                        global_map[11, ny - 1, nx - 1] = 1.0

        # === Object Map: 10 objects with (x, y, present) ===
        object_map = torch.zeros(10 * 3).to(self.device)
        object_map[2::3] = 1.0  # Start not present

        for obj in objects:
            channel = self.object_to_channel(obj)
            x, y = obj["pos"]
            # Normalize position
            x_norm = float(x) / W
            y_norm = float(y) / H

            # Store (x, y, present)
            idx = channel * 3
            object_map[idx : idx + 3] = x_norm, y_norm, 0.0

        object_map = object_map.unsqueeze(0)

        # === Past Visited Locations ===
        visited = -torch.ones(10 * 2).to(self.device)

        for i in range(len(self.visited_cells)):
            x, y = self.visited_cells[i]
            x_norm = float(x) / W
            y_norm = float(y) / H
            visited[i * 2 : i * 2 + 2] = x_norm, y_norm

        visited = visited.unsqueeze(0)

        # === Past Actions ===
        past_actions = torch.zeros(5 * 11).to(self.device)
        past_actions[10::11] = 1.0  # Start invalid

        for i in range(len(self.past_actions)):
            past_actions[i * 11 + self.past_actions[i]] = 1.0
            past_actions[i * 11 + 10] = 0.0

        past_actions = past_actions.unsqueeze(0)

        # Make sure shapes are correct for batched input (batch_size=1)
        ego_map = ego_map.unsqueeze(0)
        global_map = global_map.unsqueeze(0)

        return (
            image,
            agent_state,
            context,
            ego_map,
            global_map,
            object_map,
            visited,
            past_actions,
        )

    def choose_action(self, state, obs, info, testing=False):
        """
        Choose an action using epsilon-greedy policy.
        Modified to use the new state format.
        """
        (
            image,
            agent_state,
            context,
            ego_map,
            global_map,
            object_map,
            visited,
            past_actions,
        ) = state

        uncertainty = self.uncertainty_score(state)
        cost = 0

        # Check if we should use LLM hints
        if (
            uncertainty > self.hint_threshold
            and self.num_llm_calls < self.max_llm_calls
            and not testing
        ):
            cost = self.llm_cost * (2**self.num_llm_calls)
            # Generate hint using LLM
            if self.llm_helper is not None:
                hint, _ = self.llm_helper.query_llm(obs["image"], info)
                self.num_llm_calls += 1
                print(f"Task: {self.env.task}\nHint: {hint}")
                self.current_hint = hint
                # Reprocess state with new hint
                state = self.preprocess_state(obs, info)
                (
                    image,
                    agent_state,
                    context,
                    ego_map,
                    global_map,
                    object_map,
                    visited,
                    past_actions,
                ) = state

        # Epsilon-greedy action selection
        if not testing and random.random() < self.epsilon:
            return self.env.action_space.sample(), cost, state

        # Greedy action selection
        with torch.no_grad():
            q_values = self.model(
                agent_state,
                context,
                ego_map,
                global_map,
                object_map,
                visited,
                past_actions,
            )
            action = torch.argmax(q_values, dim=1).item()

        return action, cost, state

    def uncertainty_score(self, state):
        """
        Compute uncertainty based on the entropy of the softmax over Q-values.
        Modified to use the new state format.
        """
        (
            image,
            agent_state,
            context,
            ego_map,
            global_map,
            object_map,
            visited,
            past_actions,
        ) = state

        with torch.no_grad():
            q_values = self.model(
                agent_state,
                context,
                ego_map,
                global_map,
                object_map,
                visited,
                past_actions,
            )
            q_values = q_values.squeeze(0).cpu().numpy()

        # Compute entropy (on CPU with numpy)
        q_values = q_values - np.max(q_values)  # for numerical stability
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
        max_entropy = np.log(len(q_values))

        return entropy / max_entropy

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

    def shaped_reward(self, base_reward, info, gamma=1):
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

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Extract and concatenate all state components
        # Each state is (image, agent_state, context, ego_map, global_map, object_map, visited, past_actions)

        # Prepare batched tensors for current states
        image_batch = torch.cat([s[0] for s in states], dim=0)
        agent_state_batch = torch.cat([s[1] for s in states], dim=0)
        context_batch = torch.cat([s[2] for s in states], dim=0)
        ego_map_batch = torch.cat([s[3] for s in states], dim=0)
        global_map_batch = torch.cat([s[4] for s in states], dim=0)
        object_map_batch = torch.cat([s[5] for s in states], dim=0)
        visited_batch = torch.cat([s[6] for s in states], dim=0)
        past_actions_batch = torch.cat([s[7] for s in states], dim=0)

        # Prepare batched tensors for next states
        next_image_batch = torch.cat([s[0] for s in next_states], dim=0)
        next_agent_state_batch = torch.cat([s[1] for s in next_states], dim=0)
        next_context_batch = torch.cat([s[2] for s in next_states], dim=0)
        next_ego_map_batch = torch.cat([s[3] for s in next_states], dim=0)
        next_global_map_batch = torch.cat([s[4] for s in next_states], dim=0)
        next_object_map_batch = torch.cat([s[5] for s in next_states], dim=0)
        next_visited_batch = torch.cat([s[6] for s in next_states], dim=0)
        next_past_actions_batch = torch.cat([s[7] for s in next_states], dim=0)

        # Prepare other tensors
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Convert importance sampling weights to tensor if using PER
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(
                next_agent_state_batch,
                next_context_batch,
                next_ego_map_batch,
                next_global_map_batch,
                next_object_map_batch,
                next_visited_batch,
                next_past_actions_batch,
            )
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values

        # Compute current Q-values
        current_q_values = self.model(
            agent_state_batch,
            context_batch,
            ego_map_batch,
            global_map_batch,
            object_map_batch,
            visited_batch,
            past_actions_batch,
        )
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

        # Set the model to training mode
        self.model.train()

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
            self.reset_episode(obs, info)

            self.num_llm_calls = 0
            self.current_hint = ""

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
                prev_state = self.state
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.update_state(obs, info, action)

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

                # Create transition and add to replay buffer with maximum priority initially
                transition = [prev_state, action, shaped_reward, self.state, terminated]
                self.add_experience(task_id, transition)

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

        # Set the model to evaluation mode
        self.model.eval()

        total_rewards = []  # Original rewards
        shaped_rewards = []  # Shaped rewards
        steps_to_success = []
        task_success_rates = {}

        for episode in range(episodes):
            obs, info = self.env.reset()

            # Reset episode-specific variables and parse objectives
            self.reset_episode(obs, info)

            self.num_llm_calls = 0
            self.current_hint = ""

            # Track current task
            current_task = self.env.task
            if current_task not in task_success_rates:
                task_success_rates[current_task] = {"attempts": 0, "successes": 0}
            task_success_rates[current_task]["attempts"] += 1

            total_reward = 0  # Original reward
            total_shaped_reward = 0  # Shaped reward

            for step in range(self.env.max_steps):
                self.current_step += 1
                action, cost, state = self.choose_action(state, obs, info, testing=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.update_state(obs, info, action)

                self.total_steps += 1

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
