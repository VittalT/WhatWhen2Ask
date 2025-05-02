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
import os
import json
from PIL import Image
from datetime import datetime
import time
from sentence_transformers import SentenceTransformer

# Define embedding dimensions
TASK_EMBED_DIM = 384  # all-MiniLM-L6-v2 embedding dimension
HINT_EMBED_DIM = 385  # all-MiniLM-L6-v2 embedding dimension + 1 for flag


def visualize_image(image):
    plt.imshow(image)  # expects (H, W, 3) in RGB
    plt.axis("off")
    plt.show()


class DQN(nn.Module):
    def __init__(
        self,
        num_actions,
        task_embed_dim=384,
        hint_embed_dim=385,
    ):
        super().__init__()

        # === Agent State ===
        self.agent_state_proj = nn.Sequential(
            nn.Linear(2 + 4 + 5, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # === Context ===
        self.context_proj = nn.Sequential(
            nn.Linear(task_embed_dim + hint_embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # === Egocentric Map CNN ===
        # input: (batch, 11, V, V)
        self.ego_cnn = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),  # preserve V×V
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # downsample V→V/2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (batch, 64, 1, 1)
            nn.Flatten(),  # → (batch, 64)
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # === Global Map CNN ===
        # input: (batch, 12, H, W)
        self.global_cnn = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1, stride=2),  # H,W→H/2,W/2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # →H/4,W/4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (batch, 32,1,1)
            nn.Flatten(),  # → (batch, 32)
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # === Other streams ===
        self.object_map_proj = nn.Sequential(
            nn.Linear(10 * 3, 64),  # unchanged
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.visited_proj = nn.Sequential(
            nn.Linear(10 * 2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # dynamically pick up your real action-space size
        action_dim = num_actions
        hist_len = 5  # last 5 actions
        self.past_action_proj = nn.Sequential(
            nn.Linear(hist_len * (action_dim + 1), 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # === Final fusion ===
        total_dim = 32 + 128 + 64 + 64 + 32 + 16 + 16  # sum of each branch
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, num_actions)

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
        a = self.agent_state_proj(agent_state)
        c = self.context_proj(context)
        e = self.ego_cnn(ego_map)
        g = self.global_cnn(global_map)
        o = self.object_map_proj(object_map)
        v = self.visited_proj(visited)
        p = self.past_action_proj(past_actions)
        x = torch.cat([a, c, e, g, o, v, p], dim=1)
        h = self.final_fusion(x)
        return self.policy_head(h)


class DQNAgent:
    def __init__(self, env_name="homegrid-task", episodes=500, checkpoint_dir=None):
        assert checkpoint_dir is not None, "checkpoint_dir must be provided"
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the sentence transformer model
        # Initialize on CPU for better compatibility with transformers
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device=self.device
        )

        # Initialize environment and hyperparameters
        self.env = gym.make(env_name, disable_env_checker=True)
        self.alpha = 0.0005  # Lower learning rate for more stable learning
        self.gamma = 0.99
        self.epsilon = 1.0
        self.batch_size = 64  # Larger batch size for better gradient estimates
        self.episodes = episodes
        self.epsilon_decay = 0.99975  # Slower decay helps explore more thoroughly
        self.epsilon_min = 0.01  # Higher minimum exploration rate
        self.llm_cost = 0.01
        self.num_llm_calls = 0
        self.max_llm_calls = 0  # Disable LLM queries for pure DQN training
        self.current_hint = ""
        self.agent_view_size = 3

        # Get grid dimensions from environment
        self.width = 12  # bit less than env.width
        self.height = 10  # bit less than env.height

        # Track episode number from previous training
        self.previous_episode = 0

        # Target network update frequency (update every N steps)
        self.target_update_freq = 500
        self.total_steps = 0

        # Memory parameters
        self.replay_buffer = {}
        self.max_replay_buffer_size = 100_000  # Reduced for computational efficiency

        # Prioritized experience replay parameters
        self.use_per = True  # Can set to False if computationally expensive
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.001
        self.per_epsilon = 0.01

        # Number of actions from environment
        self.output_dim = self.env.action_space.n

        # Initialize models and move them to the appropriate device
        # Using TASK_EMBED_DIM and HINT_EMBED_DIM for the transformer model
        self.model = DQN(
            self.output_dim,
            task_embed_dim=TASK_EMBED_DIM,
            hint_embed_dim=HINT_EMBED_DIM,
        ).to(self.device)
        self.target_model = DQN(
            self.output_dim,
            task_embed_dim=TASK_EMBED_DIM,
            hint_embed_dim=HINT_EMBED_DIM,
        ).to(self.device)
        self.update_target_network()

        # Optimizer with GPU support
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss(reduction="none")  # For prioritized replay

        # Initialize LLMHelper
        self.llm_helper = None
        self.hint_threshold = 0.95

        # Checkpoint interval
        self.checkpoint_interval = 500
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
        self.visited_cells = set()
        self.recent_cells = deque(maxlen=10)  # Track last 10 visited cells
        self.recent_actions = deque(maxlen=5)  # Track last 5 actions
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

        self.recent_cells.append(pos)
        self.visited_cells.add(pos)

    # Add this method to track past actions
    def update_past_actions(self, action):
        """Track the last 5 actions"""
        if action is not None:
            self.recent_actions.append(action)

    def update_visited_rooms(self, room):
        """Track the last 5 visited rooms"""
        self.visited_rooms.add(room)

    def update_state(self, obs, info, action=None):
        """Update the state of the agent"""
        current_pos = info["symbolic_state"]["agent"]["pos"]
        agent_room = info["symbolic_state"]["agent"]["room"]
        self.update_visited_cells(current_pos)
        self.update_visited_rooms(agent_room)
        if action is not None:
            self.update_past_actions(action)
        self.state = self.encode_state(obs, info)

    def reset_episode(self, obs, info):
        """Reset episode-specific variables and parse task objectives"""
        # Reset visited locations and past actions
        self.visited_cells = set()
        self.visited_rooms = set()
        self.recent_cells = deque(maxlen=10)
        self.recent_actions = deque(maxlen=5)
        self.update_state(obs, info)

        # Reset other variables
        self.current_step = 0
        self.previous_potential = None

        # Parse objectives if needed
        self.objectives = self._find_matching_objects(info)

    def get_sentence_embedding(self, text):
        """Convert text to sentence embedding and move to the correct device."""
        # Handle empty text
        if not text or text == "none" or text == "":
            # Return zeros for empty text
            return torch.zeros(TASK_EMBED_DIM, dtype=torch.float32, device=self.device)

        # Get embedding and move to the appropriate device
        embedding = self.sentence_model.encode(text, convert_to_tensor=True)
        return embedding.to(self.device)

    def encode_str(self, text, is_hint=False):
        """
        Convert a string to a sentence embedding using all-MiniLM-L6-v2 and move to device.
        For hints, adds an additional flag dimension.
        """
        # Get the base embedding
        embedding = self.get_sentence_embedding(text)

        # For hints, add an extra dimension for the flag
        if is_hint:
            # Add a flag indicating if there's a hint (1.0) or not (0.0)
            flag = torch.tensor(
                [1.0 if text else 0.0], dtype=torch.float32, device=self.device
            )
            embedding = torch.cat([embedding, flag])

        # Add batch dimension and move to device
        return embedding.unsqueeze(0).to(self.device)

    def object_to_channel(self, obj):
        if isinstance(obj, str):  # Handle string inputs (for inventory objects)
            obj_name = obj
        else:
            obj_name = obj["name"]
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
        if "bin" in obj_name and obj["state"] == "closed":
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

        def normalize_position(x, y):
            """Normalize position coordinates from 1-width/height range to 0-1 range"""
            return (float(x) - 1) / (self.width - 1), (float(y) - 1) / (self.height - 1)

        sym_state = info["symbolic_state"]
        agent_info = sym_state["agent"]
        objects = sym_state["objects"]

        # === Agent State ===
        # Position: (x, y) - normalized by grid size (width, height)
        pos_x, pos_y = agent_info["pos"]
        pos_x_norm, pos_y_norm = normalize_position(pos_x, pos_y)

        # Direction: one-hot encode (0-3) and normalize by 4
        dir_int = agent_info["dir"]
        direction = torch.zeros(4, device=self.device)
        direction[dir_int] = 1.0

        # Inventory: 5 dimensions (bottle, fruit, papers, plates, none)
        inventory = torch.zeros(5, device=self.device)
        carrying = agent_info["carrying"]
        if carrying is None:
            inventory[4] = 1.0  # None/empty slot
        else:
            # Map object name to index (0-3)
            obj_idx = self.object_to_channel(carrying)
            assert obj_idx < 4, f"Invalid object index: {obj_idx}"
            inventory[obj_idx] = 1.0

        # Combine agent state
        agent_state = (
            torch.cat(
                [
                    torch.tensor([pos_x_norm, pos_y_norm], device=self.device),
                    direction,
                    inventory,
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # === Context: Task + Hint ===
        # We'll use the existing encode_str method for these
        task_embed = self.encode_str(self.env.task)
        hint_embed = self.encode_str(self.current_hint, is_hint=True)
        context = torch.cat([task_embed, hint_embed], dim=1).to(self.device)

        # === Egocentric Object Map ===
        # partial view around agent with 11 channels
        # (10 object types + 1 blocked)
        ego_map = torch.zeros(11, self.agent_view_size, self.agent_view_size).to(
            self.device
        )

        # Populate egocentric map by checking objects within view of agent
        for obj in objects:
            obj_pos = obj["pos"]
            obj_x, obj_y = obj_pos
            # Check if object is within the agent's view
            rel_x = obj_x - pos_x + self.agent_view_size // 2  # + to center
            rel_y = obj_y - pos_y + self.agent_view_size // 2  # + to center

            if 0 <= rel_x < self.agent_view_size and 0 <= rel_y < self.agent_view_size:
                # Map object to channel
                obj_channel = self.object_to_channel(obj)
                ego_map[obj_channel, rel_y, rel_x] = 1.0

                # Set blocked channel (last channel) - placeholder for walls/obstacles
                if obj["name"] == "wall" or obj["type"] in ["Storage", "Inanimate"]:
                    ego_map[10, rel_y, rel_x] = 1.0

        ego_map = ego_map.unsqueeze(0)

        # === Global Semantic Map ===
        # Full map with 12 channels
        # 11 object state pairs + 1 for seen cells
        global_map = torch.zeros(12, self.height, self.width).to(self.device)

        # Populate global map with objects from symbolic state
        for obj in objects:
            x, y = obj["pos"]
            channel = self.object_to_channel(obj)
            global_map[channel, y - 1, x - 1] = 1.0

            # If this is a wall or non-overlappable object, mark it in the blocked channel
            if obj["name"] == "wall" or obj["type"] in ["Storage", "Inanimate"]:
                global_map[10, y - 1, x - 1] = 1.0

        # Add "seen" cells to the last channel
        for visited_pos in self.visited_cells:
            x, y = visited_pos
            for d_x in range(-self.agent_view_size // 2, self.agent_view_size // 2 + 1):
                for d_y in range(
                    -self.agent_view_size // 2, self.agent_view_size // 2 + 1
                ):
                    nx, ny = x + d_x, y + d_y
                    if 0 < nx <= self.width and 0 < ny <= self.height:
                        global_map[11, ny - 1, nx - 1] = 1.0

        global_map = global_map.unsqueeze(0)

        # === Object Map: 10 objects with (x, y, present) ===
        object_map = torch.zeros(10 * 3).to(self.device)

        for obj in objects:
            channel = self.object_to_channel(obj)
            x, y = obj["pos"]
            # Normalize position
            x_norm, y_norm = normalize_position(x, y)

            # Store (x, y, present)
            idx = channel * 3
            object_map[idx : idx + 3] = torch.tensor(
                [x_norm, y_norm, 1.0], dtype=torch.float32, device=self.device
            )

        object_map = object_map.unsqueeze(0)

        # === Past Visited Locations ===
        recent_cells_tensor = -torch.ones(10 * 2).to(self.device)

        for i in range(len(self.recent_cells)):
            x, y = self.recent_cells[i]
            x_norm, y_norm = normalize_position(x, y)
            recent_cells_tensor[i * 2 : i * 2 + 2] = torch.tensor(
                [x_norm, y_norm], dtype=torch.float32, device=self.device
            )

        recent_cells_tensor = recent_cells_tensor.unsqueeze(0)

        # === Past Actions ===
        recent_actions_tensor = torch.zeros(5 * 11).to(self.device)
        recent_actions_tensor[10::11] = 1.0  # Start invalid

        for i in range(len(self.recent_actions)):
            recent_actions_tensor[i * 11 + self.recent_actions[i]] = 1.0
            recent_actions_tensor[i * 11 + 10] = 0.0

        recent_actions_tensor = recent_actions_tensor.unsqueeze(0)

        return (
            agent_state,
            context,
            ego_map,
            global_map,
            object_map,
            recent_cells_tensor,
            recent_actions_tensor,
        )

    def uncertainty_score(self, q_values=None):
        """
        Compute uncertainty based on the entropy of the softmax over Q-values.

        Args:
            q_values: Pre-computed Q-values. If None, will compute using the model.
        """
        # Get q_values if not provided
        if q_values is None:
            with torch.no_grad():
                q_values = self.model(*self.state)

        # Compute softmax and entropy on GPU for better performance
        with torch.no_grad():
            # Apply softmax with numerical stability
            q_values = q_values.squeeze(0)
            q_values = q_values - torch.max(q_values)
            probabilities = torch.softmax(q_values, dim=0)

            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
            max_entropy = torch.log(
                torch.tensor(float(len(q_values)), device=self.device)
            )

            # Normalize and explicitly move to CPU before computing result
            result = (entropy / max_entropy).cpu().item()

        return result

    def check_state_shapes(self, state):
        """
        Validates the shapes of each tensor in the state tuple.

        Args:
            state: Tuple of tensors returned by encode_state

        This function prints the actual shapes, expected shapes, and
        performs assertions to ensure dimensions are correct.

        Returns:
            bool: True if all assertions pass
        """
        agent_state, context, ego_map, global_map, object_map, visited, past_actions = (
            state
        )

        print("\n=== State Tensor Shapes ===")

        # Agent State: (batch_size, 11) - position(2), direction(4), inventory(5)
        print(f"Agent State: {tuple(agent_state.shape)} (Expected: [1, 11])")
        assert agent_state.shape == (
            1,
            11,
        ), f"Agent state shape mismatch: {agent_state.shape}"

        # Context: (batch_size, task_embed_dim + hint_embed_dim)
        expected_context_dim = 384 + 385  # task_embed_dim + hint_embed_dim
        print(
            f"Context: {tuple(context.shape)} (Expected: [1, {expected_context_dim}])"
        )
        assert (
            context.shape[0] == 1
        ), f"Context batch dimension mismatch: {context.shape[0]}"
        assert (
            context.shape[1] == expected_context_dim
        ), f"Context feature dimension mismatch: {context.shape[1]}"

        # Egocentric Map: (batch_size, 11, agent_view_size, agent_view_size)
        print(
            f"Egocentric Map: {tuple(ego_map.shape)} (Expected: [1, 11, {self.agent_view_size}, {self.agent_view_size}])"
        )
        assert ego_map.shape == (
            1,
            11,
            self.agent_view_size,
            self.agent_view_size,
        ), f"Ego map shape mismatch: {ego_map.shape}"

        # Global Map: (batch_size, 12, height, width)
        print(
            f"Global Map: {tuple(global_map.shape)} (Expected: [1, 12, {self.height}, {self.width}])"
        )
        assert global_map.shape == (
            1,
            12,
            self.height,
            self.width,
        ), f"Global map shape mismatch: {global_map.shape}"

        # Object Map: (batch_size, 10*3) - 10 objects with (x, y, present)
        print(f"Object Map: {tuple(object_map.shape)} (Expected: [1, 30])")
        assert object_map.shape == (
            1,
            30,
        ), f"Object map shape mismatch: {object_map.shape}"

        # Visited Cells: (batch_size, 10*2) - 10 recent cells with (x, y)
        print(f"Recent Visited Cells: {tuple(visited.shape)} (Expected: [1, 20])")
        assert visited.shape == (
            1,
            20,
        ), f"Visited cells shape mismatch: {visited.shape}"

        # Past Actions: (batch_size, 5*(num_actions+1)) - 5 recent actions with one-hot encoding + invalid flag
        action_dim = self.env.action_space.n
        expected_past_actions_dim = 5 * (action_dim + 1)
        print(
            f"Past Actions: {tuple(past_actions.shape)} (Expected: [1, {expected_past_actions_dim}])"
        )
        assert past_actions.shape == (
            1,
            expected_past_actions_dim,
        ), f"Past actions shape mismatch: {past_actions.shape}"

        print("\n=== Neural Network Layer Input Shapes ===")
        print(
            f"agent_state_proj expects: {tuple(agent_state.shape)} → {self.model.agent_state_proj[0].in_features} features"
        )
        assert (
            self.model.agent_state_proj[0].in_features == agent_state.shape[1]
        ), "Agent state feature mismatch"

        print(
            f"context_proj expects: {tuple(context.shape)} → {self.model.context_proj[0].in_features} features"
        )
        assert (
            self.model.context_proj[0].in_features == context.shape[1]
        ), "Context feature mismatch"

        print(
            f"ego_cnn expects: {tuple(ego_map.shape)} → {self.model.ego_cnn[0].in_channels} channels, any spatial size"
        )
        assert (
            self.model.ego_cnn[0].in_channels == ego_map.shape[1]
        ), "Ego map channel mismatch"

        print(
            f"global_cnn expects: {tuple(global_map.shape)} → {self.model.global_cnn[0].in_channels} channels, any spatial size"
        )
        assert (
            self.model.global_cnn[0].in_channels == global_map.shape[1]
        ), "Global map channel mismatch"

        print(
            f"object_map_proj expects: {tuple(object_map.shape)} → {self.model.object_map_proj[0].in_features} features"
        )
        assert (
            self.model.object_map_proj[0].in_features == object_map.shape[1]
        ), "Object map feature mismatch"

        print(
            f"visited_proj expects: {tuple(visited.shape)} → {self.model.visited_proj[0].in_features} features"
        )
        assert (
            self.model.visited_proj[0].in_features == visited.shape[1]
        ), "Visited cells feature mismatch"

        print(
            f"past_action_proj expects: {tuple(past_actions.shape)} → {self.model.past_action_proj[0].in_features} features"
        )
        assert (
            self.model.past_action_proj[0].in_features == past_actions.shape[1]
        ), "Past actions feature mismatch"

        print("\nAll shape checks passed ✓")
        return True

    def choose_action(self, obs, info, testing=False):
        """
        Choose an action using epsilon-greedy policy.
        """
        cost = 0

        # Epsilon-greedy action selection - check first to avoid unnecessary computation
        if not testing and random.random() < self.epsilon:
            return self.env.action_space.sample(), cost

        # Get Q-values from model (used for both uncertainty calculation and action selection)
        with torch.no_grad():
            q_values = self.model(*self.state)

        # Calculate uncertainty using pre-computed Q-values
        uncertainty = self.uncertainty_score(q_values)

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
                self.update_state(obs, info)

                # Get new Q-values with the updated hint
                with torch.no_grad():
                    q_values = self.model(*self.state)

        # Greedy action selection using pre-computed q_values
        action = torch.argmax(q_values, dim=1).item()
        return action, cost

    def _find_matching_objects(self, info):
        """
        Find all objects in the environment mentioned in the text description
        and sort them by their position in the text
        """
        matches = []
        text = self.env.task

        # Check all objects and rooms
        for obj in info["symbolic_state"]["objects"] + self.env.rooms:
            obj_name = obj["name"]
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
        agent_dir = info["symbolic_state"]["agent"]["dir"]
        carrying = info["symbolic_state"]["agent"]["carrying"]

        # Start with base potential
        potential = 0

        # Determine the current objective index
        current_objective_idx = 0

        # Check if carrying first objective (in a multi-objective task)
        carrying_first_obj = (
            carrying is not None
            and carrying == self.objectives[0]["name"]
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

            # Compute sum of distances on GPU
            pot_dist = 0.0
            for p, q in zip(positions[:-1], positions[1:]):
                v_p = torch.tensor(p, dtype=torch.float32, device=self.device)
                v_q = torch.tensor(q, dtype=torch.float32, device=self.device)
                pot_dist += torch.norm(v_p - v_q)
            pot_dist = pot_dist.item()

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
        weighted_orientation = 0.2 * pot_orientation
        weighted_carrying = 1.0 * pot_carrying
        weighted_expl = 0.01 * pot_expl
        weighted_time = -0.01 * pot_time  # penalize as time goes

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
            # Keep priorities as a single GPU tensor
            priorities = torch.tensor(
                self.replay_buffer[task_id]["priorities"], device=self.device
            )

            if priorities.sum() <= 0:
                # Uniform fallback
                indices = torch.randperm(buffer_size, device=self.device)[
                    :actual_batch_size
                ]
                weights = torch.ones(actual_batch_size, device=self.device)
            else:
                probs = priorities / priorities.sum()
                indices = torch.multinomial(probs, actual_batch_size, replacement=False)
                # importance-sampling weights
                self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
                weights = (buffer_size * probs[indices]) ** (-self.per_beta)
                weights /= weights.max()  # normalize

            # Bring indices back to CPU list
            indices = indices.cpu().tolist()

            # Get selected experiences
            batch = [self.replay_buffer[task_id]["experiences"][i] for i in indices]
            return batch, indices, weights
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

        # Use detach() to avoid gradient computation but keep tensors on GPU until necessary
        td_errors_abs = td_errors.abs().detach()

        # Update priorities directly with torch tensors
        for i, idx in enumerate(indices):
            priority = (td_errors_abs[i].item() + self.per_epsilon) ** self.per_alpha
            self.replay_buffer[task_id]["priorities"][idx] = priority

    def prepare_state_batch(self, state_batch, component_idx):
        """Helper function to move state components to GPU and batch them"""
        return torch.cat(
            [s[component_idx].to(self.device) for s in state_batch],
            dim=0,
        )

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

        # Prepare batched tensors for current and next states
        # Each state has 7 components (agent_state, context, ego_map, global_map, object_map, recent_cells, recent_actions)
        state_components = [self.prepare_state_batch(states, i) for i in range(7)]
        next_state_components = [
            self.prepare_state_batch(next_states, i) for i in range(7)
        ]

        # Prepare other tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(
            1
        )
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        # Convert importance sampling weights to tensor if using PER
        if weights is not None:
            weights = (
                weights.clone().detach().to(dtype=torch.float32, device=self.device)
            )

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(*next_state_components)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values

        # Compute current Q-values
        current_q_values = self.model(*state_components)
        current_q_values = current_q_values.gather(1, actions).squeeze(1)

        # Calculate TD errors (used for updating priorities)
        td_errors = target_q_values - current_q_values

        # Apply weights to the loss if using PER
        if weights is not None:
            # Element-wise loss
            losses = self.loss_fn(current_q_values, target_q_values)
            # Ensure weights match the batch dimension for proper broadcasting
            weights = weights.view(-1, 1)
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
        self.update_priorities(task_id, indices, td_errors)

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
                action, cost = self.choose_action(obs, info)
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
                # detach and move states to CPU before storing in replay buffer
                cpu_prev_state = [s.detach().cpu() for s in prev_state]
                cpu_next_state = [s.detach().cpu() for s in self.state]
                transition = [
                    cpu_prev_state,
                    action,
                    shaped_reward,
                    cpu_next_state,
                    terminated,
                ]
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
                action, cost = self.choose_action(obs, info, testing=True)
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
                # Calculate success rate properly using task-specific attempts
                success_rate = (
                    task_success_rates[current_task]["successes"]
                    / task_success_rates[current_task]["attempts"]
                ) * 100
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


if __name__ == "__main__":
    # Initialize the Simulator
    agent = DQNAgent(
        env_name="homegrid-task",
        episodes=5,  # Just 5 test episodes to start
        checkpoint_dir="checkpoints15",  # or any directory you want
    )

    # Reset environment and agent state
    obs, info = agent.env.reset()
    agent.reset_episode(obs, info)

    # Run shape checks to validate tensor dimensions
    try:
        agent.check_state_shapes(agent.state)

        # Take a step and check shapes again to ensure consistency
        action = agent.env.action_space.sample()
        obs, _, _, _, info = agent.env.step(action)
        agent.update_state(obs, info, action)
        agent.check_state_shapes(agent.state)

        print("\nSuccess! All tensor shapes match network expectations.")
        print("You can now use this agent for training or testing.")
    except AssertionError as e:
        print(f"\nERROR: Shape validation failed: {e}")
        print("Fix the tensor shapes before training the model.")

    # Check environment properties (optional)
    print("\nEnvironment properties:")
    print(f"Action space: {agent.env.action_space}")
    print(f"Observation space: {agent.env.observation_space}")
    print(f"Current task: {agent.env.task}")
