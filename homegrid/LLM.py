import gym
import numpy as np
import torch
from homegrid.BLIP import BLIP2Helper
from homegrid.GPT import GPT4Helper
from pprint import pprint
import os
import json
from PIL import Image
from datetime import datetime
from homegrid.utils import format_symbolic_state


class LLMAgent:
    def __init__(
        self, model_name, env_name="homegrid-task", episodes=500, checkpoint_dir=None
    ):
        assert checkpoint_dir is not None, "checkpoint_dir must be provided"
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize environment and hyperparameters
        self.env = gym.make(env_name, disable_env_checker=True)
        self.episodes = 0

        # Directory to save checkpoints
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create separate folders for training and testing outputs
        self.testing_dir = os.path.join(self.checkpoint_dir, "testing")
        os.makedirs(self.testing_dir, exist_ok=True)

        # Initialize state
        self.state = None

        if model_name == "blip-2":
            self.llm = BLIP2Helper()
        else:
            self.llm = GPT4Helper(model=model_name)

    def encode_state(self, obs, info):
        """Convert observations and info into format for LLM query"""
        observation = Image.fromarray(obs["image"])
        return observation, info

    def update_state(self, obs, info):
        """Update the state of the agent"""
        self.state = self.encode_state(obs, info)

    def reset(self):
        obs, info = self.env.reset()
        """Reset episode-specific variables and parse task objectives"""
        # Reset variables
        self.visited_rooms = {info["symbolic_state"]["agent"]["room"]}
        self.visited_cells = {tuple(info["symbolic_state"]["agent"]["pos"])}
        self.current_step = 0
        self.previous_potential = None
        self.update_state(obs, info)
        return obs, info

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
        agent_room = info["symbolic_state"]["agent"]["room"]
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

    def choose_action(self, obs, info, testing=True):
        """Get action from LLM using current state"""
        cost = 0.0

        action, confidence = self.llm.query_action(self.state, self.env.task)
        return action, cost

    def test(self, episodes, render=False):

        total_rewards = []  # Original rewards
        shaped_rewards = []  # Shaped rewards
        steps_to_success = []
        task_success_rates = {}

        for episode in range(episodes):
            # Reset episode-specific variables and parse objectives
            obs, info = self.reset()

            # Track current task
            current_task = self.env.task
            if current_task not in task_success_rates:
                task_success_rates[current_task] = {"attempts": 0, "successes": 0}
            task_success_rates[current_task]["attempts"] += 1

            total_reward = 0  # Original reward
            total_shaped_reward = 0  # Shaped reward

            for step in range(self.env.max_steps):
                self.current_step += 1
                action, cost = self.choose_action(obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.update_state(obs, info)

                # Track success without using shaped rewards
                if reward > 0:
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
                    "a"
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


if __name__ == "__main__":
    # Initialize the Simulator
    sim = LLMAgent(
        model_name="gpt-4o",
        env_name="homegrid-task",
        checkpoint_dir="checkpoints15",  # or any directory you want
    )

    # Test it
    sim.test(episodes=5, render=True)  # Set render=True if you want to see the env
