#!/usr/bin/env python3

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from homegrid.window import Window
from homegrid.DQN import DQNAgent
from tokenizers import Tokenizer
from homegrid.LLM import LLMAgent
import argparse
import os

tok = Tokenizer.from_pretrained("t5-small")


class AgentSimulator:
    def __init__(
        self,
        env_name="homegrid-task",
        model_path="best",
        rate=0.1,
        render_agent_view=False,
        checkpoint_number=13,
    ):
        """
        Simulates a trained agent in the homegrid environment with visualization.

        Args:
            env_name: Name of the environment
            model_path: Path to the model or identifier ("best", "final", or episode number)
            rate: Time delay between steps (seconds)
            render_agent_view: Whether to render from agent's perspective
            checkpoint_number: Checkpoint folder number (e.g., 13 for checkpoints13)
        """
        self.env = gym.make(env_name, disable_env_checker=True)

        # Store checkpoint number for model loading
        self.checkpoint_number = checkpoint_number

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Limit GPU memory usage to approximately 45% of allocation
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.45)

        self.agent = self.load_agent(model_path)
        self.rate = rate
        self.render_agent_view = render_agent_view
        self.window = None
        self.fig = None
        self.ax_env = None
        self.ax_reward = None
        self.ax_info = None

        # Setup plot data
        self.steps = []
        self.shaped_rewards = []
        self.actual_rewards = []
        self.cumulative_shaped = []
        self.cumulative_actual = []
        self.running = True

        # Setup matplotlib
        for k in plt.rcParams:
            if "keymap" in k:
                plt.rcParams[k] = []

    def load_agent(self, model_path):
        """
        Load the DQN agent from a checkpoint.

        Args:
            model_path: Can be "best", "final", an integer episode number, or a file path
        """
        # Create directory paths
        checkpoint_dir = f"checkpoints{self.checkpoint_number}"
        training_dir = os.path.join(checkpoint_dir, "training")

        # Initialize agent
        if any(x in str(model_path) for x in ["blip", "gpt", "o1", "o2", "o3", "o4"]):
            agent = LLMAgent(
                model_name=model_path,
                env_name=self.env.unwrapped.spec.id,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            # Initialize DQN agent first
            agent = DQNAgent(
                env_name=self.env.unwrapped.spec.id,
                episodes=0,
                checkpoint_dir=checkpoint_dir,
            )

            # Handle different checkpoint formats
            checkpoint_path = None

            if isinstance(model_path, int) or model_path.isdigit():
                # Convert string to int if it's a digit string
                episode_num = (
                    int(model_path) if isinstance(model_path, str) else model_path
                )
                checkpoint_path = os.path.join(
                    training_dir, f"model_checkpoint_{episode_num}.pth"
                )
            elif model_path == "best":
                checkpoint_path = os.path.join(training_dir, "best_model.pth")
            elif model_path == "final":
                # Look for the most recent final model
                final_models = []
                final_models.extend(
                    [
                        os.path.join(training_dir, f)
                        for f in os.listdir(training_dir)
                        if f.startswith("final_model_") and f.endswith(".pth")
                    ]
                )

                if final_models:
                    # Sort by modification time, newest first
                    checkpoint_path = sorted(
                        final_models, key=os.path.getmtime, reverse=True
                    )[0]
            else:
                checkpoint_path = model_path

            print(f"Loading model from: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            agent.model.load_state_dict(checkpoint["model_state_dict"])
            if "epsilon" in checkpoint:
                agent.epsilon = checkpoint["epsilon"]
            if "total_steps" in checkpoint:
                agent.total_steps = checkpoint["total_steps"]
            if "episode" in checkpoint:
                agent.previous_episode = checkpoint["episode"]
            print(f"Loaded checkpoint with epsilon {agent.epsilon:.4f}")

            # Set to evaluation mode
            agent.model.eval()

        agent.env = self.env  # Connect agent to environment
        return agent

    def setup_visualization(self):
        """Set up the matplotlib visualization."""
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, height_ratios=[3, 1])

        # Environment view
        self.ax_env = self.fig.add_subplot(gs[0, 0])
        self.ax_env.set_title("Environment")
        self.ax_env.axis("off")

        # Information panel
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.set_title("Agent Information")
        self.ax_info.axis("off")

        # Reward plot
        self.ax_reward = self.fig.add_subplot(gs[1, :])
        self.ax_reward.set_title("Rewards")
        self.ax_reward.set_xlabel("Steps")
        self.ax_reward.set_ylabel("Reward")
        self.ax_reward.grid(True)

        # Plot legends and initial empty plots
        (self.shaped_line,) = self.ax_reward.plot([], [], "b-", label="Shaped Reward")
        (self.actual_line,) = self.ax_reward.plot([], [], "r-", label="Actual Reward")
        (self.cum_shaped_line,) = self.ax_reward.plot(
            [], [], "b--", label="Cumulative Shaped", alpha=0.7
        )
        (self.cum_actual_line,) = self.ax_reward.plot(
            [], [], "r--", label="Cumulative Actual", alpha=0.7
        )

        self.ax_reward.legend()
        plt.tight_layout()

        # Enable interactive mode
        plt.ion()
        self.fig.canvas.mpl_connect("key_press_event", self.key_handler)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def on_close(self, event):
        """Handle window close event."""
        plt.close("all")
        self.running = False

    def update_info_panel(self, obs, info, action, shaped_reward, actual_reward):
        """Update the information panel with current state info."""
        self.ax_info.clear()
        self.ax_info.axis("off")

        # Information to display
        task = self.env.task
        token = tok.decode([obs["token"]])
        step_count = self.env.step_cnt
        carrying = info["symbolic_state"]["agent"]["carrying"] or "nothing"
        front_obj = info["symbolic_state"]["front_obj"] or "nothing"
        room = info["symbolic_state"]["agent"]["room"]
        direction = ["Right", "Down", "Left", "Up"][
            info["symbolic_state"]["agent"]["dir"]
        ]

        # Action name mapping
        action_names = {
            0: "left",
            1: "right",
            2: "up",
            3: "down",
            4: "pickup",
            5: "drop",
            6: "get",
            7: "pedal",
            8: "grasp",
            9: "lift",
        }
        action_str = action_names.get(action, str(action))

        # Format text for info panel
        info_text = (
            f"Task: {task}\n\n"
            f"Step: {step_count}\n"
            f"Room: {room}\n"
            f"Direction: {direction}\n"
            f"Action: {action_str}\n\n"
            f"Carrying: {carrying}\n"
            f"In front: {front_obj}\n\n"
            f"Token: {token}\n\n"
            f"Shaped Reward: {shaped_reward:.4f}\n"
            f"Actual Reward: {actual_reward:.4f}\n"
        )

        # Add text to the panel
        self.ax_info.text(
            0.05,
            0.95,
            info_text,
            verticalalignment="top",
            horizontalalignment="left",
            transform=self.ax_info.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    def update_reward_plot(self):
        """Update the reward plot with current data."""
        # Update line data
        self.shaped_line.set_data(self.steps, self.shaped_rewards)
        self.actual_line.set_data(self.steps, self.actual_rewards)
        self.cum_shaped_line.set_data(self.steps, self.cumulative_shaped)
        self.cum_actual_line.set_data(self.steps, self.cumulative_actual)

        # Adjust limits
        if len(self.steps) > 0:
            self.ax_reward.set_xlim(0, max(10, max(self.steps)))

            all_rewards = self.shaped_rewards + self.actual_rewards
            if all_rewards:
                min_reward = min(min(all_rewards), 0) - 0.1
                max_reward = max(max(all_rewards), 0) + 0.1

                cum_min = (
                    min(min(self.cumulative_shaped + self.cumulative_actual), 0) - 0.1
                )
                cum_max = (
                    max(max(self.cumulative_shaped + self.cumulative_actual), 0) + 0.1
                )

                self.ax_reward.set_ylim(
                    min(min_reward, cum_min), max(max_reward, cum_max)
                )

    def update_env_display(self, img):
        """Update the environment display."""
        self.ax_env.clear()
        self.ax_env.imshow(img)
        self.ax_env.axis("off")

    def reset(self):
        """Reset the environment and visualization data."""
        obs, info = self.env.reset()
        self.steps = []
        self.shaped_rewards = []
        self.actual_rewards = []
        self.cumulative_shaped = []
        self.cumulative_actual = []

        # Reset agent tracking
        self.agent.current_hint = ""
        self.agent.reset_episode(obs, info)

        # Get image
        img = obs["image"] if self.render_agent_view else self.env.get_frame()
        self.update_env_display(img)

        # Update info panel with initial state
        self.update_info_panel(obs, info, None, 0, 0)
        self.update_reward_plot()
        plt.draw()
        plt.pause(0.001)

        return obs, info

    def key_handler(self, event):
        """Handle key press events."""
        if event.key == "escape":
            plt.close("all")
            self.running = False
        elif event.key == "backspace":
            self.reset()

    def step(self, action):
        """Take a step in the environment and update visualizations."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate shaped reward using the updated method
        shaped_reward, actual_reward = self.agent.shaped_reward(reward, info)

        # Track rewards
        step_count = len(self.steps) + 1
        self.steps.append(step_count)
        self.shaped_rewards.append(shaped_reward)
        self.actual_rewards.append(actual_reward)

        # Update cumulative rewards
        prev_cum_shaped = self.cumulative_shaped[-1] if self.cumulative_shaped else 0
        prev_cum_actual = self.cumulative_actual[-1] if self.cumulative_actual else 0
        self.cumulative_shaped.append(prev_cum_shaped + shaped_reward)
        self.cumulative_actual.append(prev_cum_actual + actual_reward)

        # Get image
        img = obs["image"] if self.render_agent_view else self.env.get_frame()

        # Update visualization
        self.update_env_display(img)
        self.update_info_panel(obs, info, action, shaped_reward, actual_reward)
        self.update_reward_plot()
        plt.draw()
        plt.pause(0.001)

        return obs, shaped_reward, actual_reward, terminated, truncated, info

    def run_episode(self):
        """Run a complete episode with the trained agent."""
        self.running = True
        obs, info = self.reset()
        self.agent.update_state(obs, info)

        total_shaped_reward = 0
        total_actual_reward = 0
        success = False

        print(f"Starting new episode with task: {self.env.task}")

        while self.running:
            # Get action from agent
            with torch.no_grad():
                action, cost = self.agent.choose_action(obs, info, testing=True)

            # Take step in environment
            obs, shaped_reward, actual_reward, terminated, truncated, info = self.step(
                action
            )

            # Update agent state
            self.agent.update_state(obs, info)

            # Update totals
            total_shaped_reward += shaped_reward
            total_actual_reward += actual_reward

            if actual_reward > 0:
                success = True
                print(f"Success! Task completed with reward {actual_reward}")

            # Pause between steps
            time.sleep(self.rate)

            if terminated or truncated:
                print(f"Episode ended: {'Success' if success else 'Failed'}")
                print(f"Total shaped reward: {total_shaped_reward:.4f}")
                print(f"Total actual reward: {total_actual_reward:.4f}")

                # Pause at the end to show final state
                time.sleep(2)
                if self.running:
                    obs, info = self.reset()
                    self.agent.update_state(obs, info)
                    total_shaped_reward = 0
                    total_actual_reward = 0
                    success = False
                    print(f"Starting new episode with task: {self.env.task}")

    def simulate(self):
        """Main simulation function."""
        self.setup_visualization()
        self.run_episode()
        plt.ioff()  # Turn off interactive mode when done


def simulate_agent(
    model_path="best",
    rate=0.1,
    env_name="homegrid-task",
    agent_view=False,
    checkpoint_number=13,
):
    """
    Simulates a trained agent in the homegrid environment.

    Args:
        model_path: Path to the model checkpoint or "best" for best model
        rate: Time delay between steps (seconds)
        env_name: Name of the environment
        agent_view: Whether to render from agent's perspective
        checkpoint_number: Checkpoint folder number (e.g., 13 for checkpoints13)
    """
    simulator = AgentSimulator(
        env_name=env_name,
        model_path=model_path,
        rate=rate,
        render_agent_view=agent_view,
        checkpoint_number=checkpoint_number,
    )
    simulator.simulate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a trained agent in the homegrid environment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=5,
        help='Path to the model or "best" for best model',
    )
    parser.add_argument(
        "--rate", type=float, default=0, help="Time delay between steps (seconds)"
    )
    parser.add_argument(
        "--env", type=str, default="homegrid-task", help="Environment name"
    )
    parser.add_argument(
        "--agent_view", action="store_true", help="Render from agent perspective"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=16,
        help="Checkpoint folder number (e.g., 13 for checkpoints13)",
    )

    args = parser.parse_args()

    print("Starting agent simulation...")
    print(f"Model: {args.model}")
    print(f"Environment: {args.env}")
    print(f"Step delay: {args.rate} seconds")
    print(f"Agent view: {args.agent_view}")
    print(f"Checkpoint number: {args.checkpoint}")

    simulate_agent(args.model, args.rate, args.env, args.agent_view, args.checkpoint)
