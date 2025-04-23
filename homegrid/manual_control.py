#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from tokenizers import Tokenizer
import cv2

from homegrid.window import Window
from homegrid.DQN import DQNAgent, get_fasttext_embedding

tok = Tokenizer.from_pretrained("t5-small")

# Global variables to track reward components
agent = None
prev_potential = None
reward_components = {
    "pot_dist": 0,
    "pot_orientation": 0,
    "pot_carrying": 0,
    "pot_expl": 0,
    "pot_time": 0,
    "potential": 0,
    "shaped_reward": 0,
    "base_reward": 0,
}
reward_history = {k: [] for k in reward_components.keys()}
reward_fig = None
reward_canvas = None


def create_reward_visualization():
    """Create a matplotlib figure for visualizing reward components"""
    global reward_fig, reward_canvas
    reward_fig = Figure(figsize=(10, 6))
    reward_canvas = FigureCanvasAgg(reward_fig)
    return reward_fig, reward_canvas


def update_reward_visualization():
    """Update the reward visualization figure with current component values"""
    global reward_fig, reward_canvas, reward_components, reward_history

    # Add current values to history (limited to last 20 steps)
    for k, v in reward_components.items():
        reward_history[k].append(v)
        if len(reward_history[k]) > 20:
            reward_history[k] = reward_history[k][-20:]

    # Clear the figure
    reward_fig.clear()

    # Create subplot for each component
    gs = reward_fig.add_gridspec(4, 2, hspace=0.4)

    # Subplot 1: Distance potential
    ax1 = reward_fig.add_subplot(gs[0, 0])
    ax1.plot(reward_history["pot_dist"], "r-")
    ax1.set_title("Distance (-0.05 * pot_dist)")
    ax1.set_ylim(
        min(min(reward_history["pot_dist"]) - 0.1, -0.1),
        max(max(reward_history["pot_dist"]) + 0.1, 0.1),
    )

    # Subplot 2: Orientation potential
    ax2 = reward_fig.add_subplot(gs[0, 1])
    ax2.plot(reward_history["pot_orientation"], "g-")
    ax2.set_title("Orientation (0.1 * pot_orientation)")
    ax2.set_ylim(
        min(min(reward_history["pot_orientation"]) - 0.1, -0.1),
        max(max(reward_history["pot_orientation"]) + 0.1, 0.1),
    )

    # Subplot 3: Carrying potential
    ax3 = reward_fig.add_subplot(gs[1, 0])
    ax3.plot(reward_history["pot_carrying"], "b-")
    ax3.set_title("Carrying (0.5 * pot_carrying)")
    ax3.set_ylim(
        min(min(reward_history["pot_carrying"]) - 0.1, -0.1),
        max(max(reward_history["pot_carrying"]) + 0.1, 0.1),
    )

    # Subplot 4: Exploration potential
    ax4 = reward_fig.add_subplot(gs[1, 1])
    ax4.plot(reward_history["pot_expl"], "c-")
    ax4.set_title("Exploration (0.025 * pot_expl)")
    ax4.set_ylim(
        min(min(reward_history["pot_expl"]) - 0.1, -0.1),
        max(max(reward_history["pot_expl"]) + 0.1, 0.1),
    )

    # Subplot 5: Time penalty
    ax5 = reward_fig.add_subplot(gs[2, 0])
    ax5.plot(reward_history["pot_time"], "m-")
    ax5.set_title("Time (-0.025 * pot_time)")
    ax5.set_ylim(
        min(min(reward_history["pot_time"]) - 0.1, -0.1),
        max(max(reward_history["pot_time"]) + 0.1, 0.1),
    )

    # Subplot 6: Overall potential
    ax6 = reward_fig.add_subplot(gs[2, 1])
    ax6.plot(reward_history["potential"], "k-")
    ax6.set_title("Total Potential")
    ax6.set_ylim(
        min(min(reward_history["potential"]) - 0.1, -0.1),
        max(max(reward_history["potential"]) + 0.1, 0.1),
    )

    # Subplot 7: Shaped reward
    ax7 = reward_fig.add_subplot(gs[3, 0])
    ax7.plot(reward_history["shaped_reward"], color="purple")
    ax7.set_title("Shaped Reward")
    ax7.set_ylim(
        min(min(reward_history["shaped_reward"]) - 0.1, -0.1),
        max(max(reward_history["shaped_reward"]) + 0.1, 0.1),
    )

    # Subplot 8: Base reward
    ax8 = reward_fig.add_subplot(gs[3, 1])
    ax8.plot(reward_history["base_reward"], color="orange")
    ax8.set_title("Base Reward")
    ax8.set_ylim(
        min(min(reward_history["base_reward"]) - 0.1, -0.1),
        max(max(reward_history["base_reward"]) + 0.1, 0.1),
    )

    reward_fig.tight_layout()
    reward_canvas.draw()

    # Convert to numpy array
    s, (width, height) = reward_canvas.print_to_buffer()
    reward_img = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    return reward_img


def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None, agent_view=False):
    global agent, prev_potential, reward_components, reward_history

    obs, info = env.reset()
    img = obs["image"] if agent_view else env.get_frame()

    # Initialize agent for reward shaping computation
    agent = DQNAgent(
        env_name="homegrid-task", episodes=0, checkpoint_dir="./checkpoints"
    )
    agent.env = env
    agent.reset_episode(info)
    prev_potential = agent.compute_potential(info)

    # Reset reward components
    reward_components = {
        "pot_dist": 0,
        "pot_orientation": 0,
        "pot_carrying": 0,
        "pot_expl": 0,
        "pot_time": 0,
        "potential": 0,
        "shaped_reward": 0,
        "base_reward": 0,
    }
    reward_history = {k: [0] for k in reward_components.keys()}

    # Create visualization window if it doesn't exist
    if reward_fig is None:
        create_reward_visualization()

    redraw_with_rewards(window, img)


def redraw_with_rewards(window, img):
    """Combine game image with reward visualization and display"""
    if reward_fig is None:
        create_reward_visualization()

    # Generate reward visualization image
    reward_img = update_reward_visualization()

    # Convert game image to RGBA if it's RGB
    if img.shape[2] == 3:
        img_rgba = np.concatenate(
            [img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2
        )
    else:
        img_rgba = img

    # Resize reward visualization to reasonable height
    target_height = min(300, img_rgba.shape[0])
    aspect_ratio = reward_img.shape[1] / reward_img.shape[0]
    target_width = int(target_height * aspect_ratio)

    # Make sure target_width doesn't exceed image width
    if target_width > img_rgba.shape[1]:
        target_width = img_rgba.shape[1]
        target_height = int(target_width / aspect_ratio)

    reward_img_resized = cv2.resize(reward_img, (target_width, target_height))

    # Create combined image (game image on top, reward visualization below)
    combined_height = img_rgba.shape[0] + reward_img_resized.shape[0]
    combined_width = max(img_rgba.shape[1], reward_img_resized.shape[1])

    combined_img = np.zeros((combined_height, combined_width, 4), dtype=np.uint8)

    # Add game image
    combined_img[: img_rgba.shape[0], : img_rgba.shape[1]] = img_rgba

    # Add reward visualization
    y_offset = img_rgba.shape[0]
    combined_img[
        y_offset : y_offset + reward_img_resized.shape[0], : reward_img_resized.shape[1]
    ] = reward_img_resized

    # Display combined image
    window.show_img(combined_img)


def step(env, window, action, agent_view=False):
    global agent, prev_potential, reward_components

    obs, reward, terminated, truncated, info = env.step(action)

    # Get the shaped reward and extract components
    if agent:
        # Calculate current potential and get components
        current_potential = agent.compute_potential(info)
        components = agent.get_potential_components()

        # Update visited rooms and cells in the agent
        agent_pos = info["symbolic_state"]["agent"]["pos"]
        agent_room = info["symbolic_state"]["agent"]["room"].lower()
        agent.visited_rooms.add(agent_room)
        agent.visited_cells.add(tuple(agent_pos))

        # Store the components
        reward_components["pot_dist"] = components["pot_dist"]
        reward_components["pot_orientation"] = components["pot_orientation"]
        reward_components["pot_carrying"] = components["pot_carrying"]
        reward_components["pot_expl"] = components["pot_expl"]
        reward_components["pot_time"] = components["pot_time"]
        reward_components["potential"] = current_potential

        # Calculate shaped reward using the formula: F(s,s') = γΦ(s') - Φ(s)
        gamma = 0.99
        shaped_reward = reward + (gamma * current_potential - prev_potential)
        reward_components["shaped_reward"] = shaped_reward
        reward_components["base_reward"] = reward

        # Update previous potential for next step
        prev_potential = current_potential

        # Update the agent's current step counter
        agent.current_step += 1

    # Print information about the state
    token = tok.decode([obs["token"]])
    print(f"step={env.step_cnt}, reward={reward:.2f}")
    print("Token: ", token)
    print(
        "Language: ", obs["log_language_info"] if "log_language_info" in obs else "None"
    )
    print("Task: ", env.task)
    if agent:
        print("Shaped Reward Components:")
        for comp, val in reward_components.items():
            print(f"  {comp}: {val:.4f}")

        # Print objectives and current objective index
        components = agent.get_potential_components()
        current_objective_idx = components["current_objective_idx"]
        objectives = components["objectives"]

        print("Current objective index:", current_objective_idx)
        print("Objectives:")
        for i, obj in enumerate(objectives):
            marker = "→" if i == current_objective_idx else " "
            print(f"  {marker} {i}: {obj['name']} at pos {obj['pos']}")

    print("-" * 20)
    print("Info: ", info)

    window.set_caption(
        f"r={reward:.2f} token_id={obs['token']} token="
        f"{token} \ncurrent: {obs['log_language_info'][:50]}..."
    )

    if terminated:
        print(f"terminated! r={reward}")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    else:
        img = obs["image"] if agent_view else env.get_frame()
        redraw_with_rewards(window, img)


def key_handler(env, window, event, agent_view=False):
    print("pressed", event.key)
    step_ = lambda a: step(env, window, a, agent_view)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step_(env.actions.left)
        return
    if event.key == "right":
        step_(env.actions.right)
        return
    if event.key == "up":
        step_(env.actions.up)
        return
    if event.key == "down":
        step_(env.actions.down)
        return

    if event.key == "k":
        step_(env.actions.pickup)
        return
    if event.key == "d":
        step_(env.actions.drop)
        return
    if event.key == "g":
        step_(env.actions.get)
        return
    if event.key == "p":
        step_(env.actions.pedal)
        return
    if event.key == "r":
        step_(env.actions.grasp)
        return
    if event.key == "l":
        step_(env.actions.lift)
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="homegrid-task"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()
    env = gym.make(args.env, disable_env_checker=True)

    for k in plt.rcParams:
        if "keymap" in k:
            plt.rcParams[k] = []
    window = Window("homegrid - " + args.env)

    window.reg_key_handler(
        lambda event: key_handler(env, window, event, args.agent_view)
    )

    # Create the reward visualization
    create_reward_visualization()

    seed = None if args.seed == -1 else args.seed
    reset(env, window, seed, args.agent_view)

    # Blocking event loop
    window.show(block=True)
