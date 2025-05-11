#!/usr/bin/env python3

# manual_control.py

import gym
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from tokenizers import Tokenizer
import cv2
from pprint import pprint
import torch
import textwrap

from homegrid.window import Window
from homegrid.DQN import DQNAgent

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
    "pot_blocked": 0,
    "potential": 0,
    "shaped_reward": 0,
    "base_reward": 0,
    "total_reward": 0,
}
reward_history = {k: [] for k in reward_components.keys()}
reward_fig = None
reward_canvas = None

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Limit GPU memory usage if available
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)


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

    # Create subplot for each component with more space between
    # gs = reward_fig.add_gridspec(4, 2, hspace=0.7, wspace=0.3)  # more space
    gs = reward_fig.add_gridspec(4, 2, hspace=1.2, wspace=0.3)

    # Subplot 1: Distance
    ax1 = reward_fig.add_subplot(gs[0, 0])
    ax1.plot(reward_history["pot_dist"], "r-")
    ax1.set_title("Distance", fontsize=20)
    ax1.set_ylim(
        min(min(reward_history["pot_dist"]) - 0.1, -0.1),
        max(max(reward_history["pot_dist"]) + 0.1, 0.1),
    )

    # Subplot 2: Orientation
    ax2 = reward_fig.add_subplot(gs[0, 1])
    ax2.plot(reward_history["pot_orientation"], "g-")
    ax2.set_title("Orientation", fontsize=20)
    ax2.set_ylim(
        min(min(reward_history["pot_orientation"]) - 0.1, -0.1),
        max(max(reward_history["pot_orientation"]) + 0.1, 0.1),
    )

    # Subplot 3: Carrying
    ax3 = reward_fig.add_subplot(gs[1, 0])
    ax3.plot(reward_history["pot_carrying"], "b-")
    ax3.set_title("Carrying", fontsize=20)
    ax3.set_ylim(
        min(min(reward_history["pot_carrying"]) - 0.1, -0.1),
        max(max(reward_history["pot_carrying"]) + 0.1, 0.1),
    )

    # Subplot 4: Exploration
    ax4 = reward_fig.add_subplot(gs[1, 1])
    ax4.plot(reward_history["pot_expl"], "c-")
    ax4.set_title("Exploration", fontsize=20)
    ax4.set_ylim(
        min(min(reward_history["pot_expl"]) - 0.1, -0.1),
        max(max(reward_history["pot_expl"]) + 0.1, 0.1),
    )

    # Subplot 5: Time
    ax5 = reward_fig.add_subplot(gs[2, 0])
    ax5.plot(reward_history["pot_time"], "m-")
    ax5.set_title("Time", fontsize=20)
    ax5.set_ylim(
        min(min(reward_history["pot_time"]) - 0.1, -0.1),
        max(max(reward_history["pot_time"]) + 0.1, 0.1),
    )

    # Subplot 6: Blocking Penalty
    ax6 = reward_fig.add_subplot(gs[2, 1])
    ax6.plot(reward_history["pot_blocked"], "r-")
    ax6.set_title("Blocking", fontsize=20)
    ax6.set_ylim(
        min(min(reward_history["pot_blocked"]) - 0.1, -0.1),
        max(max(reward_history["pot_blocked"]) + 0.1, 0.1),
    )

    # Subplot 7: Shaped Reward
    ax7 = reward_fig.add_subplot(gs[3, 0])
    ax7.plot(reward_history["shaped_reward"], color="purple")
    ax7.set_title("Shaped Reward", fontsize=20)
    ax7.set_ylim(
        min(min(reward_history["shaped_reward"]) - 0.1, -0.1),
        max(max(reward_history["shaped_reward"]) + 0.1, 0.1),
    )

    # Subplot 8: Base Reward
    ax8 = reward_fig.add_subplot(gs[3, 1])
    ax8.plot(reward_history["base_reward"], color="orange")
    ax8.set_title("Raw Reward", fontsize=20)
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

    # Initialize agent for reward shaping computation if not initialized
    if agent is None:
        agent = DQNAgent(
            env_name="homegrid-task", episodes=0, checkpoint_dir="./checkpoints"
        )
        # Explicitly move agent to correct device
        agent.device = device
        agent.env = env

    # Use agent's reset method which already calls env.reset
    obs, info = agent.reset()
    img = obs["image"] if agent_view else env.get_frame()

    # Initialize potential tracking
    prev_potential = agent.compute_potential(info)

    # Reset reward components
    reward_components = {
        "pot_dist": 0,
        "pot_orientation": 0,
        "pot_carrying": 0,
        "pot_expl": 0,
        "pot_time": 0,
        "pot_blocked": 0,
        "potential": 0,
        "shaped_reward": 0,
        "base_reward": 0,
        "total_reward": 0,
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


def draw_info_panel_on_image(img, info, action, reward_components, env):
    # Panel size in pixels (about 18 tiles wide, 5 tiles high)
    box_w = 18 * 32  # 576 px
    box_h = 5 * 32  # 160 px

    # Room code to full name mapping
    room_map = {"K": "kitchen", "D": "dining room", "L": "living room"}

    # Prepare info text, one field per line
    agent_info = info["symbolic_state"]["agent"]
    step_count = env.step_cnt
    room_code = agent_info["room"]
    room = room_map.get(room_code, str(room_code))
    direction = ["Right", "Down", "Left", "Up"][agent_info["dir"]]
    carrying = agent_info["carrying"] or "nothing"
    front_obj = info["symbolic_state"].get("front_obj", None) or "nothing"
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
    shaped_reward = reward_components["shaped_reward"]
    actual_reward = reward_components["base_reward"]
    task_str = env.task

    lines = [
        f"Step: {step_count}",
        f"Room: {room}",
        f"Dir: {direction}",
        f"Action: {action_str}",
        f"Carrying: {carrying}",
        f"In front: {front_obj}",
        f"Shaped: {shaped_reward:.4f}",
        f"Actual: {actual_reward:.4f}",
    ]

    # Draw text lines with smaller font (no background box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_color = (0, 0, 0)
    thickness = 1
    y0, dy = 24, 18
    for i, line in enumerate(lines):
        y = y0 + i * dy
        if y > box_h - 8:
            break
        cv2.putText(
            img, line, (10, y), font, font_scale, font_color, thickness, cv2.LINE_AA
        )

    # Draw the task at the bottom of the environment image, centered
    h, w = img.shape[:2]
    task_lines = textwrap.wrap("Task: " + task_str, width=70)
    task_box_h = 22 * len(task_lines) + 10
    task_box_y0 = h - task_box_h - 2
    for i, line in enumerate(task_lines):
        text_size = cv2.getTextSize(line, font, 0.48, 1)[0]
        x = (w - text_size[0]) // 2
        y = task_box_y0 + 20 + i * 22
        cv2.putText(img, line, (x, y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def step(env, window, action, agent_view=False):
    global agent, prev_potential, reward_components

    obs, reward, terminated, truncated, info = env.step(action)

    # Get the shaped reward and extract components
    if agent:
        # Update the state in the agent
        agent.update_state(obs, info, action)

        # Calculate current potential and get components
        current_potential = agent.compute_potential(info)
        components = agent.get_potential_components()

        # Store the components
        reward_components["pot_dist"] = components["pot_dist"]
        reward_components["pot_orientation"] = components["pot_orientation"]
        reward_components["pot_carrying"] = components["pot_carrying"]
        reward_components["pot_expl"] = components["pot_expl"]
        reward_components["pot_time"] = components["pot_time"]
        reward_components["pot_blocked"] = components["pot_blocked"]
        reward_components["potential"] = current_potential

        # Calculate shaped reward using the formula: F(s,s') = γΦ(s') - Φ(s)
        gamma = 0.99
        shaped_reward = reward + (gamma * current_potential - prev_potential)
        reward_components["shaped_reward"] = shaped_reward
        reward_components["base_reward"] = reward
        reward_components["total_reward"] += reward

        # Update previous potential for next step
        prev_potential = current_potential

        # Update the agent's current step counter
        agent.current_step += 1

    # Print information about the state
    print(f"step={env.step_cnt}, reward={reward:.2f}")
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

        # Print blocking penalty information if available
        if "pot_blocked" in components and components["pot_blocked"] > 0:
            print(
                f"BLOCKED MOVE DETECTED! Penalty: {components['weighted_blocked']:.4f}"
            )

        # Print LLM info if available
        llm_type = getattr(agent, "last_llm_type", None)
        llm_conf = getattr(agent, "last_llm_confidence", None)
        llm_hint = getattr(agent, "current_hint", None)
        llm_accepted = getattr(agent, "last_llm_accepted", None)
        if llm_type or llm_conf or llm_hint:
            print("LLM Assistance:")
            print(f"  Type: {llm_type if llm_type is not None else 'N/A'}")
            print(
                f"  Confidence: {llm_conf:.4f}"
                if llm_conf is not None
                else "  Confidence: N/A"
            )
            print(
                f"  Accepted: {'Yes' if llm_accepted else 'No'}"
                if llm_accepted is not None
                else "  Accepted: N/A"
            )
            print(f"  Hint: {llm_hint if llm_hint is not None else 'N/A'}")

    print("-" * 20)
    print("Info: ", obs["token_embed"].shape, obs["image"].shape)
    a = "obs, reward, terminated, truncated, info"
    b = obs, reward, terminated, truncated, info
    # for x, y in zip(a, b):
    #     pprint(x)
    #     pprint(y)
    print(info)

    # Draw info panel on the environment image
    img = obs["image"] if agent_view else env.get_frame()
    img_with_info = draw_info_panel_on_image(
        img.copy(), info, action, reward_components, env
    )
    redraw_with_rewards(window, img_with_info)

    # Only show the task in the caption (or nothing)
    window.set_caption("")

    if terminated:
        print(f"terminated! r={reward}")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    elif reward_components["total_reward"] >= 1:
        print(f"success! r={reward}")
        reset(env, window)


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
