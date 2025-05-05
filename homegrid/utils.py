# utils.py

"""Utility functions for the homegrid environment."""


def format_symbolic_state(symbolic_state):
    """Format the symbolic state into a human-readable string."""
    dir_map = {0: "right", 1: "down", 2: "left", 3: "up"}
    room_map = {"L": "living room", "D": "dining room", "K": "kitchen"}

    agent = symbolic_state["agent"]
    agent_pos = tuple(int(x) for x in agent["pos"])
    agent_dir = dir_map[agent["dir"]]
    carrying = agent["carrying"] if agent["carrying"] is not None else "nothing"
    agent_room = room_map[agent["room"]]

    description = f"The agent is in the {agent_room} at location {agent_pos}, facing {agent_dir}, carrying {carrying}.\n"
    description += "The objects in the house are:\n"

    for obj in symbolic_state["objects"]:
        if not obj["invisible"]:
            name = obj["name"]
            room = room_map[obj["room"]]
            pos = tuple(int(x) for x in obj["pos"])
            state = obj["state"]
            state_text = f" in state {state}" if state else ""
            description += f"- a {name} in the {room} at location {pos}{state_text}.\n"

    return description.strip()


def format_prompt(task, info):
    """Format the prompt for the LLM models."""
    context = format_symbolic_state(info["symbolic_state"])
    prompt_text = f"""
You are assisting a reinforcement learning agent navigating a grid-based house to complete tasks by interacting with objects.
The task is to {task}.

The agent uses a DQN that takes in its visual observation, state context, and your hint to decide which action to take. Available actions: left, right, up, down, pickup, drop, get, pedal, grasp, lift.

Current state:
{context}

You are also provided with the agent's current visual observation.

Do not repeat the task or any obvious information already present in the context — the agent already has this.

Instead, analyze the image and state to infer a new, helpful insight that will inform the agent's upcoming decisions — such as which direction to move, where an object is located, or what action might be effective.

Provide one concise, specific, and actionable hint that can help the agent over the next several steps.
"""
    return prompt_text
