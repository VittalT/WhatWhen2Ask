# utils.py

"""Utility functions for the homegrid environment."""
import requests
from io import BytesIO
from PIL import Image


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
    if symbolic_state["objects"]:
        description += "The objects in the house are:\n"
        for obj in symbolic_state["objects"]:
            if not obj["invisible"]:
                name = obj["name"]
                room = room_map[obj["room"]]
                pos = tuple(int(x) for x in obj["pos"])
                state = obj["state"]
                state_text = f" in state {state}" if state else ""
                description += (
                    f"- a {name} in the {room} at location {pos}{state_text}.\n"
                )
    else:
        description += (
            "No objects are visible within the agent's current field of view.\n"
        )

    return description.strip()


def format_prompt(task, info):
    """Format the prompt for the LLM models."""
    context = format_symbolic_state(info["symbolic_state"])
    prompt_text = f"""
You are assisting a reinforcement learning agent navigating a grid-based house to complete tasks by interacting with objects. The agent is currently uncertain and has asked for your help.

Agent task: {task}
Available agent actions: left, right, up, down, pickup, drop, get, pedal, grasp, lift

The agent uses a Deep Q-Network (DQN) that takes in a partially observable egocentric view, state context (including memory from previous steps), and your hint to decide which action to take.

State Context:
Locations are given in the format (x, y), where x represents horizontal position (left to right) and y represents vertical position (top to bottom).
{context}

The image provided is the agent's current egocentric view.

Think step by step to reason through the current situation. Then, provide one concise, specific, and actionable hint that can help the agent over the next several steps.
Do not repeat the task or state context, the agent already has these. Instead, infer a new, helpful insight that will inform the agent's upcoming decisions — such as which direction to explore, where an object might be located, or what action might be effective.

Hint:
"""
    return prompt_text


def format_action_prompt(task, info):
    context = format_symbolic_state(info["symbolic_state"])
    prompt_text = f"""
You are an intelligent robot navigating a grid-based house to complete tasks by interacting with objects.

Task: {task}
Available actions: left, right, up, down, pickup, drop, get, pedal, grasp, lift

### Important Rules:
- Moving in a direction (left, right, up, down) causes the agent to move one tile in that direction and then face that direction.
- You can only interact (pickup, drop, get, pedal, grasp, lift) objects if you are facing them.
- You can only carry one object at a time.
- There may be obstacles in the way that prevent you from moving in a direction.

### State Context:
Some of the context is memory from previous steps. Locations are given in the format (x, y), where x represents horizontal position (left to right) and y represents vertical position (top to bottom).
{context}

The image provided is your current partially observable egocentric view.

### Output Format:
Think step by step to reason through the current situation.
Then output the best next action as follows:
First line: the chosen action (in lowercase)
Second line: a brief explanation of your reasoning

Action:
"""
    return prompt_text


def get_dummy_state():
    dummy_task = "move the fruit to the dining room"

    # Use a known working image URL (Unsplash image).
    image_url = "https://images.unsplash.com/photo-1516117172878-fd2c41f4a759?ixlib=rb-1.2.1&auto=format&fit=crop&w=634&q=80"

    # Fetch the image.
    response = requests.get(image_url)
    response.raise_for_status()  # Ensure the request was successful.
    dummy_image = Image.open(BytesIO(response.content))

    # Example state and task
    dummy_context = {
        "symbolic_state": {
            "step": 5,
            "agent": {"pos": (3, 7), "room": "K", "dir": 3, "carrying": None},
            "objects": [
                {
                    "name": "recycling bin",
                    "type": "Storage",
                    "pos": (12, 10),
                    "room": "D",
                    "state": "closed",
                    "action": "pedal",
                    "invisible": None,
                    "contains": [],
                },
                {
                    "name": "compost bin",
                    "type": "Storage",
                    "pos": (11, 1),
                    "room": "L",
                    "state": "open",
                    "action": "grasp",
                    "invisible": None,
                    "contains": [],
                },
                {
                    "name": "fruit",
                    "type": "Pickable",
                    "pos": (12, 2),
                    "room": "L",
                    "state": None,
                    "action": None,
                    "invisible": False,
                    "contains": None,
                },
                {
                    "name": "papers",
                    "type": "Pickable",
                    "pos": (3, 10),
                    "room": "K",
                    "state": None,
                    "action": None,
                    "invisible": False,
                    "contains": None,
                },
                {
                    "name": "plates",
                    "type": "Pickable",
                    "pos": (9, 1),
                    "room": "L",
                    "state": None,
                    "action": None,
                    "invisible": True,
                    "contains": None,
                },
                {
                    "name": "bottle",
                    "type": "Pickable",
                    "pos": (10, 8),
                    "room": "D",
                    "state": None,
                    "action": None,
                    "invisible": True,
                    "contains": None,
                },
            ],
            "front_obj": None,
            "unsafe": {"name": None, "poss": {}, "end": -1},
        }
    }
    dummy_state = (dummy_image, dummy_context)
    return dummy_state, dummy_task


if __name__ == "__main__":
    task, state = get_dummy_state()
    print(task)
    print(state)
