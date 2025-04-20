import os
import torch
import matplotlib.pyplot as plt
from homegrid.DQN import DQNAgent

# Use a consistent checkpoint directory.
checkpoint_dir = "checkpoints6"


def load_agent(checkpoint_episode=None, env_name="homegrid-task", episodes=0):
    """
    Creates a new DQNAgent instance.
    If a valid checkpoint_path is provided, the agent's model is loaded from that checkpoint.
    """
    agent = DQNAgent(env_name=env_name, episodes=episodes)
    if checkpoint_episode is not None:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_checkpoint_{checkpoint_episode}.pth"
        )
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            agent.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"Checkpoint file not found: {checkpoint_path}\n")
    return agent


def train_agent(num_episodes=3000):
    """
    Create a new agent and train it for a specified number of episodes.
    """
    agent = load_agent()
    print("Training the agent...")
    agent.train(episodes=num_episodes)


def test_agent(num_episodes=100000, checkpoint_episode=None, render=False):
    """
    Create a new agent (with random initialization) and test it for the given number of episodes.
    """
    agent = load_agent(checkpoint_episode)
    print("Testing the agent...")
    agent.test(episodes=num_episodes, render=render)


def test_iterated(num_episodes=100000):
    """
    Evaluate baseline (untrained) performance, then evaluate checkpoints iteratively.
    Plots test performance vs training episodes.
    """
    # Baseline: untrained agent.
    print("Evaluating baseline (no training) performance...")
    baseline_agent = load_agent()
    baseline_score = baseline_agent.test(episodes=num_episodes)
    print(f"Baseline Test Score (No Training): {baseline_score}\n")

    checkpoint_scores = {}

    # Evaluate checkpoints saved every 250 episodes from 250 to 3000.
    for episode in range(250, 3001, 250):
        agent = load_agent(episode)
        test_score = agent.test(episodes=num_episodes)
        checkpoint_scores[episode] = test_score
        print(f"Checkpoint {episode}: Test Score = {test_score}\n")

    # Plot performance vs training episodes.
    episodes = list(checkpoint_scores.keys())
    scores = list(checkpoint_scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, scores, marker="o", label="Trained Agent")
    plt.axhline(
        y=baseline_score, color="r", linestyle="--", label="Baseline (No Training)"
    )
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Test Reward")
    plt.title("Test Performance vs. Training Episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Loads an agent from a checkpoint, further trains it, and then tests it.
    """
    checkpoint_episode = 300
    agent = load_agent(checkpoint_episode)

    print("Further Training the agent...")
    agent.train(episodes=1200)

    print("Final Testing the agent...")
    agent.test(episodes=10000, render=False)


if __name__ == "__main__":
    # Uncomment the desired function call:

    # For training from scratch:
    train_agent(num_episodes=3000)
    test_agent(num_episodes=50000, checkpoint_episode=3000)

    # For iterated testing of saved checkpoints:
    test_iterated(num_episodes=10000)

    # Alternatively, load a checkpoint, train further, and test:
    # main()

    # Or just test an untrained agent:
    # test_agent(num_episodes=100000, render=False)
