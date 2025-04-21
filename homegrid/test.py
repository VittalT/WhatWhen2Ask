import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from homegrid.DQN import DQNAgent
import json
import time

# Use a consistent checkpoint directory
checkpoint_dir = "checkpoints7"
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a log directory for tracking metrics
log_dir = os.path.join(checkpoint_dir, "logs")
os.makedirs(log_dir, exist_ok=True)


def load_agent(checkpoint_path=None, env_name="homegrid-task", episodes=0):
    """
    Creates a new DQNAgent instance and loads from checkpoint if provided.

    Args:
        checkpoint_path: Path to checkpoint file (can be a specific path or episode number)
        env_name: Name of the environment
        episodes: Number of episodes to train

    Returns:
        Loaded agent
    """
    agent = DQNAgent(env_name=env_name, episodes=episodes)

    if checkpoint_path is not None:
        # Handle both direct paths and episode numbers
        if isinstance(checkpoint_path, int):
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_checkpoint_{checkpoint_path}.pth"
            )
        elif (
            not os.path.exists(checkpoint_path)
            and str(checkpoint_path).lower() == "best"
        ):
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        elif (
            not os.path.exists(checkpoint_path)
            and str(checkpoint_path).lower() == "final"
        ):
            checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            try:
                # Handle different checkpoint formats
                checkpoint = torch.load(checkpoint_path)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    agent.model.load_state_dict(checkpoint["model_state_dict"])
                    if "optimizer_state_dict" in checkpoint:
                        agent.optimizer.load_state_dict(
                            checkpoint["optimizer_state_dict"]
                        )
                    if "epsilon" in checkpoint:
                        agent.epsilon = checkpoint["epsilon"]
                    if "total_steps" in checkpoint:
                        agent.total_steps = checkpoint["total_steps"]
                    print(
                        f"Loaded checkpoint with epsilon {agent.epsilon:.4f} and {agent.total_steps} total steps"
                    )
                else:
                    # Legacy format (just model state dict)
                    agent.model.load_state_dict(checkpoint)
                    print(f"Loaded legacy checkpoint format (only model weights)")

                # Make sure the target network is synced
                agent.update_target_network()
                return agent
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Creating new agent with random initialization")
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")
            print("Creating new agent with random initialization")

    return agent


def train_agent(num_episodes=1000, continue_from=None, save_interval=100):
    """
    Train an agent from scratch or continue training from a checkpoint.

    Args:
        num_episodes: Number of episodes to train
        continue_from: Checkpoint path or episode number to continue from
        save_interval: How often to save checkpoints (in episodes)
    """
    start_time = time.time()

    # Load agent, continuing from checkpoint if specified
    agent = load_agent(continue_from)
    agent.checkpoint_interval = save_interval  # Set custom checkpoint interval

    # Log training metadata
    metadata = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_episodes": num_episodes,
        "continue_from": str(continue_from) if continue_from else "None",
        "initial_epsilon": agent.epsilon,
        "learning_rate": agent.optimizer.param_groups[0]["lr"],
        "batch_size": agent.batch_size,
        "max_replay_buffer_size": agent.max_replay_buffer_size,
        "epsilon_decay": agent.epsilon_decay,
    }

    with open(
        os.path.join(log_dir, f"training_metadata_{int(start_time)}.json"), "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("Starting training with the following parameters:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Start training
    print(f"\nTraining agent for {num_episodes} episodes...")
    agent.train(episodes=num_episodes)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "total_steps": agent.total_steps,
        },
        final_path,
    )

    training_time = time.time() - start_time
    print(
        f"\nTraining completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)"
    )
    print(f"Final model saved to: {final_path}")

    return agent


def test_agent(checkpoint="best", num_episodes=1000, render=False):
    """
    Test a trained agent for a specified number of episodes.

    Args:
        checkpoint: Path to checkpoint file or episode number or "best"
        num_episodes: Number of episodes to test
        render: Whether to render the environment during testing
    """
    # Load the agent from checkpoint
    agent = load_agent(checkpoint)

    # Run testing
    print(f"\nTesting agent for {num_episodes} episodes...")
    start_time = time.time()
    average_reward = agent.test(episodes=num_episodes, render=render)
    test_time = time.time() - start_time

    print(f"Testing completed in {test_time:.1f} seconds")
    print(f"Average test reward: {average_reward:.4f}")

    return average_reward


def evaluate_checkpoints(checkpoint_range=(100, 1000, 100), test_episodes=500):
    """
    Evaluate multiple checkpoints to create a learning curve.

    Args:
        checkpoint_range: Tuple of (start, end, step) for checkpoint evaluation
        test_episodes: Number of episodes to test each checkpoint
    """
    results = {}
    start, end, step = checkpoint_range

    # Test untrained baseline first
    print("\nEvaluating baseline (untrained) agent...")
    baseline_agent = DQNAgent()
    baseline_score = baseline_agent.test(episodes=test_episodes)
    print(f"Baseline average reward: {baseline_score:.4f}")
    results[0] = baseline_score

    # Test each checkpoint
    for episode in range(start, end + 1, step):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_checkpoint_{episode}.pth"
        )
        if os.path.exists(checkpoint_path):
            print(f"\nEvaluating checkpoint at episode {episode}...")
            agent = load_agent(checkpoint_path)
            reward = agent.test(episodes=test_episodes)
            results[episode] = reward
            print(f"Checkpoint {episode} average reward: {reward:.4f}")
        else:
            print(f"Checkpoint for episode {episode} not found, skipping")

    # Also test best model if it exists
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        print("\nEvaluating best model...")
        best_agent = load_agent(best_path)
        best_reward = best_agent.test(episodes=test_episodes)
        results["best"] = best_reward
        print(f"Best model average reward: {best_reward:.4f}")

    # Create a learning curve plot
    plt.figure(figsize=(10, 6))
    episodes = [e for e in results.keys() if isinstance(e, int)]
    rewards = [results[e] for e in episodes]

    plt.plot(episodes, rewards, "o-", label="Checkpoints")
    plt.axhline(
        y=baseline_score, color="r", linestyle="--", label="Baseline (Untrained)"
    )

    if "best" in results:
        plt.axhline(y=results["best"], color="g", linestyle="--", label="Best Model")

    plt.xlabel("Training Episodes")
    plt.ylabel("Average Test Reward")
    plt.title("Learning Curve: Test Performance vs Training Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plt_path = os.path.join(log_dir, f"learning_curve_{int(time.time())}.png")
    plt.savefig(plt_path)
    plt.show()

    # Save results to JSON
    results_path = os.path.join(
        log_dir, f"checkpoint_evaluation_{int(time.time())}.json"
    )
    with open(results_path, "w") as f:
        # Convert non-serializable keys to strings
        serializable_results = {str(k): v for k, v in results.items()}
        json.dump(serializable_results, f, indent=4)

    print(f"Learning curve saved to: {plt_path}")
    print(f"Evaluation results saved to: {results_path}")

    return results


if __name__ == "__main__":
    # Uncomment one of the following code blocks to run:

    # OPTION 1: Train a new agent from scratch
    # Recommended training scheme for sparse rewards
    train_agent(num_episodes=2000, save_interval=100)
    test_agent(checkpoint="best", num_episodes=1000)

    # OPTION 2: Continue training from a checkpoint
    # train_agent(num_episodes=1000, continue_from="best", save_interval=100)
    # test_agent(checkpoint="final", num_episodes=1000)

    # OPTION 3: Evaluate checkpoints to create a learning curve
    # evaluate_checkpoints(checkpoint_range=(100, 2000, 200), test_episodes=200)

    # OPTION 4: Quick test of a specific checkpoint
    # test_agent(checkpoint=1000, num_episodes=100, render=False)
