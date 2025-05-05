import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from homegrid.DQN import DQNAgent
import json
import time
import multiprocessing
from datetime import datetime
import sys
from itertools import product


# Create a function to log output to both console and file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_agent(
    checkpoint_path=None, env_name="homegrid-task", episodes=0, use_gpu=True
):
    """
    Creates a new DQNAgent instance and loads from checkpoint if provided.

    Args:
        checkpoint_path: Path to checkpoint file (can be a specific path or episode number)
        env_name: Name of the environment
        episodes: Number of episodes to train
        use_gpu: Whether to use GPU acceleration (if available)

    Returns:
        Loaded agent
    """
    if use_gpu:
        # Optimizing CUDA performance
        if torch.cuda.is_available():
            # Set to benchmark mode for better performance when input sizes don't change much
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    agent = DQNAgent(
        env_name=env_name, episodes=episodes, checkpoint_dir=checkpoint_dir
    )

    # Track the episode number of the loaded checkpoint for proper numbering of future checkpoints
    loaded_episode = 0

    if checkpoint_path is not None:
        # Handle both direct paths and episode numbers
        if isinstance(checkpoint_path, int):
            loaded_episode = checkpoint_path
            checkpoint_path = os.path.join(
                training_dir, f"model_checkpoint_{checkpoint_path}.pth"
            )

        elif str(checkpoint_path) == "best":
            checkpoint_path = os.path.join(training_dir, "best_model.pth")
            # For "best" model, try to extract the episode number from the checkpoint
            # This will be set from the checkpoint data below

        elif str(checkpoint_path) == "final":
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

        # Load the model
        print(f"Loading checkpoint: {checkpoint_path}")
        # Load checkpoint with proper format
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epsilon" in checkpoint:
            agent.epsilon = checkpoint["epsilon"]
        if "total_steps" in checkpoint:
            agent.total_steps = checkpoint["total_steps"]
        if "episode" in checkpoint:
            loaded_episode = checkpoint["episode"]
            print(f"Loaded from episode {loaded_episode}")

        # Store the loaded episode number in the agent for proper checkpoint naming
        agent.previous_episode = loaded_episode

        print(
            f"Loaded checkpoint with epsilon {agent.epsilon:.4f} and {agent.total_steps} total steps"
        )

        # Make sure the target network is synced
        agent.update_target_network()
        return agent

    return agent


def train_agent(num_episodes=1000, continue_from=None, save_interval=100, use_gpu=True):
    """
    Train an agent from scratch or continue training from a checkpoint,
    optimized for GPU acceleration.

    Args:
        num_episodes: Number of episodes to train
        continue_from: Checkpoint path or episode number to continue from
        save_interval: How often to save checkpoints (in episodes)
        use_gpu: Whether to use GPU acceleration (if available)
    """
    # Record start time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set CUDA optimization parameters if using GPU
    if use_gpu and torch.cuda.is_available():
        # Empty cache to free up memory
        torch.cuda.empty_cache()
        # Set to higher precision for numerical stability
        torch.set_float32_matmul_precision("high")

    # Load agent, continuing from checkpoint if specified
    agent = load_agent(continue_from, use_gpu=use_gpu)
    agent.checkpoint_interval = save_interval  # Set custom checkpoint interval

    # Log training metadata with hardware info
    gpu_info = "None"
    if torch.cuda.is_available():
        gpu_info = (
            f"{torch.cuda.get_device_name(0)} - {torch.cuda.device_count()} device(s)"
        )

    metadata = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": timestamp,
        "num_episodes": num_episodes,
        "continue_from": str(continue_from) if continue_from else "None",
        "device": str(device),
        "gpu_info": gpu_info,
        "initial_epsilon": agent.epsilon,
        "learning_rate": agent.optimizer.param_groups[0]["lr"],
        "batch_size": agent.batch_size,
        "max_replay_buffer_size": agent.max_replay_buffer_size,
        "epsilon_decay": agent.epsilon_decay,
        "worker_threads": num_workers,
    }

    # Create training log file
    training_log_path = os.path.join(training_dir, f"training_log_{timestamp}.txt")
    training_log = open(training_log_path, "w")
    training_log.write("=== TRAINING LOG ===\n\n")

    # Log metadata
    training_log.write("TRAINING PARAMETERS:\n")
    for key, value in metadata.items():
        training_log.write(f"  {key}: {value}\n")
    training_log.write("\n")

    # Save metadata
    metadata_path = os.path.join(training_dir, f"training_metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Print training parameters
    print("\n" + "=" * 50)
    print("STARTING TRAINING WITH PARAMETERS:")
    print("=" * 50)
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    # Start training with performance monitoring
    print(f"Training agent for {num_episodes} episodes...")
    training_log.write(f"Starting training for {num_episodes} episodes...\n\n")

    # Run training
    agent.train(episodes=num_episodes)

    # Calculate and log total training time
    training_time = time.time() - start_time
    minutes = training_time / 60
    hours = minutes / 60
    time_per_episode = training_time / num_episodes

    # Save final model
    final_path = os.path.join(training_dir, f"final_model_{timestamp}.pth")
    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "total_steps": agent.total_steps,
            "metadata": metadata,
            "training_time_seconds": training_time,
        },
        final_path,
    )

    # Log training statistics
    training_log.write("\n=== TRAINING COMPLETED ===\n")
    training_log.write(
        f"Total training time: {training_time:.1f} seconds ({minutes:.1f} minutes, {hours:.2f} hours)\n"
    )
    training_log.write(f"Average time per episode: {time_per_episode:.2f} seconds\n")
    training_log.write(f"Final epsilon: {agent.epsilon:.4f}\n")
    training_log.write(f"Total steps: {agent.total_steps}\n")
    training_log.write(f"Final model saved to: {final_path}\n")
    training_log.close()

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Total time: {training_time:.1f} sec ({minutes:.1f} min, {hours:.2f} hr)")
    print(f"Time per episode: {time_per_episode:.2f} sec")
    print(f"Final model saved to: {final_path}")
    print(f"Training log saved to: {training_log_path}")
    print("=" * 50)

    return agent


def test_agent(checkpoint="best", num_episodes=1000, render=False, use_gpu=True):
    """
    Test a trained agent for a specified number of episodes with GPU acceleration.

    Args:
        checkpoint: Path to checkpoint file or episode number or "best"
        num_episodes: Number of episodes to test
        render: Whether to render the environment during testing
        use_gpu: Whether to use GPU acceleration
    """
    # Use the same timestamp for logging consistency
    test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup for GPU testing
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the agent from checkpoint
    agent = load_agent(checkpoint, use_gpu=use_gpu)

    # Run testing
    print(f"\nTesting agent for {num_episodes} episodes...")
    print(f"Using device: {agent.device}")

    start_time = time.time()
    average_reward, success_rate = agent.test(episodes=num_episodes, render=render)
    test_time = time.time() - start_time

    # Calculate statistics
    time_per_episode = test_time / num_episodes

    print("\n" + "=" * 40)
    print("TESTING RESULTS")
    print("=" * 40)
    print(f"Total episodes: {num_episodes}")
    print(f"Testing time: {test_time:.1f} seconds ({test_time/60:.1f} minutes)")
    print(f"Time per episode: {time_per_episode:.3f} seconds")
    print(f"Average reward: {average_reward:.4f}")
    print(f"Success rate: {success_rate:.1f}%")
    print("=" * 40)

    # Save test results with timestamp
    result_path = os.path.join(testing_dir, f"test_results_{test_timestamp}.json")

    results = {
        "checkpoint": str(checkpoint),
        "num_episodes": num_episodes,
        "average_reward": float(average_reward),
        "success_rate": float(success_rate),
        "test_time_seconds": float(test_time),
        "time_per_episode": float(time_per_episode),
        "device": str(agent.device),
        "timestamp": test_timestamp,
    }

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Test results saved to: {result_path}")

    return average_reward, success_rate


def evaluate_checkpoints(
    checkpoint_range=(100, 1000, 100), test_episodes=500, use_gpu=True
):
    """
    Evaluate multiple checkpoints to create a learning curve with GPU acceleration.

    Args:
        checkpoint_range: Tuple of (start, end, step) for checkpoint evaluation
        test_episodes: Number of episodes to test each checkpoint
        use_gpu: Whether to use GPU acceleration
    """
    results = {}
    start, end, step = checkpoint_range
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 50)
    print(f"EVALUATING CHECKPOINTS: {start} to {end} (step {step})")
    print(f"Testing {test_episodes} episodes per checkpoint")
    print(f"Using device: {device}")
    print("=" * 50)

    # Test untrained baseline first
    print("\nEvaluating baseline (untrained) agent...")
    baseline_agent = DQNAgent(
        checkpoint_dir=checkpoint_dir
    )  # Create a new agent with random initialization
    start_time = time.time()
    baseline_original_reward, baseline_shaped_reward = baseline_agent.test(
        episodes=test_episodes
    )
    baseline_time = time.time() - start_time
    print(
        f"Baseline original reward: {baseline_original_reward:.4f}, Shaped reward: {baseline_shaped_reward:.4f} (time: {baseline_time:.1f}s)"
    )
    results[0] = {
        "original_reward": baseline_original_reward,
        "shaped_reward": baseline_shaped_reward,
    }

    # Test each checkpoint
    checkpoints_tested = 0
    total_eval_time = baseline_time

    for episode in range(start, end + 1, step):
        checkpoint_path = os.path.join(training_dir, f"model_checkpoint_{episode}.pth")
        print(f"\nEvaluating checkpoint at episode {episode}...")

        # Clean up memory between checkpoints
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Explicitly load checkpoint with weights_only=False
        agent = DQNAgent(env_name="homegrid-task", checkpoint_dir=checkpoint_dir)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epsilon" in checkpoint:
            agent.epsilon = checkpoint["epsilon"]
        if "total_steps" in checkpoint:
            agent.total_steps = checkpoint["total_steps"]
        agent.update_target_network()

        # Run test
        checkpoint_start = time.time()
        original_reward, shaped_reward = agent.test(episodes=test_episodes)
        checkpoint_time = time.time() - checkpoint_start
        total_eval_time += checkpoint_time

        # Store results
        results[episode] = {
            "original_reward": original_reward,
            "shaped_reward": shaped_reward,
        }
        checkpoints_tested += 1

        print(
            f"Checkpoint {episode} original reward: {original_reward:.4f}, Shaped reward: {shaped_reward:.4f} (time: {checkpoint_time:.1f}s)"
        )

    # Also test best model if it exists
    best_path = os.path.join(training_dir, "best_model.pth")
    if os.path.exists(best_path):
        print("\nEvaluating best model...")
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Explicitly load best model with weights_only=False
        best_agent = DQNAgent(env_name="homegrid-task", checkpoint_dir=checkpoint_dir)
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        best_agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            best_agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epsilon" in checkpoint:
            best_agent.epsilon = checkpoint["epsilon"]
        if "total_steps" in checkpoint:
            best_agent.total_steps = checkpoint["total_steps"]
        best_agent.update_target_network()

        # Run test
        best_start = time.time()
        best_original_reward, best_shaped_reward = best_agent.test(
            episodes=test_episodes
        )
        best_time = time.time() - best_start
        total_eval_time += best_time

        results["best"] = {
            "original_reward": best_original_reward,
            "shaped_reward": best_shaped_reward,
        }
        print(
            f"Best model original reward: {best_original_reward:.4f}, Shaped reward: {best_shaped_reward:.4f} (time: {best_time:.1f}s)"
        )

    # Create learning curve plots for both reward and success rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.style.use("ggplot")

    episodes = [e for e in results.keys() if isinstance(e, int)]
    original_rewards = [results[e]["original_reward"] for e in episodes]
    shaped_rewards = [results[e]["shaped_reward"] for e in episodes]

    # Plot rewards
    ax1.plot(
        episodes, original_rewards, "o-", linewidth=2, markersize=8, label="Checkpoints"
    )
    ax1.axhline(
        y=results[0]["original_reward"],
        color="r",
        linestyle="--",
        linewidth=2,
        label="Baseline (Untrained)",
    )

    if "best" in results:
        ax1.axhline(
            y=results["best"]["original_reward"],
            color="g",
            linestyle="--",
            linewidth=2,
            label="Best Model",
        )

    ax1.set_xlabel("Training Episodes", fontsize=14)
    ax1.set_ylabel("Average Test Original Reward", fontsize=14)
    ax1.set_title("Original Reward: Test Performance vs Training Episodes", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot shaped rewards
    ax2.plot(
        episodes,
        shaped_rewards,
        "o-",
        linewidth=2,
        markersize=8,
        color="blue",
        label="Checkpoints",
    )
    ax2.axhline(
        y=results[0]["shaped_reward"],
        color="r",
        linestyle="--",
        linewidth=2,
        label="Baseline (Untrained)",
    )

    if "best" in results:
        ax2.axhline(
            y=results["best"]["shaped_reward"],
            color="g",
            linestyle="--",
            linewidth=2,
            label="Best Model",
        )

    ax2.set_xlabel("Training Episodes", fontsize=14)
    ax2.set_ylabel("Average Test Shaped Reward", fontsize=14)
    ax2.set_title("Shaped Reward: Test Performance vs Training Episodes", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add annotations for key points
    for i, episode in enumerate(episodes):
        if i == 0 or i == len(episodes) - 1 or (i % 3 == 0 and len(episodes) > 6):
            ax1.annotate(
                f"{original_rewards[i]:.3f}",
                (episode, original_rewards[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )
            ax2.annotate(
                f"{shaped_rewards[i]:.3f}",
                (episode, shaped_rewards[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )

    # Save plot with timestamp
    plt_path = os.path.join(testing_dir, f"learning_curve_{timestamp}.png")
    plt.savefig(plt_path, dpi=300, bbox_inches="tight")

    # Also save as PDF for publication quality
    pdf_path = os.path.join(testing_dir, f"learning_curve_{timestamp}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    # Don't call plt.show() in non-interactive environments
    if hasattr(sys, "ps1"):  # Check if running in interactive mode
        plt.show()
    else:
        plt.close(fig)  # Close the figure to free memory

    # Save detailed results to JSON
    results_path = os.path.join(testing_dir, f"checkpoint_evaluation_{timestamp}.json")

    gpu_info = "None"
    if torch.cuda.is_available():
        gpu_info = (
            f"{torch.cuda.get_device_name(0)} - {torch.cuda.device_count()} device(s)"
        )

    # Add metadata to results
    evaluation_metadata = {
        "timestamp": timestamp,
        "checkpoint_range": list(checkpoint_range),
        "test_episodes_per_checkpoint": test_episodes,
        "checkpoints_tested": checkpoints_tested + 1,  # +1 for baseline
        "total_evaluation_time_seconds": total_eval_time,
        "device": str(device),
        "gpu_info": gpu_info,
    }

    # Prepare final results with metadata
    final_results = {
        "metadata": evaluation_metadata,
        "results": {
            str(k): {
                "original_reward": float(v["original_reward"]),
                "shaped_reward": float(v["shaped_reward"]),
            }
            for k, v in results.items()
        },
    }

    # Save results
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED")
    print("=" * 50)
    print(f"Total checkpoints tested: {checkpoints_tested + 1}")
    print(f"Total evaluation time: {total_eval_time:.1f}s ({total_eval_time/60:.1f}m)")
    print(f"Learning curves saved to: {plt_path}")
    print(f"Evaluation results saved to: {results_path}")
    print("=" * 50)

    return results


def benchmark_performance(train_episodes=20, test_episodes=100, use_gpu=True):
    """
    Run a quick benchmark to estimate runtime for larger training and testing.

    Args:
        train_episodes: Small number of episodes to train (default 20)
        test_episodes: Number of episodes to test (default 100)
        use_gpu: Whether to use GPU acceleration (if available)

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Initialize a fresh agent for benchmarking
    benchmark_agent = DQNAgent(checkpoint_dir=checkpoint_dir)
    if use_gpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Warm up the GPU to ensure accurate timing
        dummy_tensor = torch.ones(1000, 1000, device=device)
        dummy_result = torch.matmul(dummy_tensor, dummy_tensor)
        torch.cuda.synchronize()
    else:
        print("Using CPU")

    # Clear any cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}

    # Test single episode timing
    print("\nTiming single episode performance...")
    single_start = time.time()
    benchmark_agent.train(episodes=1)
    single_time = time.time() - single_start
    results["single_episode_time"] = single_time
    print(f"Single training episode time: {single_time:.3f} seconds")

    # Training benchmark
    print(f"\nBenchmarking training for {train_episodes} episodes...")
    train_start = time.time()
    benchmark_agent.train(episodes=train_episodes)
    train_time = time.time() - train_start
    avg_episode_time = train_time / train_episodes
    results["train_time"] = train_time
    results["avg_episode_time"] = avg_episode_time

    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Average time per episode: {avg_episode_time:.3f} seconds")

    # Testing benchmark
    print(f"\nBenchmarking testing for {test_episodes} episodes...")
    test_start = time.time()
    benchmark_agent.test(episodes=test_episodes)
    test_time = time.time() - test_start
    avg_test_time = test_time / test_episodes
    results["test_time"] = test_time
    results["avg_test_time"] = avg_test_time

    print(f"Testing completed in {test_time:.2f} seconds")
    print(f"Average time per test episode: {avg_test_time:.3f} seconds")

    # Extrapolation estimates
    print("\nExtrapolated runtime estimates:")
    for ep_count in [100, 500, 1000, 2000, 5000]:
        est_time = avg_episode_time * ep_count
        minutes = est_time / 60
        hours = minutes / 60

        print(
            f"  • {ep_count} episodes: {est_time:.1f} sec ({minutes:.1f} min, {hours:.2f} hr)"
        )

    # Test episodes extrapolation
    print("\nExtrapolated test runtime estimates:")
    for test_count in [500, 1000, 5000, 10000]:
        est_time = avg_test_time * test_count
        minutes = est_time / 60

        print(f"  • {test_count} test episodes: {est_time:.1f} sec ({minutes:.1f} min)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_file = os.path.join(testing_dir, f"benchmark_results_{timestamp}.json")

    # Include hardware info
    hw_info = {
        "device": str(device),
        "cpu_count": multiprocessing.cpu_count(),
        "timestamp": timestamp,
    }

    if torch.cuda.is_available():
        hw_info["gpu"] = torch.cuda.get_device_name(0)
        hw_info["cuda_version"] = torch.version.cuda
        hw_info["gpu_memory_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )

    results["hardware_info"] = hw_info

    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nBenchmark results saved to:", benchmark_file)
    print("=" * 60)

    return results


# ===========================================
# GLOBAL CONFIGURATION - Easy to modify
# ===========================================

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Limit num threads
torch.set_num_threads(2)
# Limit GPU memory usage to approximately 45% of allocation
if torch.cuda.is_available():
    # Get total GPU memory
    total_mem = torch.cuda.get_device_properties(0).total_memory
    # Set to use only 45% of available memory
    torch.cuda.set_per_process_memory_fraction(0.45)
    print(f"Limiting GPU memory usage to 45% of {total_mem/1e9:.2f} GB")

# GPU optimization parameters
num_workers = min(
    8, multiprocessing.cpu_count()
)  # Use up to 8 worker threads for data loading

# Generate timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


checkpoint_dir = "checkpoints40"  ### CHANGE THIS TO THE CORRECT CHECKPOINT DIRECTORY


os.makedirs(checkpoint_dir, exist_ok=True)
training_dir = os.path.join(checkpoint_dir, "training")
testing_dir = os.path.join(checkpoint_dir, "testing")
os.makedirs(training_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# Set up logging to capture terminal output
log_file = os.path.join(testing_dir, f"terminal_output_{timestamp}.txt")
sys.stdout = Logger(log_file)


if __name__ == "__main__":
    # Print GPU information if available
    if torch.cuda.is_available():
        print("\nGPU INFORMATION:")
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "num_gpus": torch.cuda.device_count(),
        }

        print(f"  Device: {gpu_info['device']}")
        print(f"  CUDA version: {gpu_info['cuda_version']}")
        print(f"  Total memory: {gpu_info['memory_gb']}")
        print(f"  Number of GPUs: {gpu_info['num_gpus']}")
    else:
        gpu_info = "None"
        print("\nNo GPU available - will use CPU for training and evaluation")

    # Current configuration (uncomment the desired option)

    """
    # OPTION 1: Train a new agent from scratch with GPU acceleration
    # Highly recommended for sparse reward environments
    train_agent(
        num_episodes=2000,  # 2K episodes to train (more for sparse rewards)
        save_interval=100,  # Save model every 100 episodes
        use_gpu=True,  # Use GPU acceleration if available
    )
    
    test_agent(
        checkpoint="best",  # Test the best model saved during training
        num_episodes=10000,  # 10K test episodes
        use_gpu=True,  # Use GPU for faster testing
    )
    """

    # OPTION 2: Continue training from a previous checkpoint
    # Uncomment to use:
    """
    train_agent(
        num_episodes=1000,          # Additional episodes to train
        continue_from="best",       # Continue from the best model so far
        save_interval=100,          # Save checkpoints every 100 episodes
        use_gpu=True                # Use GPU acceleration
    )
    
    test_agent(
        checkpoint="final",  # Test the final model after continued training
        num_episodes=100,  # Test on 1000 episodes
        use_gpu=True,  # Use GPU for faster testing
    )
    """

    # OPTION 3: Test code quickly to check it doesnt break
    # Uncomment to use:

    # train_agent(
    #     num_episodes=10,
    #     save_interval=5,
    #     use_gpu=True,
    # )

    # evaluate_checkpoints(
    #     checkpoint_range=(
    #         5,
    #         10,
    #         5,
    #     ),
    #     test_episodes=50,
    #     use_gpu=True,
    # )

    # train_agent(
    #     num_episodes=10,
    #     continue_from=10,
    #     save_interval=5,
    #     use_gpu=True,
    # )

    # evaluate_checkpoints(
    #     checkpoint_range=(
    #         10,
    #         20,
    #         5,
    #     ),
    #     test_episodes=100,
    #     use_gpu=True,
    # )

    # OPTION 4: Evaluate multiple checkpoints to create a learning curve
    # Uncomment to use:

    train_agent(
        num_episodes=3000,
        save_interval=100,
        use_gpu=True,
    )

    evaluate_checkpoints(
        checkpoint_range=(
            200,
            3000,
            200,
        ),
        test_episodes=100,
        use_gpu=True,
    )

    # test_agent(checkpoint=6000, num_episodes=1000, use_gpu=True)

    # train_agent(
    #     num_episodes=14000,
    #     continue_from=6000,
    #     save_interval=500,
    #     use_gpu=True,
    # )

    # evaluate_checkpoints(
    #     checkpoint_range=(
    #         3000,
    #         18000,
    #         3000,
    #     ),
    #     test_episodes=10000,
    #     use_gpu=True,
    # )

    # OPTION 5: Quick test of a specific checkpoint
    # Uncomment to use:
    """
    test_agent(
        checkpoint=1000,            # Test the model saved at episode 1000
        num_episodes=100,           # Run 100 test episodes
        render=False,               # Don't render the environment
        use_gpu=True                # Use GPU acceleration
    )
    """

    # OPTION 6: Run a quick benchmark to estimate runtime for larger training
    # Uncomment to use:

    # benchmark_performance(
    #     train_episodes=20,  # Quick training benchmark with 20 episodes
    #     test_episodes=100,  # Test benchmark with 100 episodes
    #     use_gpu=True,  # Use GPU if available
    # )

    # ===========================================
    # OPTION 7: Self-contained hyperparam sweep
    # ===========================================

    # # Sweep settings
    # TRAIN_EPISODES = 4_000  # ~50 min of training
    # TEST_EPISODES = 100  # ~3 min of testing
    # LRS = [5e-4, 1e-3]
    # EDS = [0.9995, 0.99975, 0.9999]
    # FIXED_BATCH = 64
    # FIXED_REPLAY_BUFFER_SIZE = 100_000
    # FIXED_USE_PER = False

    # print("\n=== STARTING HYPERPARAMETER SWEEP ===")
    # for lr, ed in product(LRS, EDS):
    #     run_tag = f"lr{lr}_ed{ed}"
    #     run_ckpt = os.path.join(checkpoint_dir, run_tag)
    #     os.makedirs(run_ckpt, exist_ok=True)

    #     print(f"\n--- RUN {run_tag} ---")
    #     # 1) create agent with fresh checkpoint dir
    #     agent = DQNAgent(
    #         env_name="homegrid-task", episodes=TRAIN_EPISODES, checkpoint_dir=run_ckpt
    #     )
    #     # 2) set hyperparameters
    #     agent.alpha = lr
    #     agent.epsilon_decay = ed
    #     agent.batch_size = FIXED_BATCH
    #     agent.max_replay_buffer_size = FIXED_REPLAY_BUFFER_SIZE
    #     agent.use_per = FIXED_USE_PER

    #     # 3) train
    #     print(f"Training for {TRAIN_EPISODES} episodes (α={lr}, ε-decay={ed})")
    #     agent.train(episodes=TRAIN_EPISODES)

    #     # 4) test
    #     print(f"Testing for {TEST_EPISODES} episodes")
    #     avg_reward, avg_shaped = agent.test(episodes=TEST_EPISODES)

    #     print(
    #         f"RESULT {run_tag} → avg orig reward: {avg_reward:.3f}, avg shaped: {avg_shaped:.3f}"
    #     )

    # print("\n=== SWEEP COMPLETE ===")

    # Close the logger at the end of execution
    if isinstance(sys.stdout, Logger):
        # Close the log file
        sys.stdout.log.close()
        # Restore original stdout
        sys.stdout = sys.stdout.terminal
        print(f"Log file saved to: {log_file}")
