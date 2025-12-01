import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob

def plot_learning_curve(log_dir="logs", output_file="docs/images/learning_curve.png"):
    # Find the latest log file
    list_of_files = glob.glob(os.path.join(log_dir, 'training_log_*.csv')) 
    if not list_of_files:
        print("No log files found!")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Plotting from {latest_file}")
    
    # Read CSV
    df = pd.read_csv(latest_file)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Raw Reward
    plt.plot(df['episode'], df['reward'], label='Episode Reward', alpha=0.3, color='blue')
    
    # Moving Average
    window_size = 50
    rolling_mean = df['reward'].rolling(window=window_size).mean()
    plt.plot(df['episode'], rolling_mean, label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress: Double Pendulum Stabilization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output", type=str, default="docs/images/learning_curve.png")
    args = parser.parse_args()
    
    plot_learning_curve(args.log_dir, args.output)
