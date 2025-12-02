import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob

def plot_learning_curve(log_dir="logs", output="docs/images/learning_curve.png"):
    # Find latest CSV
    list_of_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not list_of_files:
        print("No log files found.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Plotting from {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['reward'], label='Reward')
        
        # Calculate rolling average
        if len(df) > 20:
            df['rolling_reward'] = df['reward'].rolling(window=20).mean()
            plt.plot(df['episode'], df['rolling_reward'], label='Avg Reward (20)', color='orange')
            
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Learning Curve')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output)
        print(f"Saved plot to {output}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output", type=str, default="docs/images/learning_curve.png")
    args = parser.parse_args()
    
    plot_learning_curve(args.log_dir, args.output)
