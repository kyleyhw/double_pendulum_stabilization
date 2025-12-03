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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    try:
        df = pd.read_csv(latest_file)
        
        # Primary Axis: Reward
        ax1 = plt.gca()
        ax1.plot(df['Episode'], df['Reward'], label='Reward', color='blue', alpha=0.3)
        
        # Calculate rolling average
        if len(df) > 20:
            df['rolling_reward'] = df['Reward'].rolling(window=20).mean()
            ax1.plot(df['Episode'], df['rolling_reward'], label='Moving Avg (20 eps)', color='blue', linewidth=2)
            
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward (Log Scale)', color='blue')
        ax1.set_yscale('symlog')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Secondary Axis: Difficulty
        if 'Difficulty' in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df['Episode'], df['Difficulty'], label='Difficulty', color='red', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Difficulty (Curriculum)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 1.05)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.title(f"Training Learning Curve (Generated: {timestamp})")
        
        plt.grid(True, alpha=0.3)
        
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
