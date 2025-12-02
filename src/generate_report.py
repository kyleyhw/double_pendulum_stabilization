import argparse
import subprocess
import os
import sys
from datetime import datetime
import glob

import random

def generate_report(log_dir="logs", seed=None):
    """
    Centralized script to generate all visualization artifacts for a report.
    Ensures consistent seeding and timestamping across all artifacts.
    """
    if seed is None:
        seed = random.randint(0, 100000)
        
    print(f"--- Generating Report (Seed: {seed}) ---")
    
    # 1. Generate Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp: {timestamp}")
    
    # Ensure docs/images exists
    output_dir = "docs/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Find Latest Run ID (for Final Run)
    # We need to find the final model path to pass to simulate.py?
    # Actually, simulate.py takes --model.
    # visualize_overlay.py finds the latest run automatically.
    # We should probably let visualize_overlay do its thing, but for final_run we need the model.
    
    # Let's find the latest final model here to be explicit.
    all_checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_final.pth"))
    if not all_checkpoints:
        print("No final checkpoints found. Trying any checkpoint...")
        all_checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_*.pth"))
        
    if not all_checkpoints:
        print("No checkpoints found in logs.")
        return

    latest_model = max(all_checkpoints, key=os.path.getctime)
    print(f"Latest Model: {latest_model}")
    
    # Extract Episode Label
    try:
        ep_label = os.path.basename(latest_model).split('_')[-1].split('.')[0]
        if ep_label == "final": ep_label = "Final"
    except:
        ep_label = "Final"
        
    # Determine Difficulty
    # We need to find the run ID from the model filename to find the log
    # Model: logs\ppo_20251202_024235_final.pth -> Run ID: 20251202_024235
    final_diff = 1.0
    try:
        basename = os.path.basename(latest_model)
        parts = basename.split('_')
        if len(parts) >= 3:
            run_id = f"{parts[1]}_{parts[2]}"
            log_file = os.path.join(log_dir, f"training_log_{run_id}.csv")
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    header = f.readline().strip().split(',')
                    try:
                        diff_idx = header.index("Difficulty")
                        # Read last line
                        last_line = None
                        for line in f:
                            if line.strip(): last_line = line
                        
                        if last_line:
                            parts = last_line.strip().split(',')
                            if len(parts) > diff_idx:
                                final_diff = float(parts[diff_idx])
                                print(f"Found Final Difficulty: {final_diff}")
                    except ValueError:
                        print("Could not find Difficulty column in log.")
    except Exception as e:
        print(f"Error extracting difficulty: {e}")

    # 3. Define Output Filenames
    overlay_mp4 = os.path.join(output_dir, f"overlay_montage_{timestamp}.mp4")
    final_run_mp4 = os.path.join(output_dir, f"final_run_{timestamp}.mp4")
    # Use static filename for learning curve to allow consistent README linking
    learning_curve_png = os.path.join(output_dir, "learning_curve.png")
    
    # 4. Run Visualize Overlay
    print("\n[1/3] Running Overlay Visualization...")
    cmd_overlay = [
        sys.executable, "src/visualize_overlay.py",
        "--log_dir", log_dir,
        "--output_mp4", overlay_mp4,
        "--seed", str(seed),
        "--save_mp4"
    ]
    subprocess.run(cmd_overlay, check=True)
    
    # 5. Run Final Run Simulation
    print("\n[2/3] Running Final Run Simulation...")
    cmd_final = [
        sys.executable, "src/simulate.py",
        "--model", latest_model,
        "--output", final_run_mp4,
        "--seed", str(seed),
        "--save_mp4",
        "--headless",
        "--duration", "20.0",
        "--difficulty", str(final_diff)
    ]
    subprocess.run(cmd_final, check=True)
    
    # 6. Run Learning Curve
    print("\n[3/3] Generating Learning Curve...")
    cmd_curve = [
        sys.executable, "src/plot_learning_curve.py",
        "--log_dir", log_dir,
        "--output", learning_curve_png
    ]
    subprocess.run(cmd_curve, check=True)
    
    # 7. Generate Side-by-Side Comparison
    print("\n[4/4] Generating Side-by-Side Comparison...")
    comparison_mp4 = os.path.join(output_dir, f"comparison_{timestamp}.mp4")
    create_comparison_video(overlay_mp4, final_run_mp4, comparison_mp4)
    
    print("\n--- Report Generation Complete ---")
    print(f"Overlay: {overlay_mp4}")
    print(f"Final Run: {final_run_mp4}")
    print(f"Comparison: {comparison_mp4}")
    print(f"Learning Curve: {learning_curve_png}")

def create_comparison_video(vid1_path, vid2_path, output_path):
    import cv2
    import numpy as np
    
    cap1 = cv2.VideoCapture(vid1_path)
    cap2 = cv2.VideoCapture(vid2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening video files for comparison.")
        return
        
    # Get properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Output is double width
    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        # Resize if necessary (should be same if generated by same pipeline)
        if frame2.shape != frame1.shape:
            frame2 = cv2.resize(frame2, (width, height))
            
        # Add labels
        cv2.putText(frame1, "Overlay Montage", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame2, "Final Run", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Stack
        combined = np.hstack((frame1, frame2))
        writer.write(combined)
        
    cap1.release()
    cap2.release()
    writer.release()
    print(f"Comparison video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()
    
    generate_report(args.log_dir, args.seed)
