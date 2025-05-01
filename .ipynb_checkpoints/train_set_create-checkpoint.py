#!/usr/bin/env python3

# To submit this I will call submit_trainset.sh

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
from matplotlib.patches import Polygon
import csv
import pandas as pd

# --- CONFIGURATION ---
ROOT_DIR = "/home/adm808/Sabatini_Lab/behavior_samples"
OUTPUT_DIR = "/n/scratch/users/a/adm808/Sabatini_Lab/ZoneDetection2-adi-2025-04-23/labeled-data/dummy_video/"
PERCENTILE = 95  # Percentile for interesting frames
WINDOW_SIZE = 0  # Can update if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CSV LOGGING ---
def init_csv(output_dir):
    csv_path = os.path.join(output_dir, "zone_frame_log.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["video", "frame_number", "num_zones", "num_points", "image_path"])
    return csv_path

def log_to_csv(csv_path, video, frame_number, num_zones, num_points, image_path):
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([video, frame_number, num_zones, num_points, image_path])

def init_zone_csv(output_dir):
    zone_csv_path = os.path.join(output_dir, "zone_labels_for_dlc.csv")
    with open(zone_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["video", "image_path", "zone_name", "x_coords", "y_coords"])
    return zone_csv_path

def log_zone_to_csv(csv_path, video, image_path, zone_name, x_coords, y_coords):
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([video, image_path, zone_name, x_coords if x_coords else "null", y_coords if y_coords else "null"])

# --- FUNCTIONS ---
def find_all_videos_and_zones(root):
    video_zone_pairs = []
    for subdir, _, files in os.walk(root):
        for file in files:
            if file.endswith(".AVI"):
                video_path = os.path.join(subdir, file)
                zone_name = file.replace(".AVI", "_zones.mat")
                zone_path = os.path.join(subdir, zone_name)
                if os.path.exists(zone_path):
                    video_zone_pairs.append((video_path, zone_path))
    return video_zone_pairs

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames

def find_interesting_frames_streaming(video_path, percentile=90):
    cap = cv2.VideoCapture(video_path)
    diffs = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 1

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, prev_gray)
        diff_sum = np.sum(diff)
        diffs.append(diff_sum)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    threshold = np.percentile(diffs, percentile)
    interesting = [i + 1 for i, d in enumerate(diffs) if d > threshold]
    return interesting

def get_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Error reading frame {frame_number}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def load_mat_file(mat_path):
    mat_data = scipy.io.loadmat(mat_path)
    return mat_data["zones"]

def overlay_zones(frame, zones, alpha=0.05, circle_radius=3):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(frame)
    height, width = frame.shape[:2]

    zone_coords_list = []

    for i in range(zones.shape[1]):
        zone = zones[0, i]
        isin = zone["isin"].flatten()
        active = np.where(isin == 1)[0]

        if len(active) == 0:
            zone_coords_list.append((f"Zone{i+1}", [], []))
            continue

        y_coords = zone["dim2points"].flatten()[active] - 1
        x_coords = zone["dim1points"].flatten()[active] - 1

        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)

        if len(x_coords) >= 3:
            polygon = Polygon(
                np.column_stack((x_coords, y_coords)),
                closed=True,
                edgecolor='white',
                facecolor='none',
                linewidth=1.0,
                alpha=0.8
            )
            ax.add_patch(polygon)

        zone_coords_list.append((f"Zone{i+1}", x_coords.tolist(), y_coords.tolist()))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    total_zones = zones.shape[1]
    total_points = sum(len(coords[1]) for coords in zone_coords_list)

    return fig, total_zones, total_points, zone_coords_list

def process_video(video_path, zone_path, frame_indices, output_dir, csv_log_path, zone_csv_path):
    print(f"\nProcessing {video_path}")
    zones = load_mat_file(zone_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    for frame_num in frame_indices:
        try:
            frame = get_frame(cap, frame_num)
        except Exception as e:
            print(f"Skipping frame {frame_num} due to error: {e}")
            continue

        base = os.path.basename(video_path).replace(".AVI", "")
        raw_path = os.path.join(output_dir, f"{base}_frame_{frame_num}_raw.png")
        plt.imsave(raw_path, frame)

        fig, num_zones, num_points, zone_coords_list = overlay_zones(frame, zones)
        output_path = os.path.join(output_dir, f"{base}_frame_{frame_num}.png")
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

        log_to_csv(csv_log_path, base, frame_num, num_zones, num_points, output_path)

        for zone_name, x_coords, y_coords in zone_coords_list:
            log_zone_to_csv(zone_csv_path, base, output_path, zone_name, x_coords, y_coords)

    cap.release()

def generate_dlc_csv(zone_csv_path, output_path):
    df = pd.read_csv(zone_csv_path)

    image_data = {}
    for _, row in df.iterrows():
        image = row['image_path']
        zone = row['zone_name']
        x_coords = eval(row['x_coords']) if row['x_coords'] != 'null' else []
        y_coords = eval(row['y_coords']) if row['y_coords'] != 'null' else []

        if image not in image_data:
            image_data[image] = {"image": image}

        if x_coords and y_coords:
            x_avg = sum(x_coords) / len(x_coords)
            y_avg = sum(y_coords) / len(y_coords)
        else:
            x_avg = float('nan')
            y_avg = float('nan')

        image_data[image][f"{zone}_x"] = x_avg
        image_data[image][f"{zone}_y"] = y_avg

    dlc_df = pd.DataFrame(image_data.values())
    dlc_csv_path = os.path.join(output_path, "CollectedData_Auto.csv")
    dlc_df.to_csv(dlc_csv_path, index=False)
    print(f"\n DeepLabCut-compatible CSV saved to: {dlc_csv_path}")

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    MAX_VIDEOS = 3
    FRAMES_PER_VIDEO = 100

    pairs = find_all_videos_and_zones(ROOT_DIR)
    print(f"Found {len(pairs)} total video-zone pairs.")

    sampled_pairs = random.sample(pairs, min(MAX_VIDEOS, len(pairs)))
    print(f"Randomly selected {len(sampled_pairs)} video-zone pairs for processing.")

    csv_log_path = init_csv(OUTPUT_DIR)
    zone_csv_path = init_zone_csv(OUTPUT_DIR)

    for video_path, zone_path in sampled_pairs:
        interesting = find_interesting_frames_streaming(video_path, percentile=PERCENTILE)
        if not interesting:
            print(f"Skipping {video_path} — not enough interesting frames.")
            continue

        selected_frames = random.sample(interesting, min(FRAMES_PER_VIDEO, len(interesting)))
        process_video(video_path, zone_path, selected_frames, OUTPUT_DIR, csv_log_path, zone_csv_path)

    generate_dlc_csv(zone_csv_path, OUTPUT_DIR)

