# To submit this I will call submit_trainset.sh

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
from matplotlib.patches import Polygon
from matplotlib.path import Path
import csv
import pandas as pd

import sys


# --- CONFIGURATION ---
ROOT_DIR = "/n/scratch/users/a/adm808/Sabatini_Lab/behavior_samples"
OUTPUT_DIR = "/n/scratch/users/a/adm808/Sabatini_Lab/ZoneDetection2-adi-2025-04-23/labeled-data/dummy_video/"
PERCENTILE = 95  # Percentile for interesting frames
WINDOW_SIZE = 0  # Can update if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

logfile = os.path.join(OUTPUT_DIR, "debug_output.log")
sys.stdout = open(logfile, "w")
sys.stderr = sys.stdout

# Creating a file to store names for top zones
top_frame_csv_path = os.path.join(OUTPUT_DIR, "top_zone_frames_summary.csv")
with open(top_frame_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["video", "zone", "frame_index", "diff_sum"])


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

def find_top_frames_per_zone(zones, gray_frames, num_frames=4):
    zone_frame_map = {}

    height, width = gray_frames[0].shape

    for i in range(zones.shape[1]):
        zone = zones[0, i]
        isin = zone["isin"].flatten()
        active = np.where(isin == 1)[0]

        if len(active) == 0:
            continue

        y_coords = zone["dim2points"].flatten()[active] - 1
        x_coords = zone["dim1points"].flatten()[active] - 1
        x_coords = np.maximum(x_coords, 0)
        y_coords = np.clip(y_coords, 0, height - 1)

        if len(x_coords) >= 3:
            mask = get_zone_mask((height, width), x_coords, y_coords)
            diffs = []

            for idx in range(1, len(gray_frames)):
                diff = cv2.absdiff(gray_frames[idx], gray_frames[idx - 1])
                masked_diff = diff[mask]
                diff_sum = np.sum(masked_diff)
                diffs.append((idx, diff_sum))  # (frame index, diff score)

            # Sort by diff_sum descending, take top N
            top_frames = sorted(diffs, key=lambda x: x[1], reverse=True)[:num_frames]
            zone_frame_map[f"Zone{i+1}"] = top_frames

    return zone_frame_map
def get_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Error reading frame {frame_number}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def load_mat_file(mat_path):
    mat_data = scipy.io.loadmat(mat_path)
    return mat_data["zones"]

def get_zone_mask(shape, x_coords, y_coords):
    """Return a binary mask of the polygon defined by x_coords, y_coords."""
    height, width = shape
    poly_path = Path(np.column_stack((x_coords, y_coords)))
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    mask = poly_path.contains_points(points).reshape((height, width))
    return mask

def overlay_zones(frame, prev_gray, zones, zone_thresholds, alpha=0.05):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(frame)
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray, prev_gray)

    zone_coords_list = []
    zone_points = {}

    for i in range(zones.shape[1]):
        zone = zones[0, i]
        isin = zone["isin"].flatten()
        active = np.where(isin == 1)[0]

        if len(active) == 0:
            zone_coords_list.append((f"Zone{i+1}", [], []))
            zone_points[f"Zone{i+1}"] = (np.nan, np.nan)
            continue

        y_coords = zone["dim2points"].flatten()[active] - 1
        x_coords = zone["dim1points"].flatten()[active] - 1
        x_coords = np.maximum(x_coords, 0)
        y_coords = np.clip(y_coords, 0, height - 1)

        if len(x_coords) >= 3:
            mask = get_zone_mask((height, width), x_coords, y_coords)
            masked_diff = diff[mask]
            diff_sum = np.sum(masked_diff)

            # Check movement threshold
            if diff_sum < zone_thresholds[f"Zone{i+1}"]:
                zone_points[f"Zone{i+1}"] = (np.nan, np.nan)
                zone_coords_list.append((f"Zone{i+1}", [], []))
                continue

            # If active, draw polygon
            polygon = Polygon(
                np.column_stack((x_coords, y_coords)),
                closed=True,
                edgecolor='white',
                facecolor='none',
                linewidth=1.0,
                alpha=0.8
            )
            ax.add_patch(polygon)

            # Save center point
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
            zone_points[f"Zone{i+1}"] = (x_center, y_center)

        else:
            zone_points[f"Zone{i+1}"] = (np.nan, np.nan)

        zone_coords_list.append((f"Zone{i+1}", x_coords.tolist(), y_coords.tolist()))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    return fig, zone_coords_list, zone_points

def process_video(video_path, zone_path, frame_indices, output_dir, csv_log_path, zone_csv_path, top_frame_csv_path):
    print(f"\nProcessing {video_path}")
    zones = load_mat_file(zone_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Precompute per-zone intensity stats across all frames
    print("Computing zone-specific intensity baselines to set thresholds")

    # Load all frames once
    all_frames = read_video_frames(video_path)
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in all_frames]

    zone_frame_map = find_top_frames_per_zone(zones, gray_frames, num_frames=4)

    # Collect top frames per zone
    zone_selected_frames = set()
    for top_frames in zone_frame_map.values():
        for frame_idx, _ in top_frames:
            zone_selected_frames.add(frame_idx)

    # Calculate how many remaining slots are available
    remaining_slots = FRAMES_PER_VIDEO - len(zone_selected_frames)

    # Pick from global interesting frames to fill
    remaining_global_frames = list(set(frame_indices) - zone_selected_frames)
    if remaining_slots > 0 and remaining_global_frames:
        additional_frames = random.sample(remaining_global_frames, min(remaining_slots, len(remaining_global_frames)))
    else:
        additional_frames = []

    # Final combined list
    final_selected_frames = sorted(set(zone_selected_frames).union(additional_frames))

    # Write this information to the CSV defined above
    with open(top_frame_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for zone_name, top_frames in zone_frame_map.items():
            for frame_idx, diff_sum in top_frames:
                writer.writerow([os.path.basename(video_path), zone_name, frame_idx, diff_sum])

    zone_means = []
    zone_stds = []

    height, width = gray_frames[0].shape
    # Calculate per-zone movement thresholds
    zone_thresholds = {}
    for i in range(zones.shape[1]):
        zone = zones[0, i]
        isin = zone["isin"].flatten()
        active = np.where(isin == 1)[0]
        if len(active) == 0:
            zone_thresholds[f"Zone{i+1}"] = np.inf
            continue

        # Get coordinates of the zone
        
        y_coords = zone["dim2points"].flatten()[active] - 1
        x_coords = zone["dim1points"].flatten()[active] - 1
        x_coords = np.maximum(x_coords, 0)
        y_coords = np.clip(y_coords, 0, height - 1)
        # and calculate the mask
        if len(x_coords) >= 3:
            mask = get_zone_mask((height, width), x_coords, y_coords)
            diffs = []
            for idx in range(1, len(gray_frames)):
                diff = cv2.absdiff(gray_frames[idx], gray_frames[idx - 1])
                masked_diff = diff[mask]
                diff_sum = np.sum(masked_diff)
                diffs.append(diff_sum)
            threshold = np.percentile(diffs, PERCENTILE)
            zone_thresholds[f"Zone{i+1}"] = threshold
        else:
            zone_thresholds[f"Zone{i+1}"] = np.inf

    dlc_rows = []
    # For each selected frame use its immediate previous frame

    for frame_num in final_selected_frames:
        try:
            frame = get_frame(cap, frame_num)
        except Exception as e:
            print(f"Skipping frame {frame_num} due to error: {e}")
            continue

        if frame_num == 0:
            prev_gray_frame = gray_frames[0]
        else:
            prev_gray_frame = gray_frames[frame_num - 1]

        fig, zone_coords_list, zone_points = overlay_zones(frame, prev_gray_frame, zones, zone_thresholds)

        # Check if at least one zone is active
        has_active_zone = any(not np.isnan(x) and not np.isnan(y) for x, y in zone_points.values())
        if not has_active_zone:
            print(f"Skipping frame {frame_num} — no active zones detected")
            continue
        
        # Saving the raw image
        base = os.path.basename(video_path).replace(".AVI", "")
        raw_path = os.path.join(output_dir, f"{base}_frame_{frame_num}_raw.png")
        plt.imsave(raw_path, frame)

        output_path = os.path.join(output_dir, f"{base}_frame_{frame_num}.png")
        # Ensuring left most LEDs do not get cutoff
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        fig.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.1)

        # Saving which frames were processed and saved (from whcih video they came, which frame, and how many zones they had for debugging purposes)
        active_zones = sum([~np.isnan(x) and ~np.isnan(y) for x, y in zone_points.values()])
        total_points = sum([len(x) for _, x, _ in zone_coords_list])
        log_to_csv(csv_log_path, base, frame_num, active_zones, total_points, output_path)

        row = {'video': base, 'image_path': output_path}
        for zone_name, (x, y) in zone_points.items():
            row[f'{zone_name}_x'] = x
            row[f'{zone_name}_y'] = y
        dlc_rows.append(row)

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cap.release()

    # Save DLC CSV
    dlc_df = pd.DataFrame(dlc_rows)
    csv_path = os.path.join(output_dir, "CollectedData_Auto.csv")

    # When the  file exists, append without writing header
    if os.path.exists(csv_path):
        dlc_df.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        dlc_df.to_csv(csv_path, mode='w', index=False, header=True)
    print(f"\nDeepLabCut-compatible CSV saved to: {os.path.join(output_dir, 'CollectedData_Auto.csv')}")

# def generate_dlc_csv(zone_csv_path, output_path):
#     df = pd.read_csv(zone_csv_path)

#     image_data = {}
#     for _, row in df.iterrows():
#         image = row['image_path']
#         zone = row['zone_name']
#         x_coords = eval(row['x_coords']) if row['x_coords'] != 'null' else []
#         y_coords = eval(row['y_coords']) if row['y_coords'] != 'null' else []

#         if image not in image_data:
#             image_data[image] = {"image": image}

#         if x_coords and y_coords:
#             x_avg = sum(x_coords) / len(x_coords)
#             y_avg = sum(y_coords) / len(y_coords)
#         else:
#             x_avg = float('nan')
#             y_avg = float('nan')

#         image_data[image][f"{zone}_x"] = x_avg
#         image_data[image][f"{zone}_y"] = y_avg

#     dlc_df = pd.DataFrame(image_data.values())
#     dlc_csv_path = os.path.join(output_path, "CollectedData_Auto.csv")
#     dlc_df.to_csv(dlc_csv_path, index=False)
#     print(f"\n DeepLabCut-compatible CSV saved to: {dlc_csv_path}")

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    MAX_VIDEOS = 15
    FRAMES_PER_VIDEO = 40

    pairs = find_all_videos_and_zones(ROOT_DIR)
    print(f"Found {len(pairs)} total video-zone pairs.")

    sampled_pairs = random.sample(pairs, min(MAX_VIDEOS, len(pairs)))
    print(f"Randomly selected {len(sampled_pairs)} video-zone pairs for processing.")

    csv_log_path = init_csv(OUTPUT_DIR)
    zone_csv_path = init_zone_csv(OUTPUT_DIR)

    for video_path, zone_path in sampled_pairs:
        interesting = find_interesting_frames_streaming(video_path, percentile=PERCENTILE)
        process_video(video_path, zone_path, interesting, OUTPUT_DIR, csv_log_path, zone_csv_path, top_frame_csv_path)

    #generate_dlc_csv(zone_csv_path, OUTPUT_DIR)

