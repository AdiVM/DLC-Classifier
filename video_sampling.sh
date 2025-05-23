#!/bin/bash

# Activate environment just to be safe
source /n/groups/patel/adithya/scenv/bin/activate

# Set base paths
SRC_DIR="/n/files/Neurobio/MICROSCOPE/Kim/KER Behavior/By date/Low speed"
# Be sure to update to this for scratch if you run later
DEST_DIR="/n/scratch/users/a/adm808/Sabatini_Lab/behavior_samples"

# Make destination directory
mkdir -p "$DEST_DIR"

# CSV log path
CSV_LOG="$DEST_DIR/video_copy_log.csv"
echo "video_filename,source_path,destination_path" > "$CSV_LOG"  ### CSV HEADER

# Find all .avi, .AVI, or .TS files across all subfolders and sample 500 randomly
find "$SRC_DIR" -type f \( -iname "*.avi" -o -iname "*.ts" \) | shuf -n 500 > sampled_videos.txt

# Loop through each sampled file and copy it + its matching _zones.mat file
while read -r video_path; do
    echo "Copying: $video_path"

    # Get full filename without extension
    video_filename=$(basename "$video_path")
    video_stem="${video_filename%.*}"

    # Define the _zones.mat file (same folder)
    zones_file="$(dirname "$video_path")/${video_stem}_zones.mat"

    # Make matching subfolder in destination
    rel_path="${video_path#$SRC_DIR/}"  # get relative path from base
    subfolder=$(dirname "$rel_path")
    dest_path="$DEST_DIR/$subfolder"
    mkdir -p "$dest_path"

    # Copy video file
    rsync -av "$video_path" "$dest_path/"

    # Log to CSV
    echo "$video_filename,$video_path,$dest_path/$video_filename" >> "$CSV_LOG"  ### CSV LOG ENTRY

    # Copy .mat file if it exists
    if [[ -f "$zones_file" ]]; then
        echo "Copying associated zones file: $zones_file"
        rsync -av "$zones_file" "$dest_path/"
    else
        echo "Zones file not found for: $video_filename"
    fi
done < sampled_videos.txt

echo "Finished copying 100 random videos and their associated zone files."
echo "Log saved to $CSV_LOG"