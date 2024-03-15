#!/bin/bash

# Get the directory where the script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define GitHub repository URL, TSV file path, and relative download directory
repo_url="https://github.com/dmlls/cannot-dataset"
tsv_file_path="cannot-dataset/cannot_dataset_v1.1.tsv"
relative_download_dir="../../data/datasets/cannot"

# Ensure the relative download directory exists within the script directory
if [ ! -d "$script_dir/$relative_download_dir" ]; then
    echo "Relative download directory does not exist. Creating it..."
    mkdir -p "$script_dir/$relative_download_dir"
fi

# Construct the raw GitHub content URL for the TSV file
raw_url="https://raw.githubusercontent.com/$(echo "$repo_url" | cut -d'/' -f4-)/main/$tsv_file_path"

# Download the TSV file
echo "Downloading TSV file from $raw_url to $script_dir/$relative_download_dir"
curl -L "$raw_url" -o "$script_dir/$relative_download_dir/$(basename $tsv_file_path)"

echo "Download completed."
