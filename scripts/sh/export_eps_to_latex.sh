#!/bin/bash

# Define source and target directories
source_dir="output/"
target_dir="/Users/vincent/Library/CloudStorage/Dropbox/Study/AI/Thesis/Writing/Thesis/images/"

# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist. Please check the path."
    exit 1
fi

# Check if the target directory exists
if [ ! -d "$target_dir" ]; then
    echo "Target directory does not exist. Please check the path."
    exit 1
fi

# Copy all .eps files from source to target directory
find "$source_dir" -type f -name "*.eps" -exec cp {} "$target_dir" \;

echo "All EPS files were copied over."
