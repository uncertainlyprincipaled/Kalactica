#!/bin/bash

# Function to download with retries
download_with_retry() {
    local dataset=$1
    local max_retries=3
    local retry_count=0
    local success=false

    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        echo "Attempting to download $dataset (Attempt $((retry_count + 1))/$max_retries)..."
        if kaggle datasets download -d "$dataset" --force; then
            success=true
            echo "Successfully downloaded $dataset"
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Download failed. Waiting 10 seconds before retry..."
                sleep 10
            fi
        fi
    done

    if [ "$success" = false ]; then
        echo "Failed to download $dataset after $max_retries attempts"
        exit 1
    fi
}

# Check if required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path-to-kaggle.json> <path-to-ec2-key.pem> <ec2-username@ec2-ip>"
    echo "Example: $0 ~/.kaggle/kaggle.json ~/keys/my-key.pem ubuntu@ec2-12-34-56-78.compute-1.amazonaws.com"
    exit 1
fi

KAGGLE_JSON=$1
EC2_KEY=$2
EC2_HOST=$3

# Verify kaggle.json exists and has correct permissions
if [ ! -f "$KAGGLE_JSON" ]; then
    echo "Error: kaggle.json not found at $KAGGLE_JSON"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download only meta-kaggle dataset (skipping meta-kaggle-code due to size ~300GB)
echo "Downloading Kaggle dataset..."
echo "Note: Skipping meta-kaggle-code dataset (~300GB) to save local disk space"
download_with_retry "kaggle/meta-kaggle"

# Verify download before proceeding
if [ ! -f "meta-kaggle.zip" ]; then
    echo "Error: meta-kaggle.zip is missing"
    exit 1
fi

# Unzip the file
echo "Unzipping dataset..."
unzip -o meta-kaggle.zip -d data/ || { echo "Error unzipping meta-kaggle.zip"; exit 1; }

# Create data directory on EC2
echo "Creating data directory on EC2..."
ssh -i "$EC2_KEY" "$EC2_HOST" "mkdir -p ~/Kalactica/data" || { echo "Error creating directory on EC2"; exit 1; }

# Transfer the data file to EC2
echo "Transferring data to EC2..."
scp -i "$EC2_KEY" data/meta-kaggle.zip "$EC2_HOST:~/Kalactica/data/" || { echo "Error transferring meta-kaggle.zip"; exit 1; }

# Unzip the file on EC2
echo "Unzipping file on EC2..."
ssh -i "$EC2_KEY" "$EC2_HOST" "cd ~/Kalactica/data && unzip -o meta-kaggle.zip" || { echo "Error unzipping file on EC2"; exit 1; }

echo "Done! Meta-kaggle data has been downloaded and transferred to EC2."
echo "Note: meta-kaggle-code dataset (~300GB) was skipped to save local disk space" 