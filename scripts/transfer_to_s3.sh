#!/bin/bash

# Check if required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path-to-meta-kaggle.zip> <path-to-ec2-key.pem> <ec2-username@ec2-ip>"
    echo "Example: $0 data/meta-kaggle.zip ~/keys/my-key.pem ubuntu@ec2-12-34-56-78.compute-1.amazonaws.com"
    exit 1
fi

ZIP_FILE=$1
EC2_KEY=$2
EC2_HOST=$3

# Verify zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: meta-kaggle.zip not found at $ZIP_FILE"
    exit 1
fi

# Create data directory on EC2
echo "Creating data directory on EC2..."
ssh -i "$EC2_KEY" "$EC2_HOST" "mkdir -p ~/Kalactica/data" || { echo "Error creating directory on EC2"; exit 1; }

# Transfer the zip file to EC2
echo "Transferring zip file to EC2..."
scp -i "$EC2_KEY" "$ZIP_FILE" "$EC2_HOST:~/Kalactica/data/" || { echo "Error transferring zip file"; exit 1; }

# Unzip the file on EC2 and upload to S3
echo "Unzipping file on EC2 and uploading to S3..."
ssh -i "$EC2_KEY" "$EC2_HOST" "cd ~/Kalactica/data && \
    unzip -o $(basename $ZIP_FILE) && \
    aws s3 cp . s3://kalactica/meta-kaggle/ --recursive" || { 
    echo "Error during unzip or S3 upload"; 
    exit 1; 
}

echo "Done! Meta-kaggle data has been transferred to EC2 and uploaded to S3." 