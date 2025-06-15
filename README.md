# KaLactica

KaLactica is a memory- and topology-enhanced successor to Galactica, designed to generate domain-aware scientific prose and code with improved factual grounding and coherence. It combines retrieval-augmented generation, a hierarchical memory system, and topological curriculum learning to produce high-quality, verifiable outputs while maintaining a small computational footprint.

## Quickstart

```bash
# Install in development mode
pip install -e .

# Run the demo
python demo.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      KaLactica Core                     │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  Retrieval  │   Memory    │  Topology   │   Safety     │
│   Layer     │   Layer     │   Layer     │   Filter     │
├─────────────┼─────────────┼─────────────┼──────────────┤
│  FAISS      │  Graph      │  Betti      │  Wasserstein │
│  Index      │  Memory     │  Signature  │  Distance    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬───────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                    Base Model (LLaMA)                    │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Retrieval-Augmented Generation**: Dual-crop FAISS index for short-term and long-term memory
- **Memory Hierarchy**: Tracks dialogue state, code executions, and consequence notes
- **Topological Curriculum**: Orders tasks by Betti complexity for progressive learning
- **Safety Filter**: Persistent homology checks for generation consistency
- **Parameter-Efficient**: Uses QLoRA and DPO for efficient fine-tuning

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FAISS-CPU
- Transformers
- PEFT
- Gudhi (optional, for topology features)

## How to Launch and Prepare Your EC2 Instance for KaLactica

1. **Launch an EC2 Instance**
   - Go to the AWS Console → EC2 → Launch Instance.
   - Choose an Ubuntu Server AMI (e.g., 22.04 or 24.04 LTS, 64-bit x86).
   - Select an instance type (e.g., `t3.large` for CPU, `g5.xlarge` for GPU).
   - Under **Network settings**, select the VPC that matches your OpenSearch domain.
   - Choose a subnet ("No preference" is fine for most cases).
   - Enable auto-assign public IP if you want to SSH from your local machine.
   - Create or select a security group:
     - Allow SSH (port 22) from your IP (recommended) or from anywhere (0.0.0.0/0) for testing.
   - Create or select a key pair (download the `.pem` file and keep it safe).
   - Launch the instance.

2. **Find Your Instance's Public IP or DNS**
   - In the EC2 dashboard, select your instance and note the **Public IPv4 address** or **Public DNS**.

3. **SSH into Your Instance**
   - On your local machine, run:
     ```sh
     chmod 400 /path/to/kalactica-key.pem
     ssh -i /path/to/kalactica-key.pem ubuntu@<your-ec2-public-ip>
     ```
   - Replace `/path/to/kalactica-key.pem` with your key file and `<your-ec2-public-ip>` with your instance's IP.

4. **Transfer Files (e.g., kaggle.json) from Local to EC2**
   - On your local machine, run:
     ```sh
     scp -i /path/to/kalactica-key.pem /path/to/kaggle.json ubuntu@<your-ec2-public-ip>:~/
     ```
   - On EC2, move it to the right place:
     ```sh
     mkdir -p ~/.kaggle
     mv ~/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

5. **Install python3-venv and pip if missing**
   - If you get errors about missing `venv` or `pip`, run:
     ```sh
     sudo apt update
     sudo apt install python3-venv python3-pip python3-full -y
     ```
   - If `python3-pip` is still missing, use:
     ```sh
     curl -O https://bootstrap.pypa.io/get-pip.py
     sudo python3 get-pip.py
     ```

6. **Troubleshooting**
   - If you see `Permission denied (publickey)`, check your `.pem` file path and permissions.
   - If you see `no installation candidate` for pip/venv, make sure you ran `sudo apt update` first.
   - If you see `externally-managed-environment` errors, always use a virtual environment for Python work.
   - If you have network issues, check your security group and VPC/subnet settings.

## How to Set Up the Environment on the EC2 Instance

1. **Update and install system packages:**
   ```sh
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3-pip python3-venv git unzip -y
   ```
2. **Clone the repository:**
   ```sh
   git clone https://github.com/uncertainlyprincipaled/Kalactica.git
   cd Kalactica
   ```
3. **Set up Python virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install Python requirements:**
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Set up AWS and Kaggle credentials:**
   - Upload your `.env` directory (with `aws/credentials` and `kaggle/kaggle.json`) to the EC2 instance, or create them there.
   - Make sure your environment variables are loaded (the code uses `python-dotenv`).

## How to Run Preprocess on EC2 and Store Data

1. **Download Meta Kaggle and Meta Kaggle Code datasets using Kaggle API:**
   ```sh
   pip install kaggle
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d kaggle/meta-kaggle
   kaggle datasets download -d kaggle/meta-kaggle-code
   unzip meta-kaggle.zip -d data/
   unzip meta-kaggle-code.zip -d data/
   ```
2. **(Optional) Upload raw data to S3 for backup:**
   ```sh
   aws s3 cp data/meta-kaggle.zip s3://your-bucket-name/meta-kaggle.zip
   aws s3 cp data/meta-kaggle-code.zip s3://your-bucket-name/meta-kaggle-code.zip
   ```
   - Limit the number of requests to S3 by uploading only once and using local copies for processing.

## Instructions to Complete Preprocessing and Indexing

1. **Run preprocessing:**
   ```sh
   source venv/bin/activate
   python -m kalactica.preprocess --input data/KernelVersions.csv --output data/processed.jsonl
   # Optionally, add --sample 1000 to process a subset for testing
   ```
2. **Index the data into OpenSearch:**
   ```sh
   kalactica index --input data/processed.jsonl
   ```
   - This will use your `.env/aws/credentials` for S3 and OpenSearch.
3. **(Optional) Upload processed data to S3:**
   ```sh
   aws s3 cp data/processed.jsonl s3://your-bucket-name/processed.jsonl
   ```

## License

MIT License
