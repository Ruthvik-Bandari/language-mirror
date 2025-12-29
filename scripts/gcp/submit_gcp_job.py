#!/usr/bin/env python3
"""
🚀 Language Mirror Pro - GCP Vertex AI Job Submission
=====================================================
Submit training job to Google Cloud Vertex AI.

Prerequisites:
1. GCP Account with billing enabled
2. gcloud CLI installed & authenticated
3. Vertex AI API enabled
4. Training data uploaded to GCS

Usage:
    # Setup GCP (first time)
    python submit_gcp_job.py --setup --project YOUR_PROJECT_ID
    
    # Upload data
    python submit_gcp_job.py --upload-data --project YOUR_PROJECT_ID
    
    # Submit training job
    python submit_gcp_job.py --train --project YOUR_PROJECT_ID --gpu T4
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# GCP Configuration
DEFAULT_REGION = "us-central1"
DEFAULT_MACHINE = "n1-standard-8"

GPU_CONFIGS = {
    "T4": {"type": "NVIDIA_TESLA_T4", "count": 1, "cost": "$0.35/hr"},
    "V100": {"type": "NVIDIA_TESLA_V100", "count": 1, "cost": "$2.48/hr"},
    "A100": {"type": "NVIDIA_TESLA_A100", "count": 1, "cost": "$3.67/hr"},
    "4xT4": {"type": "NVIDIA_TESLA_T4", "count": 4, "cost": "$1.40/hr"},
}


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command"""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ Error: {result.stderr}")
    return result


def setup_gcp(project_id: str, region: str):
    """Setup GCP project for Vertex AI training"""
    print("\n🔧 Setting up GCP project...")
    
    # Set project
    run_cmd(f"gcloud config set project {project_id}")
    
    # Enable APIs
    print("\n📡 Enabling required APIs...")
    apis = [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "compute.googleapis.com",
        "containerregistry.googleapis.com",
    ]
    for api in apis:
        run_cmd(f"gcloud services enable {api}", check=False)
    
    # Create bucket
    bucket_name = f"{project_id}-training"
    print(f"\n🪣 Creating storage bucket: {bucket_name}")
    run_cmd(f"gsutil mb -l {region} gs://{bucket_name}", check=False)
    
    print(f"\n✅ GCP setup complete!")
    print(f"   Project: {project_id}")
    print(f"   Region: {region}")
    print(f"   Bucket: gs://{bucket_name}")
    
    return bucket_name


def upload_data(project_id: str, data_dir: str):
    """Upload training data to GCS"""
    bucket_name = f"{project_id}-training"
    
    print(f"\n📤 Uploading data to gs://{bucket_name}/data/...")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run prepare_large_datasets.py first!")
        return False
    
    # Upload training files
    for pattern in ["*.json", "*.jsonl"]:
        run_cmd(f"gsutil -m cp {data_path}/{pattern} gs://{bucket_name}/data/", check=False)
    
    # Upload training script
    script_dir = Path(__file__).parent
    run_cmd(f"gsutil cp {script_dir}/train_large.py gs://{bucket_name}/code/")
    
    print(f"\n✅ Data uploaded to gs://{bucket_name}/data/")
    
    # List uploaded files
    run_cmd(f"gsutil ls gs://{bucket_name}/data/")
    
    return True


def create_training_script():
    """Create the training script that will run on Vertex AI"""
    
    script = '''#!/usr/bin/env python3
"""Training script for Vertex AI"""

import os
import json
import subprocess

# Install dependencies
subprocess.run(["pip", "install", "torch", "datasets", "google-cloud-storage", "tqdm"])

# Download training script from GCS
from google.cloud import storage

bucket_name = os.environ.get("BUCKET_NAME")
client = storage.Client()
bucket = client.bucket(bucket_name)

# Download code
blob = bucket.blob("code/train_large.py")
blob.download_to_filename("/tmp/train_large.py")

# Download data
data_dir = "/tmp/data"
os.makedirs(data_dir, exist_ok=True)

for blob in bucket.list_blobs(prefix="data/"):
    if blob.name.endswith((".json", ".jsonl")):
        filename = os.path.basename(blob.name)
        blob.download_to_filename(f"{data_dir}/{filename}")
        print(f"Downloaded: {filename}")

# Run training
import sys
sys.path.insert(0, "/tmp")
from train_large import main

# Set environment
os.environ["DATA_DIR"] = data_dir
os.environ["OUTPUT_DIR"] = "/tmp/output"
os.environ["GCS_BUCKET"] = bucket_name

# Override sys.argv for argparse
sys.argv = [
    "train_large.py",
    "--data-dir", data_dir,
    "--output-dir", "/tmp/output",
    "--epochs", os.environ.get("EPOCHS", "10"),
    "--batch-size", os.environ.get("BATCH_SIZE", "64"),
    "--gcs-bucket", bucket_name,
]

main()

# Upload final model
print("Uploading final model to GCS...")
for f in os.listdir("/tmp/output"):
    if f.endswith(".pt"):
        blob = bucket.blob(f"models/{f}")
        blob.upload_from_filename(f"/tmp/output/{f}")
        print(f"Uploaded: models/{f}")
'''
    
    return script


def submit_vertex_job(
    project_id: str,
    region: str,
    gpu_type: str,
    epochs: int,
    batch_size: int,
):
    """Submit training job to Vertex AI"""
    
    bucket_name = f"{project_id}-training"
    gpu_config = GPU_CONFIGS.get(gpu_type, GPU_CONFIGS["T4"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"language_mirror_{timestamp}"
    
    print(f"\n🚀 Submitting Vertex AI training job...")
    print(f"   Job name: {job_name}")
    print(f"   GPU: {gpu_type} ({gpu_config['cost']})")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Create training script
    script_content = create_training_script()
    script_path = Path("/tmp/vertex_train.py")
    script_path.write_text(script_content)
    
    # Upload script
    run_cmd(f"gsutil cp {script_path} gs://{bucket_name}/code/vertex_train.py")
    
    # Create job config
    job_config = {
        "displayName": job_name,
        "jobSpec": {
            "workerPoolSpecs": [{
                "machineSpec": {
                    "machineType": DEFAULT_MACHINE,
                    "acceleratorType": gpu_config["type"],
                    "acceleratorCount": gpu_config["count"],
                },
                "replicaCount": 1,
                "pythonPackageSpec": {
                    "executorImageUri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13.py310:latest",
                    "packageUris": [f"gs://{bucket_name}/code/vertex_train.py"],
                    "pythonModule": "vertex_train",
                    "env": [
                        {"name": "BUCKET_NAME", "value": bucket_name},
                        {"name": "EPOCHS", "value": str(epochs)},
                        {"name": "BATCH_SIZE", "value": str(batch_size)},
                    ],
                },
            }],
        },
    }
    
    config_path = Path("/tmp/job_config.json")
    config_path.write_text(json.dumps(job_config, indent=2))
    
    # Submit job
    cmd = f"""gcloud ai custom-jobs create \
        --region={region} \
        --display-name={job_name} \
        --worker-pool-spec=machine-type={DEFAULT_MACHINE},replica-count=1,accelerator-type={gpu_config['type']},accelerator-count={gpu_config['count']},executor-image-uri=gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13.py310:latest,local-package-path=.,python-module=train_large \
        --args="--data-dir=/gcs/{bucket_name}/data,--output-dir=/gcs/{bucket_name}/models,--epochs={epochs},--batch-size={batch_size}" \
    """
    
    # Simpler submission using gcloud ai
    print("\n📋 Job configuration:")
    print(f"   Machine: {DEFAULT_MACHINE}")
    print(f"   GPU: {gpu_config['type']} x {gpu_config['count']}")
    print(f"   Container: pytorch-gpu.1-13")
    
    print("\n⚡ Submitting job...")
    
    # Use the Python SDK for more reliable submission
    try:
        from google.cloud import aiplatform
        
        aiplatform.init(project=project_id, location=region)
        
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[{
                "machine_spec": {
                    "machine_type": DEFAULT_MACHINE,
                    "accelerator_type": gpu_config["type"],
                    "accelerator_count": gpu_config["count"],
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13.py310:latest",
                    "command": ["python", "-c", script_content],
                    "env": [
                        {"name": "BUCKET_NAME", "value": bucket_name},
                        {"name": "EPOCHS", "value": str(epochs)},
                        {"name": "BATCH_SIZE", "value": str(batch_size)},
                    ],
                },
            }],
        )
        
        job.run(sync=False)
        
        print(f"\n✅ Job submitted successfully!")
        print(f"   Job ID: {job.resource_name}")
        print(f"\n📊 Monitor at:")
        print(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
        
    except ImportError:
        print("❌ google-cloud-aiplatform not installed")
        print("   Run: pip install google-cloud-aiplatform")
        print("\n📋 Manual submission command:")
        print(cmd)


def download_model(project_id: str, output_dir: str):
    """Download trained model from GCS"""
    bucket_name = f"{project_id}-training"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📥 Downloading model from gs://{bucket_name}/models/...")
    
    run_cmd(f"gsutil -m cp gs://{bucket_name}/models/*.pt {output_path}/")
    
    print(f"\n✅ Model downloaded to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GCP Vertex AI Training")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--region", default=DEFAULT_REGION, help="GCP Region")
    parser.add_argument("--setup", action="store_true", help="Setup GCP project")
    parser.add_argument("--upload-data", action="store_true", help="Upload data to GCS")
    parser.add_argument("--train", action="store_true", help="Submit training job")
    parser.add_argument("--download", action="store_true", help="Download trained model")
    parser.add_argument("--data-dir", default="datasets/large_scale", help="Local data directory")
    parser.add_argument("--output-dir", default="trained_model_gcp", help="Output directory")
    parser.add_argument("--gpu", default="T4", choices=GPU_CONFIGS.keys(), help="GPU type")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Language Mirror Pro - GCP Vertex AI Training")
    print("=" * 60)
    
    if args.setup:
        setup_gcp(args.project, args.region)
    
    if args.upload_data:
        upload_data(args.project, args.data_dir)
    
    if args.train:
        submit_vertex_job(
            args.project,
            args.region,
            args.gpu,
            args.epochs,
            args.batch_size,
        )
    
    if args.download:
        download_model(args.project, args.output_dir)
    
    if not any([args.setup, args.upload_data, args.train, args.download]):
        parser.print_help()
        print("\n📋 Quick start:")
        print(f"  1. Setup:      python {sys.argv[0]} --setup --project YOUR_PROJECT")
        print(f"  2. Upload:     python {sys.argv[0]} --upload-data --project YOUR_PROJECT")
        print(f"  3. Train:      python {sys.argv[0]} --train --project YOUR_PROJECT --gpu T4")
        print(f"  4. Download:   python {sys.argv[0]} --download --project YOUR_PROJECT")


if __name__ == "__main__":
    main()
