"""
🚀 Submit Training Job to Vertex AI
"""
import subprocess
import sys

# Install aiplatform if needed
try:
    from google.cloud import aiplatform
except ImportError:
    print("Installing google-cloud-aiplatform...")
    subprocess.run([sys.executable, "-m", "pip", "install", "google-cloud-aiplatform"], check=True)
    from google.cloud import aiplatform

from datetime import datetime

PROJECT_ID = "language-mirror-pro"
REGION = "us-central1"
BUCKET = "language-mirror-pro-training"

# Training code that will run on Vertex AI
TRAINING_CODE = '''
import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "torch", "tqdm"], check=True)

# Download from GCS
subprocess.run(["gsutil", "-m", "cp", "-r", "gs://language-mirror-pro-training/data", "/tmp/"], check=True)
subprocess.run(["gsutil", "cp", "gs://language-mirror-pro-training/code/train_large.py", "/tmp/"], check=True)

# Run training
os.chdir("/tmp")
subprocess.run([sys.executable, "train_large.py", 
    "--data-dir", "/tmp/data",
    "--output-dir", "/tmp/output", 
    "--epochs", "10",
    "--batch-size", "64"
], check=True)

# Upload results
subprocess.run(["gsutil", "-m", "cp", "-r", "/tmp/output/*.pt", "gs://language-mirror-pro-training/models/"], check=True)
print("Done!")
'''

def main():
    print("🚀 Submitting Language Mirror training to Vertex AI...")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Region: {REGION}")
    print(f"   GPU: NVIDIA T4")
    print()
    
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket="gs://language-mirror-pro-training")
    
    # Create job
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"language-mirror-{timestamp}"
    
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-8",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0.py310:latest",
                    "command": ["python", "-c", TRAINING_CODE],
                },
            }
        ],
    )
    
    print(f"📋 Job name: {job_name}")
    print("⏳ Submitting job (this may take a minute)...")
    
    # Submit job (non-blocking)
    job.submit()
    
    print()
    print("✅ Job submitted successfully!")
    print()
    print("📊 Monitor your job at:")
    print(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print()
    print("💰 Estimated cost: ~$0.70 for full training (T4 GPU)")
    print()
    print("📥 When complete, download model with:")
    print(f"   gsutil cp gs://{BUCKET}/models/best_model.pt trained_model_gcp/")

if __name__ == "__main__":
    main()
