#!/bin/bash
# Upload FolDE data to Google Cloud Storage bucket
# This script uploads ~152GB of data required for running FolDE simulations

set -e  # Exit on error

BUCKET="gs://foldedata"
DATA_DIR="backend/folde/data"

echo "==================================="
echo "FolDE Data Upload to GCS"
echo "==================================="
echo ""
echo "This will upload approximately 152GB of data to ${BUCKET}"
echo "Breakdown:"
echo "  - DMS_ProteinGym_substitutions: ~1.0GB"
echo "  - embeddings: ~151GB"
echo "  - naturalness: ~257MB"
echo "  - DMS_substitutions.csv: ~208KB"
echo "  - FLIP-AAV_multimutant_dataset.csv: ~462MB"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 1
fi

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil not found. Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    echo "Please run this script from the repository root."
    exit 1
fi

echo ""
echo "Starting upload..."
echo ""

# Upload with progress and parallelism for faster transfer
# -m: parallel upload
# -r: recursive
# -o GSUtil:parallel_process_count=16: use 16 parallel processes
# -o GSUtil:parallel_thread_count=10: use 10 threads per process

echo "Uploading DMS_ProteinGym_substitutions (~1.0GB)..."
gsutil -m -o GSUtil:parallel_process_count=16 cp -r "${DATA_DIR}/DMS_ProteinGym_substitutions" "${BUCKET}/"

echo ""
echo "Uploading embeddings (~151GB, this will take a while)..."
gsutil -m -o GSUtil:parallel_process_count=16 cp -r "${DATA_DIR}/embeddings" "${BUCKET}/"

echo ""
echo "Uploading naturalness (~257MB)..."
gsutil -m -o GSUtil:parallel_process_count=16 cp -r "${DATA_DIR}/naturalness" "${BUCKET}/"

echo ""
echo "Uploading CSV files..."
gsutil cp "${DATA_DIR}/DMS_substitutions.csv" "${BUCKET}/"
gsutil cp "${DATA_DIR}/FLIP-AAV_multimutant_dataset.csv" "${BUCKET}/"

echo ""
echo "==================================="
echo "Upload complete!"
echo "==================================="
echo ""
echo "Verifying upload..."
gsutil ls -lh "${BUCKET}"

echo ""
echo "Done! Data is now available at ${BUCKET}"
echo ""
echo "To make the bucket publicly readable (recommended for open release):"
echo "  gsutil iam ch allUsers:objectViewer ${BUCKET}"
