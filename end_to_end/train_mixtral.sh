#!/bin/bash

# This script is designed for internal use within Google. External users can adapt it by:
#  - Updating GCS paths (gs://) to your accessible locations.
#  - Using the checkpoint generated from train.py or available one in open source (i.e. https://files.mixtral-8x7b-v0-1.mistral.ai/Mixtral-8x7B-v0.1-Instruct.tar).

set -e
idx=$(date +%Y-%m-%d-%H-%M)
echo $idx

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

# Training
python3 MaxText/train.py MaxText/configs/base.yml load_parameters_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct-maxtext/0/default run_name=runner_${idx} per_device_batch_size=1 model_name=mixtral-8x7b ici_tensor_parallelism=4 ici_fsdp_parallelism=16 steps=10
