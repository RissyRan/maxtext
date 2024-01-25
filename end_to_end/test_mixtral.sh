#!/bin/bash
set -e
idx=$(date +%Y-%m-%d-%H-%M)

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

# Download checkpoint, convert it to MaxText, and run inference
pip3 install torch
gsutil -m cp -r gs://maxtext-external/mixtral-8x7B-v0.1-Instruct /tmp
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/mixtral-8x7B-v0.1-Instruct --model-size mixtral-8x7b --maxtext-model-path gs://maxtext-mixtral/test/${idx}/decode-ckpt-maxtext/
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=gs://maxtext-mixtral/test/${idx}/decode-ckpt-maxtext/0/default run_name=runner_direct_${idx} per_device_batch_size=1 model_name='mixtral-8x7b' assets_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert=". I love to learn. I love to grow. I love to be challenged. I love to be inspired. I love to be encouraged. I love to be motivated. I love to be transformed. I love to be changed. I love to be renewed. I love to be refreshed. I love to be revived. I love to be reborn. I love to be re-created. I love to be re-imagined. I love to be re-invented. I love to be re-purposed. I love to be re-used. I love to be re-cycled. I love to be re-newed. I love to be re-freshed. I love to be re-vived. I love to be re-born. I love to be re-created. I love to be re-imagined. I love to be re-invented. I love to be re-purposed. I love to be re-used. I love to be re-cycled. I love to be re-" attention=dot_product
