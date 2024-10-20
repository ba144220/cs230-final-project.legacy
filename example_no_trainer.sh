python3 example_no_trainer.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name self_generated \
    --dataset_split_type test \
    --table_ext csv \
    --batch_size 4 \
    --max_new_tokens 100 \
    --temperature 0.1 \
