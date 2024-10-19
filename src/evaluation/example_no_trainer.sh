python src/evaluation/example_no_trainer.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name self_generated \
    --dataset_split_type test \
    --table_ext .csv \
    --batch_size 16 \
    --max_new_tokens 100 \
    --temperature 0.1 \
