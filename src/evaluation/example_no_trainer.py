"""
This is an example of how to evaluate a Llama model.
"""
import sys
import os
import torch
from dataclasses import dataclass, field
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from tqdm import tqdm
from dataset_reader import DatasetReader

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
@dataclass
class GenerationArguments:
    """
    Arguments pertaining to the generation of the model.
    """
    max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for the model"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for the model"}
    )
    
    
@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what dataset we are going to evaluate on.
    """
    dataset_name: str = field(
        metadata={"help": "Name of the dataset to evaluate on"}
    )
    dataset_split_type: str = field(
        metadata={"help": "Type of the dataset split to evaluate on"}
    )
    table_ext: str = field(
        default=".csv",
        metadata={"help": "Extension of the table file"}
    )


def parse_args():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, GenerationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, dataset_args, generation_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, dataset_args, generation_args = parser.parse_args_into_dataclasses()
    return model_args, dataset_args, generation_args


def load_dummy_dataset(dataset_args: DatasetArguments):
    """
    Load a dummy dataset for evaluation.
    """
    SYSTEM_PROMPT = "You are a helpful assistant. Please only answer the name of the person, place, or thing."
    dummy_dataset = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Who is the first man to walk on the moon?"}
            ],
            "answer": "Neil Armstrong",
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is the capital of United States?"}
            ],
            "answer": "Washington D.C.",
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is the capital of China?"}
            ],
            "answer": "Beijing",
        }
    ] * 5
    return dummy_dataset

def data_preprocessing(dataset: List[dict], tokenizer: AutoTokenizer, batch_size: int = 4):
    """
    Preprocess the dataset.
    """
    dataset_input_strings = []
    batches = []
    for data in dataset:
        input_string = tokenizer.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=True)
        dataset_input_strings.append(input_string)
        
    # Batch encode the dataset
    for i in range(0, len(dataset_input_strings), batch_size):
        batch = dataset_input_strings[i:i+batch_size]
        batches.append(tokenizer(batch, padding=True, truncation=True, return_tensors="pt"))
    
    return batches
    

def main():
    model_args, dataset_args, generation_args = parse_args()
    print(dataset_args)

    print("Loading model...")
    if torch.cuda.is_available():   
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
   
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token


    dataset_reader = DatasetReader(dataset_args.dataset_name, dataset_args.dataset_split_type, dataset_args.table_ext)
    dataset = dataset_reader.read_file_as_prompt()
    print("Preprocessing data...")
    batches = data_preprocessing(dataset, tokenizer, generation_args.batch_size)
    
    inference_results = []

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(batches):
            # Move the batch to the model's device
            batch = batch.to(model.device)
            outputs = model.generate(
                **batch, 
                max_new_tokens=generation_args.max_new_tokens,
                temperature=generation_args.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Get only the output tokens
            batch_length = batch.input_ids.shape[1]
            output_ids = outputs[:, batch_length:]
            inference_results.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    
    # Evaluate the results
    print("Evaluating results...")
    correct_count = 0
    for i, result in enumerate(inference_results):
        print(f"Result {i+1}: {result}")
        print(f"Expected: {dataset[i]['answer']}")
        if result == dataset[i]['answer']:
            correct_count += 1
        print()
    print("-" * 100)
    print(f"Accuracy: {correct_count / len(inference_results) * 100:.2f}%")

if __name__ == "__main__":
    main()
