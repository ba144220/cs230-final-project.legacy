import numpy as np
import pandas as pd
import os

from data.generate_dataset import generate_table, generate_qa, save_table
from data.data_types import TaskEnum, FileTypeEnum
from typing import List
from tqdm import tqdm

np.random.seed(42)

def main():
    sizes = [4, 6, 8, 10, 12]
    split_sizes = {
        "train": 14000,
        "val": 1000,
        "test": 4000
    }
    
    if not os.path.exists("./datasets/self_generated/csv"):
        os.makedirs("./datasets/self_generated/csv")
    if not os.path.exists("./datasets/self_generated/data"):
        os.makedirs("./datasets/self_generated/data")
    
    for split in ["train", "val", "test"]:
        all_qa_df = pd.DataFrame()
        for table_idx in tqdm(range(split_sizes[split] // (8*2*len(sizes)))):
            for size in sizes:
                for task in TaskEnum:
                    # Randomly select a task
                    df = generate_table(task, size, size)
                    table_path = os.path.join("./datasets/self_generated", "csv", split)
                    save_table(df, FileTypeEnum.CSV, table_path, f"{table_idx}_{task.value}_{size}")
                    save_table(df, FileTypeEnum.TSV, table_path, f"{table_idx}_{task.value}_{size}")
                    save_table(df, FileTypeEnum.HTML, table_path, f"{table_idx}_{task.value}_{size}")
                    
                    table_path = f"csv/{split}/{table_idx}-{task.value}-{size}.csv"
                    qa_df = generate_qa(task, df)
                    # Set all "context" to the table path
                    qa_df["context"] = table_path
                    qa_df["task"] = task.value
                    qa_df["size"] = str(int(size))
                    all_qa_df = pd.concat([all_qa_df, qa_df])
            
        # Assign id="{split}-<row_number>" to each row
        all_qa_df["id"] = [f"{split}-{i}" for i in range(len(all_qa_df))]
        
        # Reorder columns in question,answer,context,id,task,direction,size
        all_qa_df = all_qa_df[["question", "answer", "context", "id", "task", "direction", "size"]]
        
        # Statistics
        ## Task distribution
        print(all_qa_df["task"].value_counts())
        ## Direction distribution
        print(all_qa_df["direction"].value_counts())
        ## Size distribution
        print(all_qa_df["size"].value_counts())
        
        # Save to csv
        all_qa_df.to_csv(f"./datasets/self_generated/data/{split}.csv", index=False)
        

if __name__ == "__main__":
    main()