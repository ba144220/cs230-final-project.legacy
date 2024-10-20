import os
import pandas as pd
import numpy as np
import string
from typing import List, Dict

from data.data_types import TaskEnum, FileTypeEnum, DirectionEnum

# Set seed for reproducibility
np.random.seed(42)

def generate_table(
        task: TaskEnum, 
        row_size: int, 
        col_size: int, 
        row_name: str = "Row", 
        col_name: str = "Col"
    ):
    """
    task: QA task to be performed any of (arithmetic, item)
    row_size: row size of the dataset
    col_size: col size of the dataset
    col_name: name of the columns
    row_name: name of the rows
    """
    columns = [col_name + " " + str(i+1) for i in range(col_size)]
    rows = [row_name + " " + str(i+1) for i in range(row_size)]
    if task == TaskEnum.ARITHMETIC: 
        df = pd.DataFrame(np.random.randint(0, 100, size=(row_size, col_size)), columns=columns, index=rows)
    elif task == TaskEnum.LIST_ITEMS:
        df = pd.DataFrame(np.random.choice(list(string.ascii_uppercase), size=(row_size, col_size)), columns=columns, index=rows)
    return df


def generate_qa(task: TaskEnum, df: pd.DataFrame) -> pd.DataFrame:
    # Sanity check
    if df.shape[0] < 4 or df.shape[1] < 4:
        raise ValueError("DataFrame must have at least 4 rows and 4 columns")
    # Generate 4 questions on rows and 4 questions on columns
    qa_list = []
    # Pick 4 random rows and 4 random columns
    row_indices = np.random.choice(np.arange(df.shape[0]), 4, replace=False)
    col_indices = np.random.choice(np.arange(df.shape[1]), 4, replace=False)
    
    if task == TaskEnum.ARITHMETIC:
        for row_idx in row_indices:
            row_name = df.index[row_idx]
            operation_type = np.random.choice(["minimum", "maximum"], 1)[0]
            question = f"What is the {operation_type} of the values in {row_name}?"
            answer = df.iloc[row_idx].min() if operation_type == "minimum" else df.iloc[row_idx].max()
            qa_list.append({"question": question, "answer": answer, "direction": DirectionEnum.ROW.value})
        for col_idx in col_indices:
            col_name = df.columns[col_idx]
            operation_type = np.random.choice(["minimum", "maximum"], 1)[0]
            question = f"What is the {operation_type} of the values in {col_name}?"
            answer = df.iloc[:, col_idx].min() if operation_type == "minimum" else df.iloc[:, col_idx].max()
            qa_list.append({"question": question, "answer": answer, "direction": DirectionEnum.COLUMN.value})
    elif task == TaskEnum.LIST_ITEMS:
        for row_idx in row_indices:
            row_name = df.index[row_idx]
            question = f"Please list all the items in {row_name} and separate them by commas."
            answer = ",".join(df.iloc[row_idx])
            qa_list.append({"question": question, "answer": answer, "direction": DirectionEnum.ROW.value})
        for col_idx in col_indices:
            col_name = df.columns[col_idx]
            question = f"Please list all the items in {col_name} and separate them by commas."
            answer = ",".join(df.iloc[:, col_idx])
            qa_list.append({"question": question, "answer": answer, "direction": DirectionEnum.COLUMN.value})
    df = pd.DataFrame(qa_list)
    return df
    
    
def save_table(df: pd.DataFrame, file_type: FileTypeEnum, file_path: str, file_name: str):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if file_type == FileTypeEnum.TSV:
        df.to_csv(os.path.join(file_path, file_name + '.tsv'), sep='\t', index=True)
    elif file_type == FileTypeEnum.CSV:
        df.to_csv(os.path.join(file_path, file_name + '.csv'), index=True)
    elif file_type == FileTypeEnum.HTML:
        df.to_html(os.path.join(file_path, file_name + '.html') , index=True)

