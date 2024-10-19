from dataclasses import dataclass
import pandas as pd
import os
from typing import Optional
from enum import Enum

class TaskEnum(Enum):
    ARITHMETIC = "arithmetic"
    LIST_ITEMS = "list_items"

class DirectionEnum(Enum):
    ROW = "row"
    COLUMN = "column"

@dataclass
class DatasetOutput:
    question: str
    answer: str
    context: str
    id: Optional[str] = None
    task: Optional[TaskEnum] = None
    direction: Optional[DirectionEnum] = None
    size: Optional[str] = None

class DatasetReader:
    def __init__(self, dataset_name: str, dataset_split_type: str, table_ext: str):
        self.dataset_name = dataset_name
        self.dataset_split_type = dataset_split_type
        self.table_ext = table_ext
        self.root_path = "./datasets"

    def read_table(self, table_name: str):
        if not table_name:
            return ""
        if table_name.endswith('.csv'):
            table_name = table_name[:-4]

        file_path = os.path.join(self.root_path, self.dataset_name, table_name + self.table_ext)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist in the path {self.root_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        return file_content

    def read_file(self):
        file_path = os.path.join(self.root_path, self.dataset_name, 'data', self.dataset_split_type + '.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist in the path {self.root_path}")
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['question', 'answer', 'context', 'id'])

        outputs = []
        for index, row in df.iterrows():
            context = self.read_table(row['context'])
            task = TaskEnum(row['task']) if row['task'] in TaskEnum._value2member_map_ else None
            direction = DirectionEnum(row['direction']) if row['direction'] in DirectionEnum._value2member_map_ else None
            output = DatasetOutput(
                question=row['question'],
                answer=row['answer'],
                context=context,
                id=row['id'],
                task=task,
                direction=direction,
                size=row['size'] if pd.notna(row['size']) else None  # Convert NaN to None
            )
            outputs.append(output)

        return outputs
    
    def read_file_as_prompt(self):
        outputs = self.read_file()
        print(len(outputs))
        SYSTEM_PROMPT = "You are a helpful assistant. Please only answer the name of the person, place, or thing."

        dataset = []
        for output in outputs:
            data = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": output.question},
                ],
                "answer": output.answer
            }
            dataset.append(data)
        return dataset