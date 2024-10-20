import pandas as pd
import os
from typing import List, Dict
from transformers import LlamaTokenizerFast

from data.data_types import DatasetOutput, TaskEnum, DirectionEnum, FileTypeEnum, DatasetSplitTypeEnum

class DatasetReader:
    def __init__(
        self, 
        dataset_name: str, 
        dataset_split_type: DatasetSplitTypeEnum, 
        table_format: FileTypeEnum,
        tokenizer: LlamaTokenizerFast,
        root_path: str = "./datasets",
        system_prompt: str = "You are a helpful assistant."
    ):
        self.dataset_name = dataset_name
        if not isinstance(dataset_split_type, DatasetSplitTypeEnum):
            self.dataset_split_type = DatasetSplitTypeEnum(dataset_split_type)
        else:
            self.dataset_split_type = dataset_split_type
            
        if not isinstance(table_format, FileTypeEnum):
            self.table_format = FileTypeEnum(table_format)
        else:
            self.table_format = table_format
        self.root_path = root_path
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
    def read_table(self, table_name: str):
        if not table_name:
            return ""
        if table_name.endswith('.csv'):
            table_name = table_name[:-4]

        file_path = os.path.join(self.root_path, self.dataset_name, table_name + "." + self.table_format.value)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist in the path {self.root_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        return file_content

    def read_file(self) -> List[DatasetOutput]:
        file_path = os.path.join(self.root_path, self.dataset_name, 'data', self.dataset_split_type.value + '.csv')
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
    
    def apply_chat_template(self):
        outputs = self.read_file()
        messages: List[List[Dict[str, str]]] = []
        for output in outputs:
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": output.question + "\n" + output.context},
            ]
            messages.append(message)
            
        dataset_strings: List[str] = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return dataset_strings
    
    def read_file_as_prompt(self):
        outputs = self.read_file()

        dataset = []
        for output in outputs:
            data = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": output.question},
                ],
                "answer": output.answer
            }
            dataset.append(data)
        return dataset