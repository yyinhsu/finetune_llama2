import pandas as pd
import json
import math
import numpy as np
from transformers import AutoTokenizer
from typing import Tuple
import huggingface_hub


class DataStatsProvider:
    def __init__(self, train_data_path: str, test_data_path: str,
                 pretrained_model: str, huggingface_hub_token: str) -> None:
        self.train_data, self.test_data = self.load_train_test_data(train_data_path, test_data_path)
        self.tokenizer = self.load_tokenizer(pretrained_model, huggingface_hub_token)

        self.input_max_length, self.output_max_length, self.unknown_tokens = self.calculate_data_stats()


    def calculate_data_stats(self) -> Tuple[int, int, list]:
        """
        Calculate the data statistics of the train and test data. This is for setting the parameters of the model training.
        Returns:
            input_max_length (int): the max length of the input text for the model
            output_max_length (int): the max length of the output text for the model
            unknown_tokens (list): the unknown tokens from the data that the tokenizer cannot read. These tokens are going to be add to the training tokenizer.
        """
        train_input_lengths, train_output_lengths, train_unknown_tokens = self.get_tokenized_data_info(self.train_data)
        test_input_lengths, test_output_lengths, test_unknown_tokens = self.get_tokenized_data_info(self.test_data)
        input_max_length = self.get_avg_plus_n_std(train_input_lengths + test_input_lengths, n=3)
        output_max_length = self.get_avg_plus_n_std(train_output_lengths + test_output_lengths, n=3)

        return input_max_length, output_max_length, list(set(train_unknown_tokens + test_unknown_tokens))


    def load_train_test_data(self, train_data_path: str, test_data_path: str) -> Tuple[list, list]:
        """
        Load the train and test data from the data path, and return the list of the train and test data.
        The jsonl file should be in the format of:
        {"input": "input text", "output": "output text"}\n{...}\n...
        The json file should be in the format of:
        {"data": [{"input": "input text", "output": "output text"}, {...}, ...]}
        Args:
            train_data_path: the path to the train data
            test_data_path: the path to the test data
        Returns:
            train_data: the list of dict of the train data
            test_data: the list of dict of the test data
            Example: [{"input": "input text", "output": "output text"}, {...}, ...]
        """
        if ".jsonl" in train_data_path and ".jsonl" in test_data_path:
            with open(train_data_path, 'r') as f:
                train_data = f.readlines()
            train_data = [json.loads(data) for data in train_data]
            with open(test_data_path, 'r') as f:
                test_data = f.readlines()
            test_data = [json.loads(data) for data in test_data]
        elif ".json" in train_data_path and ".json" in test_data_path:
            with open(train_data_path, 'r') as f:
                train_data = json.load(f)["data"]
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)["data"]
        else:
            raise ValueError("The data format is not supported. Only support jsonl and json format.")
        return train_data, test_data


    def load_tokenizer(self, pretrained_model: str, huggingface_hub_token: str) -> AutoTokenizer:
        """
        Load the tokenizer from the pretrained model.
        Args:
            None
        Returns:
            tokenizer: the tokenizer
        """
        if not huggingface_hub_token == 'login_not_required':
            huggingface_hub.login(token=huggingface_hub_token)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    def tokenize(self, data_text: str, max_length: int=1000) -> dict:
        """
        Tokenize the data with tokenizer, and return the model inputs with keys: ['input_ids', 'attention_mask', 'labels'].
        Args:
            data_text (str): the data text to be tokenized
            max_length (int): the max length of the tokenized text
        Returns:
            model_inputs: the model inputs with keys: ['input_ids', 'attention_mask', 'labels']
        """

        model_inputs = self.tokenizer(data_text,
                                      max_length=max_length,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
        return model_inputs


    def get_tokenized_data_info(self, data_list: list) -> Tuple[int, int, list]:
        """
        Count the tokenized length of the input and output text, and the unknown token from data that the tokenizer cannot read.
        Args:
            data_list (list): the list of dict of the data, with keys: ['input', 'output']
        Returns:
            input_lengths: the tokenized length of the input text
            output_lengths: the tokenized length of the output text
            unknown_tokens: the unknown token from data that the tokenizer cannot read
        """
        print("Start getting the tokenized data info...")
        input_lengths = list()
        output_lengths = list()
        data_count = 0

        unknown_tokens = list()

        for data in data_list:
            ## Get the tokenized object of the input and output text
            input_text = data['input']
            output_text = data['output']
            tokenized_input = self.tokenize(input_text)
            tokenized_output = self.tokenize(output_text)

            ## Get the unknown token from data that the tokenizer cannot read
            tokenized_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"].squeeze().tolist())
            unknown_tokens.extend([token for token in tokenized_tokens if token == '[UNK]'])
            tokenized_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"].squeeze().tolist())
            unknown_tokens.extend([token for token in tokenized_tokens if token == '[UNK]'])

            ## Get the length of the tokenized input and output text
            length_input = (tokenized_input['attention_mask'] > 0).sum().item()
            length_output = (tokenized_output['attention_mask'] > 0).sum().item()
            input_lengths.append(length_input)
            output_lengths.append(length_output)
            data_count += 1

        print('data_count: ', data_count)
        unknown_tokens = list(set(unknown_tokens))
        print('unknown_tokens: ', unknown_tokens)

        return input_lengths, output_lengths, unknown_tokens


    def get_avg_plus_n_std(self, lengths: list, n: int=3) -> int:
        """
        Get the average plus n times standard deviation of the lengths.
        Args:
            lengths (list): the list of the lengths
            n (int): the number of times of the standard deviation
        Returns:
            int: the average plus n times standard deviation of the lengths
        """
        avg = sum(lengths) / len(lengths)
        std = np.std(lengths)
        return math.ceil(avg + n * std)
