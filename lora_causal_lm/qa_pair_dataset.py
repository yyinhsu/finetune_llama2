from transformers import AutoTokenizer
import torch


class QAPairDataset(torch.utils.data.Dataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, q_max_length: int, a_max_length: int) -> None:
        """Initialize the QA pair dataset.

        :param: str data_blob: the path to the data blob
        :param: AutoTokenizer tokenizer: the tokenizer
        :param: int q_max_length: the maximum length of the question (input)
        :param: int a_max_length: the maximum length of the answer (output)
        """
        self.data_dicts = data
        self.tokenizer = tokenizer
        self.tokenized_data = self.__tokenize_dataset(self.data_dicts, q_max_length, a_max_length)
        self.q_max_length = q_max_length
        self.a_max_length = a_max_length

    @staticmethod
    def collate_fn(data) -> dict:
        """Function that use as the collate function for DataLoader."""
        keys = ['input_ids', 'attention_mask', 'labels']
        batch = dict()
        for row in data:
            for k in keys:
                print("\nkey:", k)
                if k not in batch:
                    batch[k] = row[k].unsqueeze(0).clone()
                else:
                    print("batch[k] shape:", batch[k].shape)
                    print("torch.tensor(row[k]).unsqueeze(0):", torch.tensor(row[k]).unsqueeze(0).shape)
                    batch[k] = torch.cat([batch[k], torch.tensor(row[k]).unsqueeze(0)], dim=0)

        return batch


    def __getitem__(self, index) -> dict:
        """Override the __getitem__ method to return the tokenized data."""
        return self.tokenized_data[index]


    def __len__(self) -> int:
        """Override the __len__ method to return the length of the dataset."""
        return len(self.data_dicts)


    def __tokenize_dataset(self, data_dicts: list, q_max_length: int, a_max_length: int) -> list:
        """Tokenize dataset with tokenizer, and return a list of tokenized data.

        :param: list data_dicts: the list of data dictionaries
        :param: int q_max_length: the maximum length of the question (input)
        :param: int a_max_length: the maximum length of the answer (output)
        :return: tokenized_data_list: the list of tokenized data
        :rtype: list
        """
        tokenized_data_list = []
        for data_dict in data_dicts:
            model_inputs = self.__preprocess_function(data_dict, q_max_length, a_max_length)
            tokenized_data_list.append(model_inputs)

        return tokenized_data_list


    def __preprocess_function(self, data, q_max_length, a_max_length) -> dict:
        """Preprocess the text data, and return the model inputs with keys: ['input_ids', 'attention_mask', 'labels'].

        :param: dict data: the dictionary of the data
        :param: int q_max_length: the maximum length of the question (input)
        :param: int a_max_length: the maximum length of the answer (output)
        :return model_inputs: the model inputs with keys: ['input_ids', 'attention_mask', 'labels']
        :rtype: dict
        """

        # max_length = q_max_length + a_max_length
        # model_inputs = self.tokenizer(data['input'] + "<answer>")
        # labels = self.tokenizer(data['output'] + "</answer>")

        # sample_input_ids = model_inputs["input_ids"].copy()
        # label_input_ids = labels["input_ids"].copy() + [self.tokenizer.eos_token_id]

        # model_inputs["input_ids"] = sample_input_ids + label_input_ids
        # labels["input_ids"] = [-100] * len(sample_input_ids) + label_input_ids
        # model_inputs["attention_mask"] = [1] * len(sample_input_ids + label_input_ids)

        # ## left padding
        # sample_input_ids = model_inputs["input_ids"].copy()
        # label_input_ids = labels["input_ids"].copy()

        # pad_num = max_length - len(sample_input_ids)
        # model_inputs["input_ids"] = [self.tokenizer.pad_token_id] * pad_num + sample_input_ids
        # model_inputs["attention_mask"] = [0] * pad_num + model_inputs["attention_mask"]
        # labels["input_ids"] = [-100] * pad_num + label_input_ids

        # input_ids = model_inputs["input_ids"].copy()
        # model_inputs["input_ids"] = torch.tensor(input_ids[:max_length])
        # model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"][:max_length])

        # ## Various ways to set the labels
        # # model_inputs["labels"] = torch.tensor(labels["input_ids"][:max_length])
        # model_inputs["labels"] = torch.tensor(input_ids[:max_length])
        # # model_inputs["labels"] = torch.tensor(labels["input_ids"][1:max_length] + [-100])
        # # model_inputs["labels"] = torch.tensor(input_ids[1:max_length] + [-100])


        max_length = q_max_length + a_max_length
        if '<answer>' not in data['input']:
            model_inputs = self.tokenizer(data['input'] + " <answer> " + data['output'] + " </answer>")
        else:
            model_inputs = self.tokenizer(data['input'] + " " + data['output'])
        model_inputs["input_ids"] = model_inputs["input_ids"] + [self.tokenizer.eos_token_id]
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        ## left padding
        orig_input_ids = model_inputs["input_ids"].copy()
        model_inputs["input_ids"] = [self.tokenizer.pad_token_id] * (max_length - len(model_inputs["input_ids"])) + orig_input_ids
        model_inputs["attention_mask"] = [0] * (max_length - len(model_inputs["attention_mask"])) + model_inputs["attention_mask"]
        model_inputs["labels"] = [-100] * (max_length - len(orig_input_ids)) + orig_input_ids

        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"][:max_length])
        model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"][:max_length])
        if model_inputs["input_ids"][-1] == self.tokenizer.eos_token_id:
            model_inputs["labels"] = torch.tensor(model_inputs["labels"][:max_length - 1] + [-100])
        else:
            model_inputs["labels"] = torch.tensor(model_inputs["labels"][:max_length])


        return model_inputs
