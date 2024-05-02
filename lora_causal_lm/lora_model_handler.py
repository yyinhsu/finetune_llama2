from lora_causal_lm.qa_pair_dataset import QAPairDataset
from lora_causal_lm.lora_trainer import LoRATrainer
from lora_causal_lm.model_evaluator import ModelEvaluator
from lora_causal_lm.reporter import Reporter
from transformers import AutoTokenizer
from torch.utils import data
import time
import json
import huggingface_hub


class LoraModelHandler:
    def __init__(self, train_data: str, test_data: str,
                 pretrained_model: str, huggingface_hub_token: str,
                 input_max_length: int, output_max_length: int, unknown_tokens: list,
                 batch_size: int, num_epochs: int, learning_rate: float, weight_decay: float,
                 lora_rank: int, lora_alpha: int, lora_dropout: float,
                 finetuned_model_file: str,
                 note: str):
        self.train_data = train_data
        self.test_data = test_data

        self.pretrained_model = pretrained_model
        self.huggingface_hub_token = huggingface_hub_token

        self.input_max_length = input_max_length
        self.output_max_length = output_max_length

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if "_tstmp_" in finetuned_model_file:
            self.finetuned_model_file = finetuned_model_file
        else:
            self.finetuned_model_file = finetuned_model_file + f"_tstmp_{time.strftime('%Y%m%d%H%M%S')}"
        self.note = note
        if isinstance(unknown_tokens, str):
            self.unknown_tokens = unknown_tokens.split(',')
        elif isinstance(unknown_tokens, list):
            self.unknown_tokens = unknown_tokens
        else:
            self.unknown_tokens = list()

        self.tokenizer = self.setup_tokenizer()

        self.reporter = Reporter(note=self.note + json.dumps(self.get_training_parameters()))


    def train_model(self):
        train_dataloader, test_dataloader = self.build_dataset()

        exp = LoRATrainer(train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          pretrained_model=self.pretrained_model,
                          huggingface_hub_token=self.huggingface_hub_token,
                          lora_rank=self.lora_rank,
                          lora_alpha=self.lora_alpha,
                          lora_dropout=self.lora_dropout,
                          lr=self.learning_rate,
                          weight_decay=self.weight_decay,
                          num_epochs=self.num_epochs,
                          tokenizer=self.tokenizer,
                          model_save_path=self.finetuned_model_file,
                          reporter=self.reporter)
        exp.training()


    def inference(self, input_path: str, report_file_prefix: str='inference_set'):
        """
        Inference the model on the input data and produce the inference report.
        Args:
            input_path (str): the path of the input data
            report_file_prefix (str): the prefix of the report file name
        """
        if ".jsonl" in input_path:
            with open(input_path, 'r') as f:
                input_data = f.readlines()
            input_data = [json.loads(data) for data in input_data]
        elif ".json" in input_path:
            with open(input_path, 'r') as f:
                input_data = json.load(f)["data"]
        else:
            raise ValueError("The data format is not supported. Only support jsonl and json format.")

        dataset = QAPairDataset(data=input_data,
                                tokenizer=self.tokenizer,
                                q_max_length=self.input_max_length,
                                a_max_length=self.output_max_length)
        evaluate = ModelEvaluator(peft_model_folder=self.finetuned_model_file,
                                  tokenizer=self.tokenizer,
                                  reporter=self.reporter)
        evaluate.produce_inference_report(train_params=self.get_training_parameters(),
                                          dataset=dataset,
                                          report_file_prefix=report_file_prefix)

        self.reporter.upload_all_reports_to_blob()


    def build_dataset(self):
        train_dataset = QAPairDataset(data=self.train_data,
                                      tokenizer=self.tokenizer,
                                      q_max_length=self.input_max_length,
                                      a_max_length=self.output_max_length)

        train_dataloader = data.DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=QAPairDataset.collate_fn)

        test_dataset = QAPairDataset(data=self.test_data,
                                     tokenizer=self.tokenizer,
                                     q_max_length=self.input_max_length,
                                     a_max_length=self.output_max_length)

        test_dataloader = data.DataLoader(dataset=test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          collate_fn=QAPairDataset.collate_fn)
        return train_dataloader, test_dataloader


    def setup_tokenizer(self):
        huggingface_hub.login(token=self.huggingface_hub_token)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        tokenizer.add_tokens(self.unknown_tokens)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        return tokenizer


    def get_training_parameters(self):
        return {
            'pretrained_model': self.pretrained_model,
            'huggingface_hub_token': self.huggingface_hub_token,
            'input_max_length': self.input_max_length,
            'output_max_length': self.output_max_length,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'model_file': self.finetuned_model_file,
            'note': self.note,
            'unknown_tokens': self.unknown_tokens
        }



if __name__ == '__main__':
    pass
