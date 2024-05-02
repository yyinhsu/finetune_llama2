from reporter import Reporter
from transformers import AutoModelForCausalLM, \
    get_linear_schedule_with_warmup, \
    AutoModelForCausalLM, \
    AutoTokenizer, \
    BitsAndBytesConfig, \
    TrainingArguments, \
    Trainer, \
    DefaultDataCollator
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import torch
import time
import os
from typing import Tuple
import huggingface_hub
import numpy as np
import json


## reference of fine-tuning llama2 with PEFT:
## https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19

class LoRAFP16Trainer:
    """The class for the prefix tuning experiment.

    prefix tuning is a method to fine-tune the pretrained model on a specific task.
    Only the prefix parameters are optimized and added to the hidden states in every layer of the model.
    The tokens of the input sequence can still attend to the prefix as virtual tokens.
    ref: https://huggingface.co/docs/peft/task_guides/seq2seq-prefix-tuning
    """
    def __init__(self,
                 train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 pretrained_model: str,
                 huggingface_hub_token: str,
                 lora_rank: int,
                 lora_alpha: int,
                 lora_dropout: float,
                 batch_size: int,
                 lr: float,
                 weight_decay: float,
                 num_epochs: int,
                 tokenizer: AutoTokenizer,
                 model_save_path: str,
                 reporter: Reporter):
        """Initialize the prefix tuning experiment.

        :param: torch.utils.data.DataLoader train_dataloader: the dataloader of the training dataset
        :param: torch.utils.data.DataLoader test_dataloader: the dataloader of the evaluation dataset
        :paran: str pretrained_model: the name of the pretrained model
        :param: str huggingface_hub_token: the credential token of the huggingface hub
        :param: int lora_rank: the rank of the low-rank attention matrix
        :param: int lora_alpha: the alpha parameter that controls how the low-rank attention matrix is mixed with the original attention matrix
        :param: float lora_dropout: the dropout rate of the low-rank attention matrix
        :param: float lr: the learning rate
        :param: float weight_decay: the weight decay
        :param: int num_epochs: the number of epochs of training
        :param: transformers.AutoTokenizer tokenizer: the tokenizer of the pretrained model
        :param: str model_save_path: the path to save the model
        :param: reporter: the reporter to record the training performance
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_dataset = train_dataset
        self.eval_dataset = test_dataset
        self.tokenizer = tokenizer
        self.pretrained_model = pretrained_model
        self.lora_params = {'lora_rank': lora_rank,
                            'lora_alpha': lora_alpha,
                            'lora_dropout': lora_dropout}

        self.model = self.__load_model(pretrained_model, huggingface_hub_token, tokenizer)

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        # self.optimizer = torch.optim.AdamW(self.model.parameters(),
        #                                    lr=self.lr,
        #                                    weight_decay=self.weight_decay)
        # self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
        #                                                     num_warmup_steps=0, # TODO: adjust this parameter to see the effect
        #                                                     num_training_steps=(len(self.train_dataloader) * self.num_epochs))

        self.model_save_path = model_save_path
        self.reporter = reporter


    def training(self) -> None:
        """Using transformers.Trainer to train the model.

        Because the model is load as 4-bit quantized model, the training process would be different from the original one.
        We cannot use the original pytorch code training process directly.
        transformers.Trainer would handle the quantization process automatically.
        The arguments of transformers.Trainer could be referenced here:
        https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
        The optimizer is set to paged_adamw_8bit, of which number of bits could be different from model's quantization config (4).

        :return: None
        """
        start_time = time.time()

        self.__record_training_hyper_params()

        self.model = self.model.to(self.device)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.model_save_path, 'output_dir/'), exist_ok=True)
        os.makedirs(os.path.join(self.model_save_path, 'output_dir/peft_model/'), exist_ok=True)
        os.makedirs(os.path.join(self.model_save_path, 'peft_model/'), exist_ok=True)

        print("Model Training started.....")

        trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=TrainingArguments(
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                num_train_epochs=self.num_epochs,
                learning_rate=self.lr,
                weight_decay=self.weight_decay,
                fp16=True,
                logging_steps=1,
                output_dir=os.path.join(self.model_save_path, 'output_dir/'),
                optim="paged_adamw_8bit"
            ),
            data_collator=DefaultDataCollator(return_tensors="pt"),
        )

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        # lora_file_path = os.path.join(self.model_save_path, 'peft_model/adapter_configg.bin')
        # torch.save(trainer.model.state_dict(), lora_file_path)
        trainer.model.save_pretrained(os.path.join(self.model_save_path, 'peft_model'))
        self.__produce_adapter_json()

        # self.reporter.write_log("Best performance epoch: %d" % best_performance_epoch)

        end_time = time.time()
        self.reporter.write_log("Training time: %s" % time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))


    def __load_model(self, model_name: str, huggingface_hub_token: str, tokenizer: AutoTokenizer) -> PeftModel:
        """Load the model and add the prefix tuning configuration to the model.

        :param: str model_name: the name of the pretrained model
        :param: str huggingface_hub_token: the credential token of the huggingface hub
        :param: transformers.AutoTokenizer tokenizer: the tokenizer of the pretrained model
        :return model: the model with the prefix tuning configuration
        :rtype model: transformers.modeling_utils.PreTrainedModel
        """
        ## ref: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/training/run_clm_pt_with_peft.py
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['v_proj', 'q_proj'], # 'k_proj', 'v_proj', 'q_proj'
            inference_mode=False,
            r=self.lora_params['lora_rank'],
            lora_alpha=self.lora_params['lora_alpha'],
            lora_dropout=self.lora_params['lora_dropout'],
            # modules_to_save=['embed_tokens', 'lm_head']
        )
        if huggingface_hub_token != 'login_not_required':
            huggingface_hub.login(token=huggingface_hub_token)
        # model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

        ## quantization config
        ## ref: https://huggingface.co/blog/hf-bitsandbytes-integration
        # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=a9EUEDAl0ss3
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        model.resize_token_embeddings(len(tokenizer))

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model


    def __save_model(self, path) -> None:
        """Save the model and the tokenizer.

        The tuned model would be saved in the folder: path/peft_model
        The tokenizer would be saved in the folder: path/tokenizer

        :param: str path: the virtual path to save the model and the tokenizer
        :return: None
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, 'peft_model'))
        # self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))


    def __produce_adapter_json(self):
        """Produce the adapter json file.

        Example:
        {
            "auto_mapping": null,
            "base_model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
            "bias": "none",
            "fan_in_fan_out": false,
            "inference_mode": true,
            "init_lora_weights": true,
            "layers_pattern": null,
            "layers_to_transform": null,
            "lora_alpha": 8,
            "lora_dropout": 0.01,
            "modules_to_save": null,
            "peft_type": "LORA",
            "r": 8,
            "revision": null,
            "target_modules": [
                "v_proj",
                "q_proj"
            ],
            "task_type": "CAUSAL_LM"
        }
        """
        json_dict = {}
        json_dict['auto_maping'] = None
        json_dict['base_model_name_or_path'] = self.pretrained_model
        json_dict['bias'] = 'none'
        json_dict['fan_in_fan_out'] = False
        json_dict['inference_mode'] = True
        json_dict['init_lora_weights'] = True
        json_dict['layers_pattern'] = None
        json_dict['layers_to_transform'] = None
        json_dict['lora_alpha'] = self.lora_params['lora_alpha']
        json_dict['lora_dropout'] = self.lora_params['lora_dropout']
        json_dict['modules_to_save'] = None
        json_dict['peft_type'] = 'LORA'
        json_dict['r'] = self.lora_params['lora_rank']
        json_dict['revision'] = None
        json_dict['target_modules'] = ['v_proj', 'q_proj']
        json_dict['task_type'] = 'CAUSAL_LM'

        open(os.path.join(self.model_save_path, 'peft_model/adapter_configg.json'), 'w').write(json.dumps(json_dict))


    def __record_training_hyper_params(self) -> None:
        """Record the hyper parameters of the training.

        :return: None
        """
        input_max_length = self.train_dataset.q_max_length
        output_max_length = self.train_dataset.a_max_length
        text = "lr: %f\tweight_decay: %f\tnum_epochs: %d\tbatch_size: %d\tinput_max_length: %d\toutput_max_length: %d" \
            % (self.lr, self.weight_decay, self.num_epochs, self.batch_size, input_max_length, output_max_length)
        self.reporter.write_log(text)


if __name__ == '__main__':
    pass