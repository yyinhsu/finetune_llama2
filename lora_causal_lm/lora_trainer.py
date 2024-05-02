import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"
from lora_causal_lm.reporter import Reporter
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import time
import os
from typing import Tuple
import huggingface_hub
import numpy as np

## reference of fine-tuning llama2 with PEFT:
## https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19

class LoRATrainer:
    """The class for the prefix tuning experiment.

    prefix tuning is a method to fine-tune the pretrained model on a specific task.
    Only the prefix parameters are optimized and added to the hidden states in every layer of the model.
    The tokens of the input sequence can still attend to the prefix as virtual tokens.
    ref: https://huggingface.co/docs/peft/task_guides/seq2seq-prefix-tuning
    """
    def __init__(self,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 pretrained_model: str,
                 huggingface_hub_token: str,
                 lora_rank: int,
                 lora_alpha: int,
                 lora_dropout: float,
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

        self.train_dataloader = train_dataloader
        self.eval_dataloader = test_dataloader
        self.tokenizer = tokenizer

        self.model = self.__load_model(pretrained_model, huggingface_hub_token, lora_rank, lora_alpha, lora_dropout, tokenizer)
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=0, # TODO: adjust this parameter to see the effect
                                                            num_training_steps=(len(self.train_dataloader) * self.num_epochs))

        self.model_save_path = model_save_path
        self.reporter = reporter


    def training(self, best_performance_selector='accuracy') -> None:
        """Train the model on the training dataset, and present the training loss and accuracy.

        During the training, the model with the best performance on the evaluation dataset would be saved.
        The performance (loss and accuracy) is printed and recorded in training_record.txt once an epoch.
        training_record.txt is saved in blob store.

        :param: str. best_performance_selector: the decision critirion to select the model with the best performance.
                                                must be 'accuracy', 'loss', or 'last_epoch'
        :return: None
        """
        start_time = time.time()

        self.__record_training_hyper_params()

        self.model = self.model.to(self.device)
        print("Model Training started.....")

        ## Initialize the best performance metric, and the epoch of the best performance
        best_performance_epoch = -1
        if best_performance_selector == 'accuracy':
            best_performance_metric = -1
        elif best_performance_selector == 'loss':
            best_performance_metric = 100000
        elif best_performance_selector == 'last_epoch':
            best_performance_metric = self.num_epochs
        else:
            raise Exception("best_performance_selector must be 'accuracy', 'loss', or 'last_epoch'")

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            for _, batch in enumerate(self.train_dataloader):
                print("@@", batch)
                batch['input_ids'] = batch['input_ids'].to(self.device)
                batch['attention_mask'] = batch['attention_mask'].to(self.device)
                batch['labels'] = batch['labels'].to(self.device)

                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     labels=batch['labels'])

                loss =  outputs.loss
                epoch_loss = epoch_loss + loss.detach().float()
                loss = torch.nan_to_num(loss, nan=1e-7)
                loss.backward() # compute gradient
                self.optimizer.step() # update parameters through gradient
                self.lr_scheduler.step() # update learning rate
                self.optimizer.zero_grad() # clean gradient as zero for next step because gradient is accumulated

            eval_epoch_loss, eval_epoch_accuracy = self.__get_performance(self.eval_dataloader)
            train_epoch_loss, train_epoch_accuracy = self.__get_performance(self.train_dataloader)

            # print("epoch: %d\ttrain loss: %.4f\ttrain accuracy: %.4f\teval loss: %.4f\teval accuracy: %.4f" \
            #     % (epoch, train_epoch_loss, train_epoch_accuracy, eval_epoch_loss, eval_epoch_accuracy))
            self.reporter.write_training_performance(epoch, train_epoch_loss, train_epoch_accuracy, eval_epoch_loss, eval_epoch_accuracy)

            ## save the model with the best performance
            avg_epoch_accuracy = train_epoch_accuracy
            if best_performance_selector == 'accuracy':
                if avg_epoch_accuracy >= best_performance_metric:
                    best_performance_metric = avg_epoch_accuracy
                    self.__save_model(self.model_save_path)
                    best_performance_epoch = epoch
            elif best_performance_selector == 'loss':
                if eval_epoch_loss <= best_performance_metric:
                    best_performance_metric = eval_epoch_loss
                    self.__save_model(self.model_save_path)
                    best_performance_epoch = epoch
            elif best_performance_selector == 'last_epoch':
                if epoch == self.num_epochs - 1:
                    self.__save_model(self.model_save_path)

        self.reporter.write_log("Best performance epoch: %d" % best_performance_epoch)

        end_time = time.time()
        self.reporter.write_log("Training time: %s" % time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))


    def __get_performance(self, dataloader) -> Tuple[float, float]:
        """In order to evaluate model performance on training set once an epoch, get the predictions of the training dataset.

        :param: torch.utils.data.DataLoader dataloader: the dataloader of the training or evaluating dataset
        :return: loss: the loss and accuracy of the model on the dataset
        :rtype: loss: float
        :return: accuracy: the accuracy of the model on the dataset
        :rtype: accuracy: float
        """
        self.model.eval()
        loss = 0
        batch_accs = []
        for _, batch in enumerate(dataloader):
            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['attention_mask'] = batch['attention_mask'].to(self.device)
            batch['labels'] = batch['labels'].to(self.device)
            with torch.no_grad(): # the container of gradient is not needed
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     labels=batch['labels'])

            loss += outputs.loss.detach().float()
            preds = torch.argmax(outputs.logits, -1)

            ## because prediction of casual language model is a sequence contains both question and answer,
            ## we use the whole sequence includes question and answer to calculate accuracy (exclude the padding token)
            ## eos token is set as llama2 tokenizer's pad token, so it should be removed from the sequence
            preds = preds.detach().cpu().numpy()
            ground_truth = batch['labels']
            ground_truth = ground_truth.cpu().numpy()
            num_correct_batch = 0
            num_token_batch = 0

            for i in range(preds.shape[0]):
                pred = self.extract_meaningful_tokens(preds[i])
                truth = self.extract_meaningful_tokens(ground_truth[i])

                if pred.shape[0] > truth.shape[0]:
                    pred = pred[:truth.shape[0]]
                elif pred.shape[0] < truth.shape[0]:
                    truth = truth[:pred.shape[0]]

                num_correct_batch += np.sum(pred == truth)
                num_token_batch += truth.shape[0]

            if num_token_batch == 0:
                batch_accs.append(0)
            else:
                batch_accs.append(num_correct_batch / num_token_batch)

        loss = (loss / len(dataloader)).detach()
        accuracy = sum(batch_accs) / len(batch_accs)

        return loss, accuracy


    def extract_meaningful_tokens(self, token_array: np.ndarray) -> np.ndarray:
        """Extract the meaningful tokens from the token array.

        The meaningful tokens are the tokens that are not padding tokens and not eos tokens.

        :param: np.ndarray token_array: the token array to be processed
        :return: meaningful_tokens: the meaningful tokens
        :rtype: meaningful_tokens: np.ndarray
        """
        meaningful_tokens = []
        for token in token_array:
            if token != self.tokenizer.pad_token_id and token != self.tokenizer.eos_token_id:
                meaningful_tokens.append(token)

        try:
            token_25580_index = np.where(token_array == 25580)[0][0]
            token_29962_index = np.where(token_array == 29962)[0][0]
        except: # if the token is not in the token array, let meaningful_tokens = token_array
            token_25580_index = 0
            token_29962_index = 0

        ## only if 25580 is before 29962, we are sure that 25580 is the start token of the answer
        ## else we still let meaningful_tokens = token_array, so that it will get very low accuracy
        if token_25580_index + 1 == token_29962_index:
            meaningful_tokens = meaningful_tokens[token_25580_index:]

        return np.array(meaningful_tokens)


    def __load_model(self, model_name, huggingface_hub_token, lora_rank, lora_alpha, lora_dropout, tokenizer) -> PeftModel:
        """Load the model and add the prefix tuning configuration to the model.

        :param: model_name: the name of the pretrained model
        :param: lora_rank: the rank of the low-rank attention matrix
        :param: lora_alpha: the alpha parameter that controls how the low-rank attention matrix is mixed with the original attention matrix
        :param: lora_dropout: the dropout rate of the low-rank attention matrix
        :param: transformers.tokenization_utils_base.PreTrainedTokenizer tokenizer: the tokenizer of the pretrained model
        :return model: the model with the prefix tuning configuration
        :rtype model: transformers.modeling_utils.PreTrainedModel
        """
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        huggingface_hub.login(token=huggingface_hub_token)
        model = AutoModelForCausalLM.from_pretrained(model_name) #, torch_dtype="auto")
        model.resize_token_embeddings(len(tokenizer))

        ## ref: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/training/run_clm_pt_with_peft.py
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['v_proj', 'q_proj'], # 'k_proj', 'v_proj', 'q_proj'
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # modules_to_save=['embed_tokens', 'lm_head']
        )
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


    def __record_training_hyper_params(self) -> None:
        """Record the hyper parameters of the training.

        :return: None
        """
        input_max_length = self.train_dataloader.dataset.q_max_length
        output_max_length = self.train_dataloader.dataset.a_max_length
        text = "lr: %f\tweight_decay: %f\tnum_epochs: %d\tbatch_size: %d\tinput_max_length: %d\toutput_max_length: %d" \
            % (self.lr, self.weight_decay, self.num_epochs, self.train_dataloader.batch_size, input_max_length, output_max_length)
        self.reporter.write_log(text)


if __name__ == '__main__':
    pass