from lora_causal_lm.reporter import Reporter
from lora_causal_lm.qa_pair_dataset import QAPairDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from gpt_utils.openai_connection_handler import WistronGPTConnectionHandler
from gpt_utils.gpt_helper import GPTHelper
import torch
from rouge_score import rouge_scorer
import os
import re
from typing import Tuple


class GPTEvaluator():
    """This class is used to evaluate the model performance through asking GPT how similar between the predictions and groud truths on the evaluation dataset."""
    def __init__(self, chat_model: str="gpt-4"):
        self.openai_handler = WistronGPTConnectionHandler(chat_model=chat_model)
        self.prompt = """Please read the following two texts, give a semantic similarity score from 0 to 1, and explain the reason for the score. The format should be like: {"score": <score>, "reason": <reason>}\n\nTEXT_1: ###PREDICTIONTEXT###\n\nTEXT_2: ###GROUNDTRUTHTEXT###"""

    def evaluate(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Evaluate the model performance through asking GPT how similar between the predictions and groud truths on the evaluation dataset.
        Args:
            prediction (str): the prediction text
            ground_truth (str): the ground truth text
        Returns:
            score (float): the similarity score from 0 to 1. if the score is -1.0, it means the evaluation is failed.
            reason (str): the reason for the score
        """
        score = -1.0
        reason = "Fail to get the similarity score."

        prompt = self.prompt.replace("###PREDICTIONTEXT###", prediction).replace("###GROUNDTRUTHTEXT###", ground_truth)
        response = self.openai_handler.call_wistron_gpt_chat(GPTHelper.construct_gpt_input(prompt))
        print("gpt response: ", response)

        pattern = "\{.*\}"
        found_text = re.findall(pattern, response)
        if len(found_text) > 0:
            try:
                response_dict = eval(found_text[0])
                score = response_dict['score']
                reason = response_dict['reason']
            except:
                pass
        return score, reason


class RougeScoreCalculator():
    """This class is used to calculate the metric ROUGE score of the model performance."""
    def get_avg_rouge_score(preds: list, ground_truths: list) -> dict:
        """Calculate the average ROUGE score of the model performance.

        :param: list preds: the list of the predictions
        :param: list ground_truths: the list of the ground truths
        :return: the average ROUGE score invloving rouge1, rouge2, rougeL, rougeLsum of the model performance
        :rtype: dict
        """
        ## Set up rouge scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        ## Calculate rouge score and store the scores in score_sum_dict
        score_sum_dict = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
        for pred, truth in zip(preds, ground_truths):
            scores = scorer.score(pred, truth)
            for key in score_sum_dict.keys():
                score_sum_dict[key] += scores[key].fmeasure
        ## Calculate average scores
        for key in score_sum_dict.keys():
            score_sum_dict[key] /= len(preds)

        return score_sum_dict


    def get_rouge_score(prediction: str, ground_truth: str, method: list=['rouge1', 'rouge2', 'rougeL', 'rougeLsum']) -> dict:
        """
        Calculate the ROUGE score of prediction text and ground truth text.
        Args:
            prediction (str): the prediction text
            ground_truth (str): the ground truth text
            method (list): the list of the ROUGE methods, including 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
        Returns:
            dict: the ROUGE score of the model performance. for example: {'rouge1': 0.5, 'rouge2': 0.3, 'rougeL': 0.4, 'rougeLsum': 0.4}
        """
        scorer = rouge_scorer.RougeScorer(method, use_stemmer=True)
        score = scorer.score(prediction, ground_truth)
        score_dict = {k: score[k].fmeasure for k in method}
        return score_dict


class ModelEvaluator():
    """Evaluate the model on the evaluation dataset, and present the evaluation metrics."""
    def __init__(self, peft_model_folder: str, tokenizer: AutoTokenizer, reporter: Reporter) -> None:
        """
        Initialize the model evaluator.
        Args:
            peft_model_folder (str): the folder name of the prefix tuning model
            tokenizer (AutoTokenizer): the tokenizer
            reporter (Reporter): the reporter to record the model performance
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_trained_model(peft_model_folder)
        self.model_folder = peft_model_folder
        self.tokenizer = tokenizer
        self.reporter = reporter


    def load_trained_model(self, peft_model_folder: str) -> PeftModel:
        """
        Load the model and add the prefix tuning configuration to the model.
        Args:
            peft_model_folder (str): the folder name of the prefix tuning model
        Returns:
            PeftModel: the model with the prefix tuning configuration
        """
        model_path = os.path.join(peft_model_folder, 'peft_model')
        peft_config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_id=model_path) # model_id: A path to a directory containing a Lora configuration file saved
        return model


    def produce_inference_report(self,
                                 train_params: dict,
                                 dataset: QAPairDataset,
                                 eval_data_keys: list=['root_cause', 'solution'],
                                 report_file_prefix: str='inference') -> None:

        """
        Produce the inference report of the model on the evaluation dataset.
        This report would be saved as a json file, which includes:
        1. the training parameters
        2. the average ROUGE score on the evaluation dataset
        3. the similarity score evaluating of the evaluation dataset,
        4. the predictions and true labels of data.
        Args:
            train_params (dict): the training parameters
            dataset (QAPairsDataset): the evaluation dataset object
            report_file_prefix (str): the prefix of the report file name
        """
        # <answer>
        # {'issue_description': 'OD3_MTL_H28_DVT1_PER: MM25 result is fail(FR:5/5 Units)',
        # 'function_team': 'Validation',
        # 'root_cause': 'We can get results with OS .2361.',
        # 'solution': 'We can get results with OS .2361.'}
        # </answer> </s>

        report_json_dict = dict()
        metric = {'avg_rougeL_score': 0, 'max_rougeL_score': 0, 'min_rougeL_score': 1,
                  'avg_gpt_eval_score': 0, 'max_gpt_eval_score': 0, 'min_gpt_eval_score': 1}
        report_json_dict.update({'model_name': self.model_folder,
                                 'train_params': train_params})
        report_json_dict.update({k: metric.copy() for k in eval_data_keys})
        report_json_dict.update({'data': []})


        inputs = [data['input'] for data in dataset.data_dicts]
        ground_truths = [data['output'] for data in dataset.data_dicts]

        predictions = self.__inference_eval_data_text(input_text_list=inputs,
                                                      tokenizer=self.tokenizer,
                                                      input_max_length=dataset.a_max_length,
                                                      output_max_length=dataset.a_max_length)

        rougeL_scores = {k: [] for k in eval_data_keys}
        gpt_eval_scores = {k: [] for k in eval_data_keys}
        gpt_evaluator = GPTEvaluator()

        for enum, (input_text, pred_text, truth_text) in enumerate(zip(inputs, predictions, ground_truths)):
            pattern = '\{.*\}'

            pred = {'raw_text': pred_text}
            found_text = re.findall(pattern, pred_text)
            if len(found_text) > 0:
                try:
                    pred.update(eval(found_text[0]))
                except:
                    pass

            truth = {'raw_text': truth_text}
            found_text = re.findall(pattern, truth_text)
            if len(found_text) > 0:
                try:
                    truth.update(eval(found_text[0]))
                except:
                    pass

            data_dict = {'sequential_id': enum,
                         'input': input_text,
                         'ground_truth': truth_text,
                         'prediction': pred_text,
                         'rouge_score': {k: -1 for k in eval_data_keys},
                         'gpt_eval_score': {k: -1 for k in eval_data_keys},
                         'gpt_eval_reason': {k: "" for k in eval_data_keys}}

            for key in eval_data_keys + ['raw_text']:
                if key in pred and key in truth:
                    rouge_score = RougeScoreCalculator.get_rouge_score(pred[key], truth[key], method=['rougeL'])['rougeL']
                    gpt_eval_score, gpt_eval_reason = gpt_evaluator.evaluate(pred[key], truth[key])

                    rougeL_scores[key].append(rouge_score)
                    gpt_eval_scores[key].append(gpt_eval_score)

                    data_dict['rouge_score'][key] = rouge_score
                    data_dict['gpt_eval_score'][key] = gpt_eval_score
                    data_dict['gpt_eval_reason'][key] = gpt_eval_reason
                else:
                    rougeL_scores[key].append(-1)
                    gpt_eval_scores[key].append(-1)

            report_json_dict['data'].append(data_dict)

        for key in eval_data_keys:
            each_rouges = [score for score in rougeL_scores[key] if score != -1]
            each_gpt_evals = [score for score in gpt_eval_scores[key] if score != -1]

            report_json_dict[key]['avg_rougeL_score'] = sum(each_rouges) / len(each_rouges)
            report_json_dict[key]['max_rougeL_score'] = max(each_rouges)
            report_json_dict[key]['min_rougeL_score'] = min(each_rouges)

            report_json_dict[key]['avg_gpt_eval_score'] = sum(each_gpt_evals) / len(each_gpt_evals)
            report_json_dict[key]['max_gpt_eval_score'] = max(each_gpt_evals)
            report_json_dict[key]['min_gpt_eval_score'] = min(each_gpt_evals)

        self.reporter.write_inference_report_to_json(report_json_dict, report_file_prefix)


    def __decode_with_dataloader(self, dataloader: torch.utils.data.DataLoader, tokenizer: AutoTokenizer) -> list:
        """
        Decode the labels which is the ground truth of the data in the dataloader.
        The labels are decoded by the tokenizer.
        The reason why we need to decode the labels is that we have to compute metric ROUGE score.
        If we use the original text from the dataset, the special tokens and unrefined syntax would affect the ROUGE score.
        Args:
            dataloader (torch.utils.data.DataLoader): the dataloader of the training or evaluating dataset
            tokenizer (AutoTokenizer): the tokenizer
        Returns:
            list: the decoded texts of the labels
        """
        decoded_texts = []
        for _, batch in enumerate(dataloader):
            if 1 in batch['labels'].shape: # batch_size = 1 or the last batch
                to_be_decoded = [token for token in batch['labels'].squeeze().detach().tolist() if token != -100]
                decoded = tokenizer.decode(to_be_decoded, skip_special_tokens=True)
                decoded_texts.append(decoded)
            else:
                for pred in batch['labels'].squeeze().detach().tolist():
                    to_be_decoded = [token for token in pred if token != -100]
                    decoded = tokenizer.decode(to_be_decoded, skip_special_tokens=True)
                    decoded_texts.append(decoded)

        return decoded_texts


    def __inference_eval_data_text(self, input_text_list: list, input_max_length: int, output_max_length: int) -> list:
        """
        Inference the model on the evaluation dataset.
        Generation config is set to max_new_tokens, repetition_penalty=1.2.
        ref: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        Args:
            input_text_list (list): the list of the input text
            input_max_length (int): the maximum length of the input text
            output_max_length (int): the maximum length of the output text
        Returns:
            list: the decoded predictions of the evaluation dataset
        """
        self.model.to(self.device)

        predictions = []
        for text in input_text_list:
            tokenized_inputs = self.tokenizer(text + "<answer>", return_tensors="pt")

            tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'].to(self.device)
            tokenized_inputs['attention_mask'] = tokenized_inputs['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(input_ids=tokenized_inputs['input_ids'],
                                            #   attention_mask=tokenized_inputs['attention_mask'],
                                              max_new_tokens=input_max_length + output_max_length)
            predictions.append(outputs.detach().cpu().numpy().tolist()[0])

        decoded_predictions = []
        for pred in predictions:
            to_be_decoded = [token for token in pred if token != -100]
            decoded = self.tokenizer.decode(to_be_decoded, skip_special_tokens=True)
            try:
                pattern = "\s<answer>\s(.*?)\s</answer>"
                decoded = re.findall(pattern, decoded)[0]
            except:
                pass
            decoded_predictions.append(decoded)

        return decoded_predictions


    def __inference_eval_dataloader(self, eval_dataloader: torch.utils.data.DataLoader) -> list:
        """
        Inference the model on the evaluation dataset.
        Generation config is set to max_new_tokens, repetition_penalty=1.2.
        ref: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        Args:
            eval_dataloader (torch.utils.data.DataLoader): the evaluation dataloader
        Returns:
            list: the decoded predictions of the evaluation dataset
        """
        self.model.to(self.device)

        predictions = []
        for _, batch in enumerate(eval_dataloader): #TODO: one by one, need padding and truncate
            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['attention_mask'] = batch['attention_mask'].to(self.device)
            # max_new_tokens = batch['labels'].detach().cpu().numpy().shape[-1]
            max_new_tokens = eval_dataloader.dataset.a_max_length
            with torch.no_grad():
                outputs = self.model.generate(input_ids=batch['input_ids'],
                                            #   attention_mask=batch['attention_mask'],
                                              max_new_tokens=max_new_tokens,
                                            #   pad_token_id=tokenizer.eos_token_id
                                              )
            predictions.extend(outputs.detach().cpu().numpy().tolist())

        decoded_predictions = []
        for pred in predictions:
            to_be_decoded = [token for token in pred if token != -100]
            decoded = self.tokenizer.decode(to_be_decoded, skip_special_tokens=True)
            decoded_predictions.append(decoded)

        return decoded_predictions


if __name__ == '__main__':
    pass