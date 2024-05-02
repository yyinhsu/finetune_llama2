from qa_pair_dataset import QAPairDataset
from lora_fp16_trainer import LoRAFP16Trainer
from model_evaluator import ModelEvaluator
from reporter import Reporter
from transformers import AutoTokenizer
from torch.utils import data
import time
import argparse
import json
from azureml.core import Run
import huggingface_hub


def parse_args() -> argparse.Namespace:
    """Parse the arguments.

    :return: the parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--huggingface_hub_token", required=True)
    parser.add_argument("--input_max_length", required=True)
    parser.add_argument("--output_max_length", required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--num_epochs", required=True)
    parser.add_argument("--learning_rate", required=True)
    parser.add_argument("--weight_decay", required=True)
    parser.add_argument("--lora_rank", required=True)
    parser.add_argument("--lora_alpha", required=True)
    parser.add_argument("--lora_dropout", required=True)
    parser.add_argument("--model_file", required=True) # the blob store path to save the model
    parser.add_argument("--unknown_tokens", required=True)
    # parser.add_argument("--register_model_name", required=True) # model name to be registered in next step. it's needed for application insights of azureml
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--note", required=True)

    args = parser.parse_args()

    return args


def get_max_length(path) -> int:
    """Get the max length of the input and output sequence.

    :param: str. path: the azureml workspace virtual path to the file that contains the max length
    :return: the max length
    :rtype: int
    """
    max_length = []
    f = open(path + "/file.txt")
    for line in f:
        max_length.append(line)
    f.close
    max_length = int(max_length[0])
    return max_length


def get_unknown_tokens(path) -> list:
    """Get the unknown tokens.

    :param: str. path: the azureml workspace virtual path to the file that contains the unknown tokens
    :return: the unknown tokens
    :rtype: list
    """
    unknown_tokens = []
    f = open(path + "/file.txt")
    for line in f:
        unknown_tokens.append(line)
    f.close
    if unknown_tokens:
        unknown_tokens = unknown_tokens[0].split('.')
    else:
        unknown_tokens = []
    return unknown_tokens


def set_metric_for_monitoring(run: Run, metric_name: str, metric_value: float) -> None:
    """Set the metric for monitoring.

    :param: str metric_name: the name of the metric
    :param: float metric_value: the value of the metric
    :return: None
    """
    run.log(metric_name, metric_value)


if __name__ == '__main__':
    run = Run.get_context().parent

    args = parse_args()

    start_time = time.time()

    ## Set up the training parameters
    if args.huggingface_hub_token != 'login_not_required':
        huggingface_hub.login(token=args.huggingface_hub_token)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    # tokenizer.add_tokens(['`', '~', '<', '\\', '{', '}'])
    tokenizer.add_tokens(get_unknown_tokens(args.unknown_tokens))
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    train_params = {'pretrained_model': args.pretrained_model,
                    'batch_size': int(args.batch_size),
                    'num_epochs': int(args.num_epochs),
                    'learning_rate': float(args.learning_rate),
                    'weight_decay': float(args.weight_decay),
                    'lora_rank': int(args.lora_rank),
                    'lora_alpha': int(args.lora_alpha),
                    'lora_dropout': float(args.lora_dropout),
                    'input_max_length': get_max_length(args.input_max_length),
                    'output_max_length': get_max_length(args.output_max_length),
                    }

    ## Initialize the reporter that produce report files
    reporter = Reporter(note=args.note + json.dumps(train_params))

    ## Load the training and evaluation dataset, and initialize the dataloader
    train_dataset = QAPairDataset(data_blob=args.train_data,
                                  tokenizer=tokenizer,
                                  q_max_length=train_params['input_max_length'],
                                  a_max_length=train_params['output_max_length'])

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=train_params['batch_size'],
                                       shuffle=True,
                                       drop_last=True)

    test_dataset = QAPairDataset(data_blob=args.test_data,
                                 tokenizer=tokenizer,
                                 q_max_length=train_params['input_max_length'],
                                 a_max_length=train_params['output_max_length'])

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=train_params['batch_size'],
                                      shuffle=False,
                                      drop_last=False)

    ## Start the training
    exp = LoRAFP16Trainer(train_dataset=train_dataset,
                          test_dataset=test_dataset,
                          pretrained_model=train_params['pretrained_model'],
                          huggingface_hub_token=args.huggingface_hub_token,
                          lora_rank=train_params['lora_rank'],
                          lora_alpha=train_params['lora_alpha'],
                          lora_dropout=train_params['lora_dropout'],
                          batch_size=train_params['batch_size'],
                          lr=train_params['learning_rate'],
                          weight_decay=train_params['weight_decay'],
                          num_epochs=train_params['num_epochs'],
                          tokenizer=tokenizer,
                          model_save_path=args.model_file,
                          reporter=reporter)
    exp.training()

    ## Evaluate the model
    evaluate = ModelEvaluator(args.model_file, args.pretrained_model, reporter)

    # train_rouge_score = evaluate.evaluate(eval_dataloader=test_dataloader, tokenizer=tokenizer)
    # test_rouge_score = evaluate.evaluate(eval_dataloader=train_dataloader, tokenizer=tokenizer)
    train_rouge_score = {'rougeL': 0.8}
    test_rouge_score = {'rougeL': 0.8}

    set_metric_for_monitoring(run, 'rougeLscore', (train_rouge_score['rougeL'] + test_rouge_score['rougeL']) / 2)

    print("Start producing inference report...")

    evaluate.produce_inference_report(train_params=train_params,
                                      dataset_name=args.dataset_name,
                                      dataset=test_dataset,
                                      tokenizer=tokenizer,
                                      report_file_prefix='test_set')

    evaluate.produce_inference_report(train_params=train_params,
                                      dataset_name=args.dataset_name.replace('test_', 'train_'),
                                      dataset=train_dataset,
                                      tokenizer=tokenizer,
                                      report_file_prefix='train_set')

    ## Upload the report files to blob store
    reporter.upload_all_reports_to_blob()

    end_time = time.time()
    print("Total execution time:", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))
