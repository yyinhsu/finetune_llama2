import json
import argparse
from configparser import ConfigParser, ExtendedInterpolation
from prepare_data.prepare_data import DataStatsProvider
from lora_causal_lm.lora_model_handler import LoraModelHandler


def get_arguments() -> str:
    """Get the arguments from the command line.
    cfg_file: the path of the config file. the file should be written in the format of configparser.

    Return:
        argparse.Namespace: The arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", '-C', required=False, default='./cfg_files/train_lora_llama2_v1.cfg')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_arguments()
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.cfg_file)

    data_stats = DataStatsProvider(train_data_path=config["DATASET"]["train_dataset_path"],
                                   test_data_path=config["DATASET"]["test_dataset_path"],
                                   pretrained_model=config["PRETRAINED_MODEL"]["pretrained_model"],
                                   huggingface_hub_token=config["PRETRAINED_MODEL"]["huggingface_hub_token"])

    lora_model_handler = LoraModelHandler(train_data=data_stats.train_data,
                                          test_data=data_stats.test_data,
                                          pretrained_model=config["PRETRAINED_MODEL"]["pretrained_model"],
                                          huggingface_hub_token=config["PRETRAINED_MODEL"]["huggingface_hub_token"],
                                          input_max_length=data_stats.input_max_length,
                                          output_max_length=data_stats.output_max_length,
                                          unknown_tokens=data_stats.unknown_tokens,
                                          batch_size=int(config["TRAINING"]["batch_size"]),
                                          num_epochs=int(config["TRAINING"]["num_epochs"]),
                                          learning_rate=float(config["TRAINING"]["learning_rate"]),
                                          weight_decay=float(config["TRAINING"]["weight_decay"]),
                                          lora_rank=int(config["TRAINING"]["lora_rank"]),
                                          lora_alpha=int(config["TRAINING"]["lora_alpha"]),
                                          lora_dropout=float(config["TRAINING"]["lora_dropout"]),
                                          finetuned_model_file=config["TRAINING"]["finetuned_model_prefix"],
                                          note=config["TRAINING"]["note"])

    lora_model_handler.train_model()
    lora_model_handler.inference(input_path=config["INFERENCE"]["input_path_1"],
                                 report_file_prefix='inference_1_testing_set')
    lora_model_handler.inference(input_path=config["INFERENCE"]["input_path_2"],
                                 report_file_prefix='inference_2_training_set')

