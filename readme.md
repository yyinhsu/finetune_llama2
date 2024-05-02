# Finetuning Llama2

## Introduction
- This is a side project of an AI developer, so the code may not be well-organized
- Finetuning Llama2 of Meta, using [LoRA](https://huggingface.co/docs/peft/developer_guides/lora) as the method, which is supported by [PEFT](https://huggingface.co/docs/peft/index)
- You can define your own configuration file to finetune the model with your own settings. The example configuration file is [cfg_files/train_lora_llama2_fake.cfg](cfg_files/train_lora_llama2_fake.cfg)
- You can also create your own dataset as jsonl file to finetune the model. The example dataset is [data/data_example.jsonl](data/data_example.jsonl)

## Prerequisites
- python 3.9
- reference to [requirements.txt](environment/requirements.txt)

## Usage
- `python main.py` to start finetuning with the default configuration (the default file is not included in this repository)
- `python main.py --cfg_file <cfg file path>` to finetune with a specific configuration file

## Configuration
- The configuration file is in cfg format, which is a key-value pair format
- To see the default configuration, refer to [example](cfg_files/train_lora_llama2_fake.cfg)
- You can change the configuration file to finetune with your own settings
- The huggingface token should be replaced with your own token


## Acknowledgement
- This project is inspired by [PEFT](https://huggingface.co/docs/peft/index) and [LoRA](https://huggingface.co/docs/peft/developer_guides/lora)
- The pretrained model is from [Meta](https://huggingface.co/meta)

## License
- Same as the original license of the pretrained model
