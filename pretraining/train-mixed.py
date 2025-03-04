# source: https://github.com/redotvideo/mamba-chat/blob/main/train_mamba.py
# TODO: TRAINING IS TAKING 20 MINUTES PER STEP. SURELY THAT IS NOT CORRECT.
import torch
import argparse
import json
import os

from typing import Dict, Sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from dataset import TextDataset
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_trainer import MambaTrainer

# @dataclass

def make_datacollator(tokenizer):

    class DataCollatorForTextDataset(object):
        """
        Collate examples for supervised fine-tuning.
        """
        def __call__(self, instances: Sequence) -> Dict[str, torch.Tensor]:
            # input(type(instances)) # list
            # input(len(instances)) # 8 instances
            # input(len(instances[0])) # 2: each instance is (chunk, labels)
            input_ids,labels = tuple([instance[key] for instance in instances] for key in (0,1))
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            return dict(
                input_ids=input_ids,
                # labels=labels,
                # attention_mask=input_ids.ne(tokenizer.pad_token_id), # masks out own token. not sure this is the right way to go, but ehh
            )
    return DataCollatorForTextDataset()

def run(args):
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    if args.model is not None:
        model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device=device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    elif args.config is not None:
        config = {}
        with open(args.config, 'r') as f:
            config = json.load(f)
        model = MambaLMHeadModel(MambaConfig(**config),
                                 initializer_cfg=None,
                                 device=device,
                                 dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # idk why this is the default tokenizer *\_(-_-)_/*
    else:
        model = MambaLMHeadModel(MambaConfig(),
                                 initializer_cfg=None,
                                 device=device,
                                 dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # data_module = ChatDataModule(
    #     tokenizer=tokenizer,
    #     data_path=args.data_path,
    #     conversation_template=tokenizer.chat_template,
    #     max_tokens=2048
    # )
    train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/train.pt"))
    validation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/dev.pt"))

    # torch.serialization.add_safe_globals([TextDataset])
    train_data = torch.load(train_path, weights_only=False)
    validation_data = torch.load(validation_path, weights_only=False)

    data_collator = make_datacollator(tokenizer)#DataCollatorForTextDataset()
    # loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers)


    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output_dir,
            logging_strategy='steps',
            logging_dir=os.path.join(args.output_dir, 'logs')
            logging_first_step=True,
            logging_steps=1, # TODO: SET TO 50 FOR RUN?
            max_steps=5,  # TODO: TESTING ONLY. COMMENT OUT FOR RUN
            save_steps=500,
            eval_on_start=True,
            do_eval=True,
        ),
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="train")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=10)

    # max_epochs = 10
    # batch_size = 64
    # learning_rate = 3e-4
    # betas = (0.9, 0.95)
    # grad_norm_clip = 1.0
    # weight_decay = 0.1 # only applied on matmul weights
    # # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    # lr_decay = False
    # warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    # final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # # checkpoint settings
    # ckpt_path = None
    # num_workers = 0 # for DataLoader
    # writer = None

    args = parser.parse_args()

    run(args)