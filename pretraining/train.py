import torch
import os
from glob import glob
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR

from dataset import TextDataset
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

import math
import logging

from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    writer = None
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        step = 0
        def run_epoch(split):
            nonlocal step
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.val_dataset
            loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers)

            losses = []
            thing = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            
            for it, (inputs, targets) in thing:

                # place data on the correct device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    output = model(inputs)
                    loss = CrossEntropyLoss(output.view(-1, output.size(-1)), targets.view(-1), ignore_index=0)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    thing.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    
                    if config.writer is not None:
                        config.writer.add_scalar('train/loss',  loss.item(), step)
                        config.writer.add_scalar('train/lr', lr, step)
                    
                step += 1
            if not is_train:
                logger.info("validation loss: %f", np.mean(losses))

        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.val_dataset is not None:
                run_epoch('validation')

            self.save_checkpoint()

if __name__ == "__main__":
    # Load preprocessed datasets
    train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/train.pt"))
    validation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./text_data/dev.pt"))

    train_data = torch.load(train_path)
    validation_data = torch.load(validation_path)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    tconf = TrainerConfig()
    model = MambaLMHeadModel(
        MambaConfig(),
        initializer_cfg=None,
        device=device,
        dtype=torch.float16)

    trainer = Trainer(
        model,
        train_data,
        validation_data,
        tconf)

    trainer.train()

    torch.save(model.state_dict(), os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/mamba_pretrained.pt")))