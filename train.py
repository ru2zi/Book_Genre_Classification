from torchvision import datasets
from dataset import get_dataloader
from model import GoodreadsModel
from optimize import get_optimizer
from utils import train_epoch, train_one_epoch, train_one_fold
from torchvision import datasets
from dataset import GoodreadsDataset
from schedule import get_scheduler

import config0
import wandb
import yaml

class CFG:
    text_col = "Title"
    label_col ="label_encode"
    criterion = "CrossEntropyLoss"
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    scheduler = "cosine"
    max_len= 256
    n_classes= 24
    fold= 5
    epochs= 10
    lr= 1e-6
    betas=(0.9,0.99)

def train():
    wandb.init(config=config0.hyperparameter_defaults)
    w_config = wandb.config

    train_loader, valid_loader = get_dataloader(CFG.model_name, CFG.fold, CFG.text_col, CFG.max_len, CFG.label_col, w_config.train_batch_size, w_config.valid_batch_size)
    model = GoodreadsModel(CFG.model_name, w_config.dropout, CFG.n_classes).to(config0.DEVICE)
    optimizer = get_optimizer(model, CFG.lr, w_config.weight_decay, CFG.betas)
    scheduler = get_scheduler(CFG.scheduler, optimizer, train_loader, CFG.epochs, w_config.num_warmup_steps_rate, w_config.num_cycles)

    wandb.watch(model, log='all')

    train_one_fold(model, optimizer, scheduler, train_loader, valid_loader, CFG.epochs, config0.DEVICE, wandb)

sweep_id = wandb.sweep(config0.sweep_config, project='LEO0', entity='smuff11')
wandb.agent(sweep_id, function=train, count = 10)
