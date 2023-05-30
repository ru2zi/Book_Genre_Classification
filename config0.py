import math
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hyperparameter_defaults  = {
    'text_col': "Title",
    'label_col': "label_encode",
    'criterion': "CrossEntropyLoss",
    'model_name': "microsoft/deberta-v3-base",
    'scheduler': "cosine",
    'max_len': 256,
    'n_classes': 24,
    'fold': 5,
    'epochs': 10,
    'lr': 1e-3,
    'betas':(0.9,0.99)
    }


sweep_config = {
    'program': 'train.py',
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'num_warmup_steps_rate': {
            'distribution': 'categorical',
            'values': [0, 1]
        },
        'num_cycles': {
            'distribution': 'categorical',
            'values': [0, 1]
        },
        'train_batch_size': {
            'distribution': 'categorical',
            'values': [8, 16]
        },
        'valid_batch_size': {
            'distribution': 'categorical',
            'values': [8, 16]
        },
        'dropout': {
            'distribution': 'q_uniform',
            'min': 0.1,
            'max': 0.5,
            'q': 0.1
        }
    }
}


train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
