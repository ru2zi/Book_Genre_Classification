import torch.optim as optim
from torch.optim import AdamW

# def build_optimizer(network, optimizer, learning_rate):
#     if optimizer == "sgd":
#         optimizer = optim.SGD(network.parameters(),
#                               lr=learning_rate, momentum=0.9)
#     elif optimizer == "adam":
#         optimizer = optim.Adam(network.parameters(),
#                                lr=learning_rate)
#     return optimizer

def get_optimizer(model, lr, weight_decay, betas):
    # print('start_ get_optimizer')
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr,
                      weight_decay=weight_decay,
                      betas=betas,
                      )
    # print('end_ get_optimizer')

    return optimizer