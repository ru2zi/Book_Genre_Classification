import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch


class GoodreadsModel(nn.Module):
    def __init__(self, model_name, dropout, n_classes):
        # print('start_GoodreadsModel')

        super(GoodreadsModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(dropout)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, n_classes)
        # print('end_GoodreadsModel')

        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    


# class ConvNet(nn.Module):
#     def __init__(self, fc_layer_size, dropout):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
#             nn.MaxPool2d(2, 2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
#             nn.MaxPool2d(2, 2))
#         self.layer3 = nn.Sequential(
#             nn.Linear(64 * 7 * 7, fc_layer_size, bias=True), nn.ReLU(),
#             nn.Dropout2d(p=dropout))
#         self.layer4 = nn.Sequential(
#             nn.Linear(fc_layer_size, 84), nn.ReLU(),
#             nn.Dropout2d(p=dropout))
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.view(x.size(0),-1) 
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.fc3(x)
#         return x