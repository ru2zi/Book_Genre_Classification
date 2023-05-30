from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel, AutoConfig

# def SweepDataset(batch_size, transform):
#     transform = transform
#     # download MNIST training dataset
#     dataset = datasets.MNIST(".", train=True, download=True,
#                             transform=transform)
#     sub_dataset = torch.utils.data.Subset(
#         dataset, indices=range(0, len(dataset), 5))
#     loader = DataLoader(sub_dataset, batch_size=batch_size)

#     return loader

class GoodreadsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, target=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target = target
        if target is not None:
            self.target = list(target)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        if self.target is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target": torch.tensor(self.target[item], dtype=torch.float32)
            }
        else:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
            }

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

# class get_dataloader():
#     def __init__(self, model_name, fold, text_col, max_len, label_col, train_batch_size, valid_batch_size):
#         self.df = pd.read_csv('C:/Users/user/PycharmProjects/JBIG_ML/jbnu-swuniv-ai/train_data.csv')
#         self.labelEncoding()
#         self.Folding(fold)
#         self.Making_Data_loader(fold, model_name, text_col, max_len, label_col, train_batch_size, valid_batch_size)

#     def labelEncoding(self):
#         le = LabelEncoder()
#         self.df["label_encode"] = le.fit_transform(self.df.label)
    
#     def Folding(self, fold):
#         kf = StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)
#         for f, (t_, v_) in enumerate(kf.split(X=self.df.Title, y=self.df.label_encode)):
#             self.df.loc[v_, 'fold'] = f
    
#     def Making_Data_loader(self, fold, model_name, text_col, max_len, label_col, train_batch_size, valid_batch_size):
#         train_df = self.df[self.df.fold!=fold]
#         valid_df = self.df[self.df.fold==fold]

#         tokenizer = get_tokenizer(model_name)

#         train_ds = GoodreadsDataset(train_df[text_col], 
#                                 tokenizer, 
#                                 max_len, 
#                                 train_df[label_col])
#         valid_ds = GoodreadsDataset(valid_df[text_col],
#                                 tokenizer, 
#                                 max_len, 
#                                 valid_df[label_col])
        
#         train_loader = DataLoader(
#                 dataset=train_ds, batch_size=train_batch_size, shuffle=True)
#         valid_loader = DataLoader(
#                 dataset=valid_ds, batch_size=valid_batch_size, shuffle=False)
            
#         return train_loader, valid_loader
    
def get_dataloader(model_name, fold_count, text_col, max_len, label_col, train_batch_size, valid_batch_size):
    # print('start_get_dataloader')
    df = pd.read_csv('C:/Users/user/PycharmProjects/JBIG_ML/jbnu-swuniv-ai/train_data.csv')

    le = LabelEncoder()
    df["label_encode"] = le.fit_transform(df.label)

    kf = StratifiedKFold(n_splits=fold_count, random_state=42, shuffle=True)
    for f, (t_, v_) in enumerate(kf.split(X=df.Title, y=df.label_encode)):
        df.loc[v_, 'fold'] = f
    
    # df = df[:30000]
    train_df = df[df.fold!=0]
    valid_df = df[df.fold==0]

    tokenizer = get_tokenizer(model_name)

    train_ds = GoodreadsDataset(train_df[text_col], 
                            tokenizer, 
                            max_len, 
                            train_df[label_col])
    valid_ds = GoodreadsDataset(valid_df[text_col],
                            tokenizer, 
                            max_len, 
                            valid_df[label_col])
    
    train_loader = DataLoader(
            dataset=train_ds, batch_size=int(train_batch_size), shuffle=True)
    valid_loader = DataLoader(
            dataset=valid_ds, batch_size=int(valid_batch_size), shuffle=False)

    return train_loader, valid_loader