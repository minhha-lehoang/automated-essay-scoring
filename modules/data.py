import torch
import pandas as pd
from torch.utils.data import Dataset

def tokenize_text(text, tokenizer, is_train=True, max_seq_len=2048):
    if is_train:
        tokenized = tokenizer(text, padding=True,
                              return_tensors='pt')
        
        max_seq_len = tokenized['input_ids'].shape[1]

        return tokenized, max_seq_len
    else:
        tokenized = tokenizer(text, padding='max_length',
                              max_length=max_seq_len, truncation=True,
                              return_tensors='pt')

        return tokenized

class MultiFeaturesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ling_features: list):
        self.token_ids = df['input_ids'].values
        self.attention_mask = df['attention_mask'].values
        self.score = df['score'].values
        self.ling_features = []
        for feature in ling_features:
            self.ling_features.append(df[feature].values)

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        features = []
        for feature in self.ling_features:
            features.append(feature[idx])

        features = torch.tensor(features, dtype=torch.float)

        score = torch.reshape(torch.tensor(
            self.score[idx], dtype=torch.float), (1,))

        return torch.tensor(self.token_ids[idx], dtype=torch.long), torch.tensor(self.attention_mask[idx], dtype=torch.long), features, score