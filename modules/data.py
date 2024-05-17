import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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


def pair_encoding(sentences, tokenizer, max_sent_length):
    if len(sentences) == 1:
        # who tf writes one sentence essays
        return [tokenizer(sentences[0], sentences[0], 
                          max_length=max_sent_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt')]

    pair_encodings = []
    for i in range(len(sentences) - 1):
        pair = sentences[i:i + 2]
        pair_encoding = tokenizer(
            pair[0], pair[1],
            max_length=max_sent_length,
            padding='max_length',
            truncation=True, return_attention_mask=True,
            return_tensors='pt')
        pair_encodings.append(pair_encoding)

    return pair_encodings


class LSCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sent_tokenzier, ling_features: list, max_sent_length=512):
        self.essay_token_ids = df['essay_input_ids'].values
        self.essay_attention_masks = df['essay_attention_mask'].values
        self.score = df['score'].values
        self.ling_features = []
        for feature in ling_features:
            self.ling_features.append(df[feature].values)
        self.sentence = df['sentence'].values

        self.max_sent_length = max_sent_length
        self.tokenizer = sent_tokenzier

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        # essay tokenization
        essay_input_ids = torch.tensor(
            self.essay_token_ids[idx], dtype=torch.long)
        essay_attention_mask = torch.tensor(
            self.essay_attention_masks[idx], dtype=torch.long)

        # linguistic features
        features = []
        for feature in self.ling_features:
            features.append(feature[idx])
        features = torch.tensor(features, dtype=torch.float)

        # sentence pair tokenization
        sentences = self.sentence[idx]

        pair_encodings = pair_encoding(
            sentences, self.tokenizer, self.max_sent_length)

        score = torch.tensor(self.score[idx], dtype=torch.float32)

        return features, essay_input_ids, essay_attention_mask, pair_encodings, score


def collate_fn(batch):
    features, essay_input_ids, essay_attention_mask, pair_encodings, score = zip(
        *batch)

    essay_input_ids = torch.stack(essay_input_ids, dim=0)
    essay_attention_mask = torch.stack(essay_attention_mask, dim=0)
    features = torch.stack(features, dim=0)
    score = torch.stack(score, dim=0).view(-1, 1)

    try:
        sent_input_ids = pad_sequence([torch.cat([pair_encoding['input_ids'] for pair_encoding in encodings])
                                       for encodings in pair_encodings],
                                      batch_first=True, padding_value=1)
        sent_input_ids = sent_input_ids.long()
        sent_attention_mask = pad_sequence([torch.cat([pair_encoding['attention_mask']
                                                       for pair_encoding in encodings]) for encodings in pair_encodings],
                                           batch_first=True, padding_value=0)
        sent_attention_mask = sent_attention_mask.long()
    except:
        print("Error in collate_fn")
        for pair in pair_encodings:
            print(pair)
        return None
    return features, essay_input_ids, essay_attention_mask, sent_input_ids, sent_attention_mask, score
