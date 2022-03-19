from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch import Tensor 
from torch.utils.data import TensorDataset, Dataset
from transformers import Tokenizer
from typing import Dict
import pandas as pd
import torch


class ZipDataset(Dataset):
  def __init__(self, datasets: Dict[TensorDataset]):
    super(ZipDataset, self).__init__()
    self.keys = list(datasets.keys())
    self.values = list(datasets.values()) 
    self.datasets = datasets
    assert all([len(self.values[0]) == len(v) for v in self.values])

  def __len__(self):
    return len(self.values[0])
  
  def __getitem__(self, idx):
    item = {}
    for k, v in self.datasets.items():
      item[k] = v[idx]
    return item


def tokenize_bert_input(texts:str, tokenizer:Tokenizer, maxlen:int=50):
    result = tokenizer.batch_encode_plus(
            texts, 
          return_attention_mask=True, 
        return_token_type_ids=False,
        padding='longest', 
        max_length=maxlen)
    return result['input_ids'], result['attention_mask']


def _generate_bert_dataset(
    X, 
    y, 
    tokenizer:Tokenizer,
    sentence_max_len:int = 50,
    split:float = None
):
    if split is None:
        X_tk , X_mask = tokenize_bert_input(
            X.tolist(), 
            tokenizer=tokenizer, 
            maxlen=sentence_max_len
        )
        dataset = ZipDataset({
            'input_ids': Tensor(X_tk).type(torch.int32), 
            'attention_mask': Tensor(X_mask).type(torch.int32), 
            'y_true': Tensor(y).type(torch.float32)
        })
        return dataset
    else:
        X1, X2, y1, y2 = train_test_split(
            X, y, 
            test_size=split, 
            shuffle=False,
        )
        d1 = _generate_bert_dataset(X1, y1, tokenizer, sentence_max_len, split=None)
        d2 = _generate_bert_dataset(X2, y2, tokenizer, sentence_max_len, split=None)
        return (d1, d2)


def generate_bert_dataset(
    dataset_path, 
    tokenizer:Tokenizer,
    emotions: tuple, 
    sentence_max_len:int = 50,
    split:float = None
):
    D = pd.read_csv(dataset_path)
    X = D['text'].to_numpy()
    y = D[list(emotions)].to_numpy()

    return _generate_bert_dataset(
        X, y, 
        tokenizer, 
        sentence_max_len, 
        split
    )