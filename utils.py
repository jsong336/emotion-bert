import os
import re
import numpy as np 
import random
import pandas as pd
import preprocessor
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from typing import Union
from transformers import (
    AutoTokenizer, 
    AdamW, 
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
from model import SaveableModel


re_number = re.compile('[0-9]+')
re_url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
re_tag = re.compile('\[[A-Z]+\]')
re_char = re.compile('[^0-9a-zA-Z\s?!.,:\'\"//]+')
re_char_clean = re.compile('[^0-9a-zA-Z\s?!.,\[\]]')
re_punc = re.compile('[?!,.\'\"]')


def init_seed(seed:int):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  return

def _word_mapper(text:str, mapper:dict):
    for word in text.split(' '):
        if word in mapper:
            text = text.replace(word, mapper[word])
    return text


def _char_mapper(text:str, mapper:dict):
    for k, v  in mapper.items():
        text = text.replace(k, v)
    return text


def clean_text(text:str, mapChar:dict, mapC:dict, mapS:dict):
    """
        text:
        mapChar: character mapper
        mapC: contraction mapper
        mapS: spelling mistake mapper
    """
    text = re.sub(re_char, "", text) # Remove unknown character
    text = _char_mapper(text, mapChar) # Similar characters mapping
    text = _word_mapper(text, mapC) # Remove contraction
    text = _word_mapper(text, mapS) # Remove spelling mistakes

    text = re.sub(re_number, ' [number] ', text) # Replace number with tag
    text = re.sub(re_url, ' [url] ', text) # Replace URL with number

    text = re.sub(re_punc, lambda a: f" {a.group(0)} ", text) # Add space between punctuation
    text = preprocessor.clean(text) # Remove tweet clean

    text = re.sub(re_char_clean, "", text) # Only alphanumeric and punctuations.
    text = text.lower() # Lower text
    text = " ".join([w for w in text.split(' ') if w != " "]) # Remove whitespace

    return text


def prepare_split_data(
    dataset_source_path:str, 
    train_dataset_path:str, 
    test_dataset_path:str, 
    emotions:tuple, 
    drop_insignifiant=False, 
    test_split=0.2, 
    seed=0, 
):
    """
    dataset_source_path: path to csv path
    train_dataset_path: path to save csv path
    test_dataset_path: path to save csv path
    emotions: tuple of emotions must reserve order
    drop_insignifiant: remove rows that does not belong any of `emotions`
    test_split: split ratio
    seed: shuffle seed
    """
    emotions = list(emotions)
    data = pd.read_csv(dataset_source_path)
    data['text'] = data['text'].progress_apply(clean_text)
    data = data[data['text'] != '']
    data = data[['text'] + list(emotions)]

    if drop_insignifiant:
        data = data[data[list(emotions)].sum(1) > 0].reset_index(drop=True)

    train, test = train_test_split(data, 
                                    test_size=test_split, 
                                    shuffle=True, 
                                    random_state=seed)
    
    train.to_csv(train_dataset_path, index=None)
    test.to_csv(test_dataset_path, index=None)
    return


def save_checkpoint(
    model:SaveableModel, 
    archive_dir:str, 
    model_name:str, 
    checkpoint_id:Union[str, int]="?", 
    metadata:dict=None,
    tokenizer:Union[SaveableModel, PreTrainedTokenizer]=None, 
    optimizer:nn.Module=None, 
    scheduler:nn.Module=None,
):
  # create archive folder
  archive_path = os.path.join(archive_dir, model_name)
  if not os.path.exists(archive_path):
    os.makedirs(archive_path, exist_ok=True)

  # create checkpoint folder
  checkpoint_dir = os.path.join(archive_path, 'checkpoint-%s' % str(checkpoint_id))
  os.makedirs(checkpoint_dir, exist_ok=True)

  # save model in checkpoint
  model_to_save = (model.module if hasattr(model, "module") else model)
  model_to_save.save_pretrained(checkpoint_dir)
  if tokenizer is not None:
    tokenizer.save_pretrained(checkpoint_dir)
  if metadata:
    torch.save(metadata, os.path.join(checkpoint_dir, "meta.bin"))
  if scheduler is not None:
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler.pt'))
  if optimizer is not None:
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))

  return archive_path


def load_from_checkpoint(
    archive_dir:str, 
    model_name:str, 
    checkpoint_id:str="?", 
    load_tokenizer:bool=False, 
    load_metadata:bool=True, 
    load_optimizer:bool=False, 
    module_cls:type = SaveableModel,
    tokenizer_cls:type=AutoTokenizer
):
    archive_path = os.path.join(archive_dir, model_name)
    checkpoint_dir = os.path.join(archive_path, 'checkpoint-%s' % str(checkpoint_id))

    assert os.path.exists(archive_path), archive_path
    assert os.path.exists(checkpoint_dir), checkpoint_dir

    model = getattr(module_cls, 'from_pretrained')(checkpoint_dir)

    output = (model, )
    if load_tokenizer:
        tokenizer = getattr(tokenizer_cls, 'from_pretrained')(checkpoint_dir)
        output += (tokenizer, )

    if load_metadata or load_optimizer:
        metadata = torch.load(os.path.join(checkpoint_dir, 'meta.bin'))
        if load_metadata:
            output += (metadata, )

    if load_optimizer:
        grouped_parameters = [
            {'params': [param for name, param in model.named_parameters() \
                if not any(nd in name for nd in ('bias', 'LayerNorm.weight'))]}, 
            {'params': [param for name, param in model.named_parameters() \
                if any(nd in name for nd in ('bias', 'LayerNorm.weight'))]}]

        optimizer = AdamW(grouped_parameters, 
                    lr=metadata['learning_rate'], 
                    weight_decay=metadata['weight_decay']) 
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(metadata['train_max_step'] * metadata['warmup_ratio']),
            num_training_steps=metadata['train_max_step']
        )
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scheduler.pt')))

        output += (optimizer, scheduler)
        
    return output
