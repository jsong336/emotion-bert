from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from data import ZipDataset, tokenize_bert_input
from model import TransformerEncoderBase
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from transformers import (
  AdamW,
  get_linear_schedule_with_warmup
)
from utils import save_checkpoint
from typing import Union
import tqdm 
import torch
import torch.nn as nn
import numpy as np
import shutil
import os


def compute_classification_metrics(y_true, proba, threshold):
  assert len(y_true) == len(proba), 'y_true and y_pred length mismatch {} {}'.format(len(y_true), len(proba))

  results = {}
  y_true = y_true.astype(int)
  y_pred = (proba >= threshold).astype(int)

  results["accuracy"] = (y_true == y_pred).mean()
  if (np.unique(y_true) == 1).sum() == 0:
    results["auc_roc_macro"] = roc_auc_score(y_true, proba, average='macro')
    results["auc_roc_micro"] = roc_auc_score(y_true, proba, average='micro')
  results["macro_precision"], results["macro_recall"], results["macro_f1"], _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
  results["micro_precision"], results["micro_recall"], results["micro_f1"], _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
  results["weighted_precision"], results["weighted_recall"], results["weighted_f1"], _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

  return results


def predict_proba(
    model:nn.Module, 
    dataset:ZipDataset, 
    batch_size:int=16, 
    device:str='cpu', 
    back_to_cpu:bool=True):

  eval_dataloader = DataLoader(
      dataset, 
      batch_size=batch_size, 
  )

  n_batch = 0
  proba = []

  model.to(device)

  for batch in tqdm(eval_dataloader, desc='evaluation', leave=False):
    model.eval()
    batch = { k:v.to(device) for k, v in batch.items() if k != 'y_true'}

    with torch.no_grad():
      logits = model(**batch)
      logits = logits.cpu().detach().numpy()

    p = 1 / (1 + np.exp(-logits))
    proba.append(p)

    n_batch += 1

  if back_to_cpu:
    model.cpu()

  proba = np.vstack(proba)
  return proba


def predict_proba_examples(
    X:Union[dict, list], 
    model:TransformerEncoderBase, 
    tokenizer:Tokenizer=None):    
    X_tk = tokenize_bert_input(X, tokenizer) if tokenizer else X
    with torch.no_grad():
        logits = model(**X_tk).numpy()
        proba = 1 / (1 + np.exp(-logits))
    
    return proba


def train_step(
    model:nn.Module,
    batch:dict, 
    optimizer:nn.Module, 
    scheduler:nn.Module,
    grad_clip_max=1
):
  model.train()

  loss, logits = model(*batch.values())
  loss.backward()

  if grad_clip_max is not None:
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max)

  optimizer.step()
  scheduler.step()
  model.zero_grad()

  return loss.detach().item(), logits


def evaluate(model, 
             dataset, 
             batch_size=16, 
             threshold=0.5,
             device='cpu', back_to_cpu=True):
  eval_dataloader = DataLoader(
      dataset, 
      batch_size=batch_size, 
  )

  n_batch = 0
  total_loss = 0.0
  y_true = []
  proba = []

  model.to(device)

  for batch in tqdm(eval_dataloader, desc='evaluation', leave=False):
    model.eval()
    batch = { k:v.to(device) for k, v in batch.items() }

    with torch.no_grad():
      loss_per_batch, logits = model(**batch)
      total_loss += loss_per_batch.item()

      logits = logits.cpu().detach().numpy()

    p = 1 / (1 + np.exp(-logits))
    proba.append(p)
    y_true.append(batch['y_true'].cpu().detach().numpy())

    n_batch += 1

  if back_to_cpu:
    model.cpu()

  proba = np.vstack(proba)
  y_true = np.vstack(y_true)
  results = {
      'loss': total_loss / n_batch, 
      'trigger_rate': (proba >= threshold).mean(), 
      **compute_classification_metrics(y_true, proba, threshold)
  }

  return results


def train(
    model, 
    train_dataset, 
    val_dataset, 
    metadata=None,
    tokenizer = None, 
    epochs = 5,
    train_batch_size = 16, 
    val_batch_size = 16, 
    save_steps = 1e3, 
    validation_steps = 1e3, 
    archive_dir = None,
    model_name = 'model', # model & archive saved in archive_dir/model_name/..
    classification_threshold=0.5, 
    learning_rate = 1e-3, 
    grad_clip_max = 1, 
    weight_decay = 1e-5, 
    warmup_ratio=0.1,
    logging_metrics=None,
    clear_archive=False, 
    optimizer=None, 
    scheduler=None, 
    continue_training=False,
    device = 'cpu'
):
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  if clear_archive:
    archive_path = os.path.join(archive_dir, model_name)
    shutil.rmtree(archive_path)
    os.makedirs(archive_path)

  # no weight decay on LayerNorm and bias
  grouped_parameters = [
    {'params': [param for name, param in model.named_parameters() \
               if not any(nd in name for nd in ('bias', 'LayerNorm.weight'))]}, 
    {'params': [param for name, param in model.named_parameters() \
               if any(nd in name for nd in ('bias', 'LayerNorm.weight'))]}
  ]

  train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)  
  max_step = len(train_dataset) * epochs
  metadata['train_max_step'] = max_step

  if not continue_training:
    # use default b1, b2, eps
    optimizer = AdamW(grouped_parameters, 
                      lr=learning_rate, 
                      weight_decay=weight_decay) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_step * warmup_ratio),
        num_training_steps=max_step
    )
  else:
    assert optimizer is not None
    assert scheduler is not None

  print('hello')

  total_steps = metadata['total_steps'] if continue_training else 0
  tr_loss = metadata['tr_loss'] * total_steps if continue_training else 0.0

  model.zero_grad() 
  model.to(device)

  save_checkpoint(model, archive_dir, model_name, 
                  checkpoint_id='null-model', 
                  metadata=metadata,
                  tokenizer=tokenizer, 
                  optimizer=optimizer, 
                  scheduler=scheduler)

  for epoch in range(epochs):
    print('training epoch %d' % epoch)
    for batch in tqdm(train_dataloader, desc="Training", leave=None):
      batch = { k:v.to(device) for k, v in batch.items() }
      step_loss, _ = train_step(model, batch, optimizer, scheduler, grad_clip_max)

      tr_loss += step_loss
      total_steps += 1 
      
      metadata['tr_loss'] = tr_loss / total_steps

      if total_steps > 0 and (total_steps % validation_steps == 0 or \
                              total_steps % save_steps == 0):
        result = evaluate(model, val_dataset, val_batch_size, 
                          classification_threshold, device=device, 
                          back_to_cpu=False)
        metadata['val_metrics'] = result

        print('evaluating at step %d' % total_steps)
        if logging_metrics is not None:
          print({ k: v for k, v in result.items() if k in logging_metrics})

      if total_steps > 0 and total_steps % save_steps == 0:
        print('saving at step %d' % total_steps)
        metadata['total_steps'] = total_steps
        save_checkpoint(model, archive_dir, model_name, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        checkpoint_id=total_steps, 
                        metadata=metadata)
  model.cpu()
  return model, metadata