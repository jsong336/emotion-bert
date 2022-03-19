import os
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel 
from typing import List

class SaveableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def save_pretrained(self, path):
        pass

    @classmethod
    def from_pretrained(cls, path):
        pass
    

class TransformerEncoderBase(torch.nn.Module):
    def __init__(self, 
        encoder: PreTrainedModel, 
        criterion: nn.Module, 
        config: dict=None
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.config.hidden_size
        self.criterion = criterion
        self.config = config if config else {} # put all model argument except encoder, encoder_dim in the 

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask,)

    def save_pretrained(self, path):
        encoder_path = os.path.join(path, 'encoder')
        pt_path = os.path.join('model.pt')
        encoder = self.encoder
        encoder.save_pretrained(encoder_path)
        self.encoder = None
        torch.save({
            'model': self.state_dict(), 
            'config': {
                'encoder': encoder.config.to_dict(), 
                'criterion': self.criterion.__class__.__name__,
                'architecture': str(self), 
                'module':self.config, 
            }
        }, pt_path)
        self.encoder = encoder
        return 
    
    @classmethod
    def from_pretrained(cls, path):
        encoder_path = os.path.join(path, 'encoder')
        pt_path = os.path.join('model.pt')

        checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        encoder = AutoModel.from_pretrained(encoder_path)
        
        model = cls(
            encoder=None, 
            encoder_dim = config['encoder']['hidden_size'], 
            criterion = getattr(torch.nn, config['criterion'])(), 
            **config['module']
        )
        model.load_state_dict(checkpoint['model'])
        model.encoder = encoder
        return model


class GoEmotionPoolClassifer(TransformerEncoderBase):
    def __init__(self, 
        encoder: PreTrainedModel, 
        criterion: nn.Module = nn.BCEWithLogitsLoss(), 
        hiddens: List[int] = None, 
        dropout_p: float = 0.1, 
        n_cls:int = 28
    ) -> None:
        config = {
            'hiddens':  [100] if hiddens is None else hiddens, 
            'dropout_p': dropout_p, 
            'n_cls': n_cls
        }
        super().__init__(encoder, criterion, config)

        # new layers 
        self.dropout = nn.Dropout(dropout_p)

        # full connected
        fcs = []
        in_feature = self.encoder_dim 
        for h in self.hiddens:
            fcs.append(nn.Linear(in_feature, h))
            fcs.append(nn.ReLU())
            in_feature = h    

        # final layer 
        fcs.append(nn.Linear(in_feature, n_cls))
        self.fcs = nn.Sequential(*fcs)

    def forward(self, X_tk, X_mask, y_true=None):
        encoder_output = super().forward(X_tk, X_mask)
        pool_output = encoder_output['pooler_output']

        z = self.dropout(pool_output)
        logits = self.fcs(z)

        if not (y_true is None):
            loss = self.criterion(logits, y_true)
            return (loss, logits)

        return logits


class GoEmotionGRUClassifer(TransformerEncoderBase):
  def __init__(self, 
        encoder: PreTrainedModel, 
        criterion: nn.Module = nn.BCEWithLogitsLoss(), 
        seq_len=82,
        rnn_hidden = 50,  
        rnn_num_layers = 1,
        bidirectional=True, 
        hiddens = None,
        dropout_p=0.1, 
        n_cls:int = 28
    ):
    config = {
        'seq_len': seq_len, 
        'rnn_hidden': rnn_hidden, 
        'rnn_num_layers': rnn_num_layers, 
        'bidirectional': bidirectional, 
        'hiddens':  [50] if hiddens is None else hiddens, 
        'dropout_p': dropout_p, 
        'n_cls': n_cls
    }
    super().__init__(encoder, criterion, config)

    # layers
    self.encoder = encoder
    self.gru = nn.GRU(
        input_size= self.encoder_dim, 
        hidden_size = rnn_hidden, 
        batch_first = True, 
        bidirectional = bidirectional
    )
    self.dropout = nn.Dropout(dropout_p)

    # full connected
    fcs = []
    in_feature = (int(bidirectional) + 1) * rnn_hidden
    for h in self.hiddens:
      fcs.append(nn.Linear(in_feature, h))
      fcs.append(nn.ReLU())
      in_feature = h    
    
    # final layer 
    fcs.append(nn.Linear(in_feature, n_cls))
    self.fcs = nn.Sequential(*fcs)

  def forward(self, input_ids, attention_mask, y_true=None):
    encoder_output = super().forward(input_ids, attention_mask)
    contextual_emb = encoder_output['last_hidden_state']

    output, _ = self.gru(contextual_emb)
    output = output[:, -1, :]
    z = self.dropout(output) 
    logits = self.fcs(z)

    if not (y_true is None):
      loss = self.criterion(logits, y_true)
      return (loss, logits)

    return logits