class GoEmotionGRUClassifer(TransformerEncoderBase):
  def __init__(self, 
        encoder, 
        criterion=nn.BCEWithLogitsLoss(), 
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
    for h in self.config['hiddens']:
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