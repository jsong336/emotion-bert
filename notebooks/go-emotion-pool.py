class GoEmotionPoolClassifer(TransformerEncoderBase):
    def __init__(self, 
        encoder, 
        criterion = nn.BCEWithLogitsLoss(), 
        hiddens = None, 
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
        for h in self.config['hiddens']:
            fcs.append(nn.Linear(in_feature, h))
            fcs.append(nn.ReLU())
            in_feature = h    

        # final layer 
        fcs.append(nn.Linear(in_feature, n_cls))
        self.fcs = nn.Sequential(*fcs)

    def forward(self, input_ids, attention_mask, y_true=None):
        encoder_output = super().forward(input_ids, attention_mask)
        pool_output = encoder_output['pooler_output']

        z = self.dropout(pool_output)
        logits = self.fcs(z)

        if y_true is not None:
            loss = self.criterion(logits, y_true)
            return (loss, logits)

        return logits
