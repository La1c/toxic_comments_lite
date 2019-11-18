import torch
import torch.nn as nn
from transformers import BertModel

class BertSentimentClassifier(nn.Module):
    def __init__(self, freeze_bert = True, bert_model='bert-base-uncased', num_classes=6):
        super(BertSentimentClassifier, self).__init__()

        self.bert_layer = BertModel.from_pretrained(bert_model)
        
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        self.cls_layer = nn.Linear(768, num_classes)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)
        cls_rep = cont_reps[:, 0]
        logits = self.cls_layer(cls_rep)

        return logits