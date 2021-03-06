# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

#%% DataClass definition
class News_dataset(Dataset):
    def __init__(self, data_df):
        self.token_ids = torch.stack(list(data_df['token_ids']))
        self.token_types = torch.stack(list(data_df['token_types']))
        self.att_masks = torch.stack(list(data_df['att_masks']))
        self.labels = torch.LongTensor(list(data_df['Y']))
                                        
    def __len__(self):
        return len(self.token_ids)
        
    def __getitem__(self, idx):
        X_token_ids = self.token_ids[idx]
        X_token_types = self.token_types[idx]
        X_att_masks = self.att_masks[idx]
        Y_labels = self.labels[idx]
        
        return X_token_ids, X_token_types, X_att_masks, Y_labels

#%% Model definition
class News_model(nn.Module):
            
    def __init__(self, args):
        super(News_model, self).__init__()

        self.h_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_labels = args.num_labels
        self.seq_len = args.seq_len
        self.dropout = args.dropout
                     
        # Bert layer
        self.model_name = 'nlpaueb/legal-bert-small-uncased'
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        
        # Transformer layers
        #self.transf_enc_facts = nn.TransformerEncoderLayer(d_model = self.h_dim,
        #                                                   nhead = self.n_heads)
        
        # Fully connected output
        self.fc_out = nn.Linear(in_features = self.h_dim, out_features = self.n_labels)

        # Softmax
        #self.softmax = nn.Softmax(dim = 1)

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim)

    def forward(self, X_token_ids, X_token_types, X_att_masks):
        # BERT encoder
        X_bert = {'input_ids': X_token_ids,
                  'token_type_ids': X_token_types,
                  'attention_mask': X_att_masks}
        out = self.bert_model(**X_bert,
                              output_hidden_states = True)       # Tuple
        out = out['pooler_output']                               # batch_size x h_dim
        
        # Multi-label classifier      
        out = self.bn1(out)                                      # batch_size x h_dim
        out = self.fc_out(out)                                   # batch_size x n_lab

        return out
