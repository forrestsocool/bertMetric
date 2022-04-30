import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TransformerEmb(nn.Layer):
    def __init__(self, pretrained_model, emb_size=128, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.emb_layer = nn.Linear(self.ptm.config["hidden_size"], emb_size)

    def forward(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):
        _, pooled_out = self.ptm(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)
        query_token_embedding = self.dropout(pooled_out)
        #return self.emb_layer(query_token_embedding)
        return query_token_embedding

    def forward_test(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):
        _, pooled_out = self.ptm(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)
        #query_token_embedding = self.dropout(pooled_out)
        #return self.emb_layer(query_token_embedding)
        return pooled_out