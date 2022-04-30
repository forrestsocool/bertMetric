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
        # query_token_embedding, _ = self.ptm(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)
        # query_token_embedding = self.dropout(query_token_embedding)
        # # 把无意义的id的mask位置设成0，否则设成1
        # query_attention_mask = paddle.unsqueeze(
        #     (query_input_ids != self.ptm.pad_token_id).astype(self.ptm.pooler.dense.weight.dtype),axis=2)
        # # Set token embeddings to 0 for padding tokens
        # query_token_embedding = query_token_embedding * query_attention_mask
        # query_sum_embedding = paddle.sum(query_token_embedding, axis=1)
        # query_sum_mask = paddle.sum(query_attention_mask, axis=1)
        # query_mean = query_sum_embedding / query_sum_mask
        # return query_mean

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