# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from arcface import ArcMarginProduct
from transformer_emb import TransformerEmb


class BertMetric(nn.Layer):
    def __init__(self, pretrained_model, emb_size, num_labels, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.num_labels = num_labels
        self.emb_size = emb_size
        self.emb_layer = TransformerEmb(self.ptm, self.emb_size)
        self.classifier = ArcMarginProduct(self.emb_size, self.num_labels)

    def forward(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                label=None,
                is_test=False):
        query_token_embedding = self.emb_layer(query_input_ids, query_token_type_ids,
                                               query_position_ids, query_attention_mask,
                                               is_test)
        if not is_test:
            return self.classifier(query_token_embedding, label)
        else:
            return self.classifier.forward_test(query_token_embedding)

    def predict_emb(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):

        return self.emb_layer.forward_test(query_input_ids,
                              query_token_type_ids,
                              query_position_ids,
                              query_attention_mask)