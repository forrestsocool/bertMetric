from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from functools import partial

_P = os.path.dirname
dsf_root = _P(os.path.realpath(__file__))
sys.path.append(dsf_root)

import unittest
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from transformer_emb import TransformerEmb

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from datautils import read_custom_data, create_dataloader, convert_example


class TestTransformerEmb(unittest.TestCase):
    def setUp(self):
        paddle.set_device('gpu')
        # If you wanna use bert/roberta pretrained model,
        # pretrained_model = ppnlp.transformers.BertModel.from_pretrained('bert-base-chinese')
        # pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained('roberta-wwm-ext')
        self.pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
            'ernie-1.0')
        # self.pretrained_model = ppnlp.transformers.ErnieForTokenClassification.from_pretrained(
        #     'ernie-1.0')

        # If you wanna use bert/roberta pretrained model,
        # tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese')
        # tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext')
        # ErnieTinyTokenizer is special for ernie-tiny pretained model.
        self.tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(
            'ernie-1.0')

        self.model = TransformerEmb(self.pretrained_model)

    def test_forward(self):
        # test print token
        query_encoded_inputs = self.tokenizer(text="长沙是臭豆腐之都[SEP]哈尔滨是腊肠之都", max_seq_len=512)
        query_input_ids = query_encoded_inputs["input_ids"]
        query_token_type_ids = query_encoded_inputs["token_type_ids"]
        print(query_input_ids)
        print(query_token_type_ids)

        # test custom data
        train_ds = load_dataset(read_custom_data, data_path='/workspace/project/bertMetric/test.tsv', lazy=False)
        # for step, batch in enumerate(train_ds, start=1):
        #     print(convert_example(batch, self.tokenizer))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # query_input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # query_segment
            Stack(dtype="int64")  # label
        ): [data for data in fn(samples)]

        trans_func = partial(
            convert_example,
            tokenizer=self.tokenizer,
            max_seq_length=512)

        train_data_loader = create_dataloader(
            train_ds,
            mode='train',
            batch_size=1,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)

        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, labels = batch
            emb = self.model(query_input_ids, query_token_type_ids)
            print(emb)
            break