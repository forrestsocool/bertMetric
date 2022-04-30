#-*- coding: utf-8 -*-
import os
os.environ['JAVA_HOME']='/opt/Bigdata/client/JDK/jdk1.8.0_272'
os.environ['SPARK_HOME']='/opt/Bigdata/client/Spark2x/spark'
os.system("source /opt/Bigdata/client/bigdata_env")
os.system("/opt/Bigdata/client/KrbClient/kerberos/bin/kinit -kt /workspace/gpsearch.keytab GPSearch")
os.system("source /opt/Bigdata/client/Hudi/component_env")
#export HADOOP_USER_NAME=hive
#os.environ['HADOOP_USER_NAME']='hive'
import findspark
findspark.init()
import sys
# --jars hdfs:///user/lisensen/tools/jpmml-sparkml-executable-1.5.13.jar
# pyspark_submit_args = ' --executor-memory 2g --driver-memory 8g --executor-cores 2 --num-executors 30 --conf spark.shuffle.spill.numElementsForceSpillThreshold=2000000 --conf spark.memory.storageFraction=0.2 --conf spark.dlism=2000 --conf spark.sql.shuffle.partitions=2000 --conf spark.dynamicAllocation.enabled=false --conf spark.port.maxRetries=100 --conf spark.driver.maxResultSize=8g' + ' pyspark-shell'
pyspark_submit_args = ' --master local[*] --driver-memory 16g --executor-cores 1 --conf spark.driver.extraJavaOptions=" -Xss16384k" --conf spark.driver.memoryOverhead=4g --conf spark.local.dir=/opt/home/lisensen/temp --conf spark.shuffle.memoryFraction=0.1 --conf spark.kryoserializer.buffer.max=1800m' + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
os.environ['HADOOP_USER_NAME']='hdfs'
#import findspark
#findspark.init()
import argparse
import os
import sys
from pyspark import StorageLevel
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext,SQLContext,Row,SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf,col,column
import pyspark.sql.types as typ
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType, IntegerType
import jieba
import os
import pandas as pd
from pyhive import hive
import re
import faiss
import mkl
import numpy as np
import argparse
import json
import redis
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from model import BertMetric


parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default='./model_state.pdparams',
                    help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=50, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument('--emb_size',  type=int, default=1024, help="embedding dims")
parser.add_argument('--cls_size',  type=int, default=31, help="total class nums")
parser.add_argument("--redis_host", type=str, default='10.255.24.40')
parser.add_argument("--redis_password", type=str, default='S4wKxoLGRo')
parser.add_argument("--redis_port", type=int, default=6379)
parser.add_argument("--redis_db", type=int, default=14)
parser.add_argument("--exp_seconds", type=int, default=7*24*3600)
args = parser.parse_args()

remove_words = ['【福利秒杀】','【每日福利】','【福利爆款】','【专柜品质】','【1元秒杀】','【直播专用1元秒杀】','【','】','源本']

def remove(x):
    for r in remove_words:
        x = x.replace(r, '')
    x = x.strip()
    return x

def convert_example(query, tokenizer, max_seq_length=512, is_test=False):
    #print(query)
    query_encoded_inputs = tokenizer(query, max_seq_len=max_seq_length)
    query_input_ids = query_encoded_inputs["input_ids"]
    query_token_type_ids = query_encoded_inputs["token_type_ids"]
    #return query_input_ids, query_token_type_ids
    return paddle.to_tensor([query_input_ids], dtype=paddle.int64), \
           paddle.to_tensor([query_token_type_ids], dtype=paddle.int64)

if __name__ == '__main__':
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "deep_emb_infer").set("spark.ui.port", "4060")
    spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    pd_spu = spark.sql("""
        select distinct spu_sn, spu_name
        from dm_recommend.dws_recommend_dj_frxs_skusn_details_di 
        where status = 'UP'
    """).toPandas()

    pd_spu['spu_name'] = pd_spu['spu_name'].apply(lambda x:remove(x))
    print("pd_spu cnt : {}".format(len(pd_spu)),flush=True)

    paddle.set_device(args.device)
    pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained('roberta-wwm-ext-large')
    tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext-large')

    print("loading model...",flush=True)
    model = BertMetric(pretrained_model, args.emb_size, args.cls_size)
    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    model.eval()
    sentences = pd_spu['spu_name'].values
    print("model input data size : {}".format(len(sentences)), flush=True)
    vec_result = []
    for s in sentences:
        query_input_ids, query_token_type_ids = convert_example(s, tokenizer, max_seq_length=args.max_seq_length, is_test=True)
        vec_out = model.predict_emb(query_input_ids, query_token_type_ids)
        vec_result.append(vec_out[0])
    print("model output data size : {}".format(len(vec_result)), flush=True)

    mkl.get_max_threads()
    d = 1024
    index = faiss.IndexFlatL2(d)  # build the index
    #print(index.is_trained)  # 表示索引是否需要训练的布尔值
    index.add(np.array(vec_result))  # add vectors to the index
    #print(index.ntotal)
    D, I = index.search(np.array(vec_result), 13)  # actual search
    print("emb similar output cnt : {}".format(len(I)),flush=True)
    pool = redis.ConnectionPool(host=args.redis_host, port=args.redis_port, password=args.redis_password, db=args.redis_db)
    #print("redis: {}:{} {} {}".format(args.redis_host, args.redis_port, args.redis_password, args.redis_db),flush=True)
    r = redis.Redis(connection_pool=pool)
    pipe = r.pipeline()  # 创建一个管道
    for i in range(0, len(I)):
        curr_spusn = pd_spu.spu_sn.values[i]
        curr_spusn_similar_str = ''
        for similar_index in I[i][1:]:
            curr_spusn_similar_str += pd_spu.spu_sn.values[similar_index] + ','
        curr_spusn_similar_str = curr_spusn_similar_str.strip(',')
        pipe.set('dj_similar:{}'.format(curr_spusn), curr_spusn_similar_str)
        pipe.expire('dj_similar:{}'.format(curr_spusn) , args.exp_seconds)
    pipe.execute()
    sc.stop()
