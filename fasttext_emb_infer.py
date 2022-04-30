#-*- coding: utf-8 -*-
import findspark
findspark.init()
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

sparkConf = SparkConf()
sparkConf.set("spark.app.name", "fasttext_emb_infer").set("spark.ui.port", "4060")
spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
sc = spark.sparkContext

import jieba
import os
import pandas as pd
from pyhive import hive
import re
import fasttext
import faiss
import mkl
import numpy as np
import argparse
import json
import redis
from model import BertMetric


# 若下面两条命令已加入 bashrc，可注释掉
os.system('source /opt/Bigdata/client/bigdata_env')
os.system('kinit -kt /opt/Bigdata/client/user.keytab GPSearch')

parser = argparse.ArgumentParser()
parser.add_argument("--redis_host", type=str, default='10.255.24.40')
parser.add_argument("--redis_password", type=str, default='S4wKxoLGRo')
parser.add_argument("--redis_port", type=int, default=6379)
parser.add_argument("--redis_db", type=int, default=14)
parser.add_argument("--exp_seconds", type=int, default=7*24*3600)
args = parser.parse_args()

stopwords=pd.read_csv("./stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
remove_words = ['【福利秒杀】','【每日福利】','【福利爆款】','【专柜品质】','【1元秒杀】','【直播专用1元秒杀】','【','】','源本']
#分词去停用词，并整理为fasttext要求的文本格式
def preprocess_for_infer(content_lines, sentences):
    for line in content_lines:
        for r in remove_words:
            line = line.replace(r, '')
        commas = re.findall('\[[^()]*\]', line)
        for c in commas:
            line = line.replace(c, '')
        try:
            segs = jieba.lcut(line)
            segs = list(filter(lambda x:len(x)>1, segs))
            segs = list(filter(lambda x:x not in stopwords, segs))
            sentences.append(" ".join(segs))
        except Exception as e:
            print(line)
            continue

if __name__ == '__main__':
    # host = "10.67.2.104"  # hive host,可以执行beeline查看
    # port = 10000  # 端口
    # auth = "KERBEROS"  # 认证方式为kerberos
    # kerberos_service_name = "hive"  # 默认为为hive,可以再hive_site.xml里面查看
    # kerberos_service_host = 'hadoop.hadoop.com'  # 默认为hadoop.hadoop.com,可以再hive_site.xml里面查看
    # maxbufsize = 1024 * 1024 * 100
    print("start load hive data...",flush=True)
    pd_spu = spark.sql("""
        select distinct spu_sn, spu_name
        from dm_recommend.dws_recommend_dj_frxs_skusn_details_di 
        where status = 'UP'
    """).toPandas()
    # with hive.connect(host=host, port=port, auth=auth, database='dm_recommend', kerberos_service_name=kerberos_service_name,
    #                   maxbufsize=maxbufsize) as conn:
    #     pd_spu = pd.read_sql("""
    #         select *
    #         from dm_recommend.dws_recommend_dj_frxs_skusn_details_di
    #         where status = 'UP'
    #     """, conn)
    print("pd_spu cnt : {}".format(len(pd_spu)),flush=True)
    print("loading fasttext model: {}".format('{}/fasttext.bin'.format(os.getcwd())),flush=True)
    model = fasttext.load_model('{}/fasttext.bin'.format(os.getcwd()))
    sentences = []
    preprocess_for_infer(pd_spu.spu_name, sentences)
    vec_result = []
    for s in sentences:
        vec_result.append(model.get_sentence_vector(s))
    print("emb input cnt : {}".format(len(vec_result)),flush=True)
    mkl.get_max_threads()
    d = 300
    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)  # 表示索引是否需要训练的布尔值
    index.add(np.array(vec_result))  # add vectors to the index
    print(index.ntotal)
    D, I = index.search(np.array(vec_result), 13)  # actual search
    print("emb similar output cnt : {}".format(len(I)),flush=True)
    pool = redis.ConnectionPool(host=args.redis_host, port=args.redis_port, password=args.redis_password, db=args.redis_db)
    print("redis: {}:{} {} {}".format(args.redis_host, args.redis_port, args.redis_password, args.redis_db),flush=True)
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




