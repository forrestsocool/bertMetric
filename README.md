# bertMetric
使用paddlenlp实现的arcface loss，并将其应用在短文本向量化中实现高质量Embedding抽取
* 通过短文本分类任务 + arcface loss 完成finetune预训练模型
* 使用fastext实现Embedding抽取作为baseline
* 使用faiss实现最相近的K个item计算，进行线上AB实验对比
* pyspark拉取数据

# References
https://github.com/deepinsight/insightface

https://github.com/auroua/InsightFace_TF

https://github.com/MuggleWang/CosFace_pytorch

# pretrained model
pretrained model 默认使用roberta-wwm-ext-large 可以按需求替换