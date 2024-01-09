# 1 模型跑通
```shell
git clone https://github.com/Tongjilibo/bert4torch.git

git checkout v0.2.0

# 数据集下载：
http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz

# 预训练模型下载:
https://huggingface.co/bert-base-chinese/tree/main  

export PYTHONPATH=*/bert4torch:$PYTHONPATH

cd */bert4torch/examples/sequence_labeling

python3 task_sequence_labeling_ner_crf.py
```

# 2 transformer 家族介绍
- [参考链接](https://transformers.run/back/transformer/)

# 3 crf 原理介绍
- [参考链接](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html)

# 4 代码详解