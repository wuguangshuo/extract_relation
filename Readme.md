# 工业知识图谱关系抽取-高端装备制造知识图谱自动化构建

-----

比赛链接:

https://www.datafountain.cn/competitions/584

-----
本文模型gp+双仿射,复赛成绩第八

文件的结构如下

```
│  dataloader.py
│  data_gen.py#首先执行生成数据集
│  data_genccl.py#首先执行生成数据集，外部ccl数据
│  gpNet.py
│  main.py#程序入口
│  predict.py
│  Readme.md
│  utils.py
│  vote.py
│
├─data#数据集存放位置
│      evalA.json
│      evalA2.json
│      schemas.json
│      submit_example_A.json
│      train.json
│      train2.json
│      train_ccl.json
│      train_ccl2.json
│
└─pretrain_model#模型存放位置
        config.json
        special_tokens_map.json
        tokenizer.json
        tokenizer_config.json
        vocab.txt


```

第二名代码链接:

https://github.com/yandun72/2022_CCF_BDCI_Relation_Extraction_Rank2

第二名在代码中使用了55个模型，本次比赛个人只用了9个（比赛失败原因之一)

GRTE模型和gplinker融合提升较大,GRTE文章链接：

https://arxiv.org/abs/2109.06705

关系抽取方法总结文章

https://zhuanlan.zhihu.com/p/564307572
