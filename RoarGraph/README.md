RoarGraph
其中所有数据集都可以用prepare_data.sh脚本来下载整合
使用方法：bash prepare_data.sh t2i-10M/clip-webvid-2.5M/laion-10M
数据集下载地址：
一：t2i-10M数据集

base.10M.fbin： https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin

query.train.10M.fbin： https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin

query.10k.fbin： https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin

gt.10k.ibin： https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin

二：clip-webvid-2.5M数据集

base.2.5M.fbin： https://zenodo.org/records/11090378/files/clip.webvid.base.2.5M.fbin

query.train.2.5M.fbin： https://zenodo.org/records/11090378/files/webvid.query.train.2.5M.fbin

query.10k.fbin： https://zenodo.org/records/11090378/files/webvid.query.10k.fbin

gt.10k.ibin： https://zenodo.org/records/11090378/files/webvid.gt.10k.ibin

三：laion-10M数据集：

该数据集的基础数据集和训练数据集由10个1M的数据集组合而成，推荐直接用prepare_data脚本来下载整合
基础数据集：https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/img_emb_${i}.npy
训练数据集：https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/texts/text_emb_${i}.npy
query.10k.fbin ：https://zenodo.org/records/11090378/files/laion.query.10k.fbin
gt.10k.ibin ： https://zenodo.org/records/11090378/files/laion.gt.10k.ibin
