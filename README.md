# Hybrid-ANNS


### 基本数据集

| Dataset    | Object Num  | Feature Vector Dim | Query Num | Type            | Download (Vector)                                            |
| ---------- | ----------- | ------------------ | --------- | --------------- | ------------------------------------------------------------ |
| SIFT1M     | 1,000,000   | 128                | 10,000    | Image | [sift.tar.gz](http://corpus-texmex.irisa.fr/) (161MB)        |
| GIST1M     | 1,000,000   | 960                | 1,000     | Image | [gist.tar.gz](http://corpus-texmex.irisa.fr/) (2.6GB)        |
| GloVe      | 1,183,514   | 100                | 10,000    | Text  | [glove-100.tar.gz](http://downloads.zjulearning.org.cn/data/glove-100.tar.gz) (424MB) |
| Crawl      | 1,989,995   | 300                | 10,000    | Text  | [crawl.tar.gz](http://downloads.zjulearning.org.cn/data/crawl.tar.gz) (1.7GB) |
| Audio      | 53,387      | 192                | 200       | Audio | [audio.tar.gz](https://drive.google.com/file/d/1fJvLMXZ8_rTrnzivvOXiy_iP91vDyQhs/view) (26MB) |
| Msong      | 992,272     | 420                | 200       | Audio | [msong.tar.gz](https://drive.google.com/file/d/1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj/view) (1.4GB) |
| Enron      | 94,987      | 1369               | 200       | Text  | [enron.tar.gz](https://drive.google.com/file/d/1TqV43kzuNYgAYXvXTKsAG1-ZKtcaYsmr/view) (51MB) |
| UQ-V       | 1,000,000   | 256                | 10,000    | Video | [uqv.tar.gz](https://drive.google.com/file/d/1HIdQSKGh7cfC7TnRvrA2dnkHBNkVHGsF/view) (800MB) |
| Paper      | 2,029,997   | 200                | 10,000    | Text  | [paper.tar.gz](https://drive.google.com/file/d/1t4b93_1Viuudzd5D3I6_9_9Guwm1vmTn/view) (1.41GB) |
| BIGANN100M | 100,000,000 | 128                | 10,000    | Image | [bigann100m.tar.gz](https://big-ann-benchmarks.com/) (9.2GB) |
### 算法
#### NHQ 数据集使用说明

[代码仓库](https://github.com/KGLab-HDU/TKDE-under-review-Native-Hybrid-Queries-via-ANNS)
[论文](https://arxiv.org/abs/2203.13601)

NHQ在基础数据集上为每个对象添加属性，如为 SIFT1M 上的每张图像添加日期、位置、大小等属性，以形成一个具有特征向量和一组属性的对象。之后查询的真值文件通过论文中的`Definition4 ` 暴力计算得出，下面给出了已有标签文件和对应真值文件的链接。

所有原始对象和查询对象都转换为 fvecs 格式，而 groundtruth 数据则转换为 ivecs 格式



| Dataset    | NHQ Attributes Download                                      | NHQ Ground Truth Download                                    |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SIFT1M     | [sift_attribute.tar.gz](https://drive.google.com/file/d/15sflYLREoqHJGJCuBpiE1UOHad60_GKK/view) | [label_sift_groundtruth.ivecs](https://drive.google.com/file/d/1MVw1QmhQ_TnfhAV3Np-PDH9GNnH3Vm0w/view) |
| GIST1M     | [gist_attribute.tar.gz](https://drive.google.com/file/d/1PFeQev-7jywvdOVXy5ubMhltbH5sFDRx/view) | [label_gist_groundtruth.ivecs](https://drive.google.com/file/d/1KkeEbEglX6plVy4rT4GKkhCTKnOQ9jbh/view) |
| GloVe      | [glove-100_attribute.tar.gz](https://drive.google.com/file/d/10bIhmw1RC4Bk6cpJuWRli1WuwbALEKuK/view) | [label_glove_groundtruth.ivecs](https://drive.google.com/file/d/1LHbXi6Aapvnxp68aGZF1DV3kXy23bFE_/view) |
| Crawl      | [crawl_attribute.tar.gz](https://drive.google.com/file/d/1d1TURrWxYAELvfiBNermEv0iiyTxAWF6/view) | label_crawl_groundtruth.ivecs                                |
| Audio      | [audio_attribute.tar.gz](https://drive.google.com/file/d/1IsAGjhDSu2xrh2w16iVBEfw9vbOCRYjq/view) | [label_audio_groundtruth.ivecs](https://drive.google.com/file/d/1WeBC4_Aw2pfM_DlFaJUuM0GRuLAPCI3P/view) |
| Msong      | [msong_attribute.tar.gz](https://drive.google.com/file/d/1jVpJaT5GRjxRzj4C3KSsev0clQIOEplZ/view) | [label_msong_groundtruth.ivecs](https://drive.google.com/file/d/1LFWshAIoQLYJx68toTQBaoIOBZDfExue/view) |
| Enron      | [enron_attribute.tar.gz](https://drive.google.com/file/d/1tbVjQlUlFS321CxW9_hfqUf4JUiXdmLi/view) | [label_enron_groundtruth.ivecs](https://drive.google.com/file/d/1F5eZwG_u8S3StwPOnlmrHqmoFCoaGKVB/view) |
| UQ-V       | [uqv_attribute.tar.gz](https://drive.google.com/file/d/1YN6VuLPw_u9cFREXS6jgApYjCTmzmZtv/view) | [label_uqv_groundtruth.ivecs](https://drive.google.com/file/d/1o05Iq9Q_omnHosWnrwRQBYXtN4n7nu5o/view) |
| Paper      | [paper_attribute.tar.gz](https://drive.google.com/file/d/1arpB0oZne3tmRCUfTfzQmIfvWVP_kuKY/view) | [label_paper_groundtruth.ivecs](https://drive.google.com/file/d/1arpB0oZne3tmRCUfTfzQmIfvWVP_kuKY/view) |
| BIGANN100M | [bigann100m_attribute.tar.gz](https://drive.google.com/file/d/1arpB0oZne3tmRCUfTfzQmIfvWVP_kuKY/view) | label_bigann100m_groundtruth.ivecs                           |
---

#### Filtered−DiskANN 数据集使用说明

[代码仓库](https://github.com/Microsoft/DiskANN)

[论文](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)

Attributes查询标签是单标签

groundtruth由代码中compute_groundtruth_for_filters生成

| Dataset | Filtered−DiskANN Attributes Download                         |
| ------- | ------------------------------------------------------------ |
| SIFT1M  | [sift_attribute.tar.gz](https://drive.google.com/file/d/1HV97n6IvotW9jwZlC4vy-uOLtaiCDDF7/view?usp=drive_link) |
| GIST1M  | [gist_attribute.tar.gz](https://drive.google.com/file/d/1tG2_XD9iWh-yBzFsroGq_5qwOZUT5A5Q/view?usp=drive_link) |
| GloVe   | [glove-100_attribute.tar.gz](https://drive.google.com/file/d/1Hhec4cdb3epvwIfuzjjoY-Quim5kPMGe/view?usp=drive_link) |
| Audio   | [audio_attribute.tar.gz](https://drive.google.com/file/d/1_6WFWUTsimcqjfm5UCwYlYu3OqTZF6Yi/view?usp=drive_link) |
| Msong   | [msong_attribute.tar.gz](https://drive.google.com/file/d/1JB8erGu2m3SB-vv23bRWYAElYBPnFadU/view?usp=drive_link) |
| Paper   | [paper_attribute.tar.gz](https://drive.google.com/file/d/1s8xUZD6mx3JvuHJXPcRRSqkxAXuiyagg/view?usp=drive_link) |


#### RoarGraph 数据集使用说明
[代码仓库](https://github.com/matchyc/RoarGraph)

[论文](https://arxiv.org/pdf/2408.08933)

数据集中数据无属性标签，而是基础数据和查询数据是不同属性的数据。

1.text-to-image数据集：
base.10M.fbin: https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin
query.train.10M.fbin： https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
query.10k.fbin： https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
gt.10k.ibin： https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin

2.clip-webvid-2.5M数据集
base.2.5M.fbin： https://zenodo.org/records/11090378/files/clip.webvid.base.2.5M.fbin
query.train.2.5M.fbin： https://zenodo.org/records/11090378/files/webvid.query.train.2.5M.fbin
query.10k.fbin： https://zenodo.org/records/11090378/files/webvid.query.10k.fbin
gt.10k.ibin： https://zenodo.org/records/11090378/files/webvid.gt.10k.ibin


