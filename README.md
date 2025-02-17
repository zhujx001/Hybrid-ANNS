# 维护说明

## 论文

阅读新的论文之后，将论文标题**添加到论文汇总文档**，说明论文思路，如果论文当中提出了新的比较指标，或者其他一些值得借鉴的东西，将其添加到待处理文档。包括论文当中新的比较指标，如果论文无法添加到目前的论文框架，请在待处理文档说明原因，建议

论文汇总文档：https://www.yuque.com/g/passers-gexqk/amxlgr/cy14f6qo5cx5usqf/collaborator/join?token=YneAwzjaBgOB2LvK&source=doc_collaborator&goto=%2Fpassers-gexqk%2Famxlgr%2Fcy14f6qo5cx5usqf%2Fedit# 邀请你共同编辑文档《论文汇总》



待处理文档：https://www.yuque.com/g/passers-gexqk/amxlgr/ms5w5brveuw8wqnu/collaborator/join?token=ZzIrMd6RgAqZtF9D&source=doc_collaborator&goto=%2Fpassers-gexqk%2Famxlgr%2Fms5w5brveuw8wqnu%2Fedit# 邀请你共同编辑文档《待处理》



问题划分文档：https://www.yuque.com/g/passers-gexqk/amxlgr/xoc8zk8xs2zgyqqs/collaborator/join?token=SZvbcblJwL7fo7Bk&source=doc_collaborator&goto=%2Fpassers-gexqk%2Famxlgr%2Fxoc8zk8xs2zgyqqs%2Fedit# 邀请你共同编辑文档《问题划分》

论文阅读文档：https://www.yuque.com/g/passers-gexqk/amxlgr/bkg13oixmzcecui7/collaborator/join?token=u4duSXKmYSXalucQ&source=doc_collaborator&goto=%2Fpassers-gexqk%2Famxlgr%2Fbkg13oixmzcecui7%2Fedit# 邀请你共同编辑文档《论文阅读》


标签设置文档：https://www.yuque.com/g/passers-gexqk/amxlgr/izdnyuoepfglffbo/collaborator/join?token=iQClGjt7WHTH0wWX&source=doc_collaborator&goto=%2Fpassers-gexqk%2Famxlgr%2Fizdnyuoepfglffbo%2Fedit# 邀请你共同编辑文档《标签设置》

# 实验结果存储说明文档

## 1. 范围过滤实验（Range Filter）

### 数据集信息

| 数据集          | 向量类型 | 维度 | 属性类型           |
| --------------- | -------- | ---- | ------------------ |
| deep            | 图像     | 96   | 合成数据           |
| youtube8m-audio | 音频     | 128  | 发布时间、观看次数 |
| youtube8m-rgb   | 视频     | 1024 | 点赞数、评论数     |
| wit             | 图像     | 1024 | 图片尺寸           |

### 结果存储规范

存储位置：`Results/Range-Filter/`

文件命名格式：`{数据集名}_{算法名}.csv`

子目录：

- `range0/`：查询范围为原数据大小的 2^0 倍
- `range1/`：查询范围为原数据大小的 2^(-1) 倍

性能指标：

- QPS (Queries Per Second)
- Recall@10



## 2. 标签过滤实验（Label Filter）

### 数据集信息

**标签数据在 data/lable 文件夹下**

| 数据集 | 对象数量  | 特征向量维度 | 查询数量 | 类型  | 下载文件                                                     |
| ------ | --------- | ------------ | -------- | ----- | ------------------------------------------------------------ |
| SIFT1M | 1,000,000 | 128          | 10,000   | Image | [sift.tar.gz](http://corpus-texmex.irisa.fr/) (161MB)        |
| GIST1M | 1,000,000 | 960          | 1,000    | Image | [gist.tar.gz](http://corpus-texmex.irisa.fr/) (2.6GB)        |
| Audio  | 53,387    | 192          | 200      | Audio | [audio.tar.gz](https://drive.google.com/file/d/1fJvLMXZ8_rTrnzivvOXiy_iP91vDyQhs/view) (26MB) |
| Msong  | 992,272   | 420          | 200      | Audio | [msong.tar.gz](https://drive.google.com/file/d/1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj/view) (1.4GB) |
| Enron  | 94,987    | 1,369        | 200      | Text  | [enron.tar.gz](https://drive.google.com/file/d/1TqV43kzuNYgAYXvXTKsAG1-ZKtcaYsmr/view) (51MB) |

### 结果存储规范

存储位置：`Results/Label-Filter/`

存储方式：

1. 在 Label-Filter 目录下，为每个算法新建独立文件夹
2. 在对应算法文件夹中保存该算法的实验结果

命名规范：

1. 算法文件夹：使用算法名命名
2. 结果文件：`{数据集名}_{算法名}.csv`

## 注意事项

1. 所有 CSV 文件必须包含表头
2. 数值型数据使用浮点数表示，保留两位小数
3. 确保文件命名符合规范，使用下划线分隔数据集名和算法名
4. 每个实验结果需要单独保存为 CSV 文件

# 未提到的在后面补充
