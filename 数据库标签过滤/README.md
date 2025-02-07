共在4个数据库上做了标签过滤，包括vbase、pase、milvus和vearch。如果需要还可以在elastic search上继续做

### vbase
只在单线程下实验，使用的索引为hnsw

### pase
只在单线程下实验，使用的索引是ivf_flat，由于pase自身的限制，向量维度不能大于512，故enron和gist数据集无法测试

### milvus
在单线程、16线程和16进程下，使用python sdk连接单节点，索引为ivf_flat，5_1000_sift的意思是，nlist为1000，nprobe为5

### vearch
在单线程和16线程下，使用python sdk连接单节点，索引为ivf_flat

