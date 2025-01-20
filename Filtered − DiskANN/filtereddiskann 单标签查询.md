filted-diskann分别在单线程和十六线程下的实验结果

### 数据集简介
| Dataset |  Dimension | Base Size | Query Size | Type |
| --- | --- | --- | --- | --- |
| audio | 192 | 53,387 | 200 | Audio + Attributes |
| enron | 1369 | 94,987 | 200 | Text + Attributes |
| gist1M | 960 | 1,000,000 | 1,000 | Image + Attributes |
| glove-100 | 100 | 1,183,514 | 10,000 | Text + Attributes |
| msong | 420 | 992,272 | 200 | Audio + Attributes |
| sift1M | 128 | 1,000,000 | 10,000 | Image + Attributes |




对于filtereddiskann需要构建两种索引分别进行测试

（1）filteredvamana

（2）stitchedvamana



### 实验结果：
+ **Ls**：搜索参数L。
+ **QPS (Queries per Second)**：每秒查询数。
+ **Avg dist cmps (Average distance comparisons)**：平均距离比较数。
+ **Mean Latency (mus)**：平均延迟，表示处理每个查询所需的平均时间（单位是微秒）。
+ **99.9 Latency**：99.9%的查询的最大延迟，表示查询响应时间的上界。
+ **Recall@10**：前10个结果的召回率。

