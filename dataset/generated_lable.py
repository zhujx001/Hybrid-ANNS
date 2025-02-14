import numpy as np
from scipy import stats
import pandas as pd

class LabelGenerator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        # 定义每个维度的标签范围，确保不重叠
        self.label_ranges = [
            (1, 15),      # 第1维：1-15，均匀分布
            (101, 150),   # 第2维：101-150，长尾分布
            (201, 250),   # 第3维：201-250，正态分布
            (301, 350),   # 第4维：301-350，分层长尾
            (401, 450),   # 第5维：401-450，幂律分布
            (501, 600),   # 第6维：501-600，选择性维度
            (601, 700),   # 第7维：601-700，关联维度
            (801, 802),   # 第8维：801-802，稀疏维度
            (901, 910),   # 第9维：901-910，均匀分布
            (1001, 1010), # 第10维：1001-1010，均匀分布
            (1101, 1110), # 第11维：1101-1110，均匀分布
            (1201, 1210), # 第12维：1201-1210，均匀分布
            (1301, 1310), # 第13维：1301-1310，均匀分布
            (1401, 1410), # 第14维：1401-1410，均匀分布
            (1501, 1510), # 第15维：1501-1510，均匀分布
        ]
    
    def generate_uniform_labels(self, start, end):
        """生成均匀分布的标签"""
        return np.random.randint(start, end + 1, self.n_samples)
    
    def generate_normal_labels(self, start, end):
        """生成正态分布的标签"""
        mu = (end - start) / 2
        sigma = (end - start) / 6
        labels = np.random.normal(mu, sigma, self.n_samples)
        labels = np.clip(labels, 0, end - start)
        labels = labels.astype(int) + start
        return labels
    
    def generate_long_tail_labels(self, start, end):
        """生成长尾分布的标签"""
        x = np.random.exponential(0.5, self.n_samples)
        x = x / np.max(x)
        labels = (x * (end - start)).astype(int) + start
        return np.clip(labels, start, end)
    
    def generate_power_law_labels(self, start, end):
        """生成幂律分布的标签"""
        alpha = 2.5
        x = np.random.power(alpha, self.n_samples)
        labels = (x * (end - start)).astype(int) + start
        return np.clip(labels, start, end)
    
    def generate_selective_labels(self, start, end):
        """生成选择性维度的标签"""
        labels = np.zeros(self.n_samples)
        range_size = end - start + 1
        
        # 将范围分成100份
        ranges = [
            (start, start + range_size//10),                    # 1-10: 1%样本
            (start + range_size//10, start + range_size//4),    # 11-25: 25%样本
            (start + range_size//4, start + 3*range_size//4),   # 26-75: 50%样本
            (start + 3*range_size//4, end)                      # 76-99: 24%样本
        ]
        
        # 计算每个范围的样本数
        n1 = int(0.01 * self.n_samples)  # 1%
        n2 = int(0.25 * self.n_samples)  # 25%
        n3 = int(0.50 * self.n_samples)  # 50%
        n4 = self.n_samples - (n1 + n2 + n3)  # 剩余样本
        
        # 为每个范围生成标签
        labels[:n1] = np.random.randint(ranges[0][0], ranges[0][1], n1)
        labels[n1:n1+n2] = np.random.randint(ranges[1][0], ranges[1][1], n2)
        labels[n1+n2:n1+n2+n3] = np.random.randint(ranges[2][0], ranges[2][1], n3)
        labels[n1+n2+n3:] = np.random.randint(ranges[3][0], ranges[3][1], n4)
        
        np.random.shuffle(labels)
        return labels.astype(int)
    
    def generate_correlated_labels(self, primary_labels, start, end):
        """生成与第1维相关联的标签"""
        labels = np.zeros(self.n_samples)
        step = (end - start) // 100
        
        for i in range(1, 16):
            mask = (primary_labels == i)
            n_samples = np.sum(mask)
            
            # 基本范围：[i*3, i*3+2]映射到新范围
            base_start = start + (i*3)*step
            base_end = start + (i*3+2)*step
            base_values = np.random.randint(base_start, base_end + 1, n_samples)
            
            # 添加10%的重叠值
            overlap_samples = int(0.1 * n_samples)
            if overlap_samples > 0:
                overlap_indices = np.random.choice(n_samples, overlap_samples, replace=False)
                neighbor_category = (i % 15) + 1
                neighbor_start = start + (neighbor_category*3)*step
                neighbor_end = start + (neighbor_category*3+2)*step
                base_values[overlap_indices] = np.random.randint(
                    neighbor_start,
                    neighbor_end + 1,
                    overlap_samples
                )
            
            labels[mask] = base_values
            
        return labels.astype(int)
    
    def generate_sparse_labels(self, start):
        """生成稀疏维度的标签（99%为0，1%为1）"""
        labels = np.zeros(self.n_samples) + start  # 默认值为start（表示0）
        n_positive = int(0.01 * self.n_samples)  # 1%的样本为正
        positive_indices = np.random.choice(self.n_samples, n_positive, replace=False)
        labels[positive_indices] = start + 1  # 正样本值为start+1
        return labels.astype(int)
    
    def generate_all_labels(self):
        """生成所有维度的标签"""
        labels = np.zeros((self.n_samples, 15))
        
        for i, (start, end) in enumerate(self.label_ranges):
            if i == 0:  # 第1维：均匀分布
                labels[:, i] = self.generate_uniform_labels(start, end)
            elif i == 1:  # 第2维：长尾分布
                labels[:, i] = self.generate_long_tail_labels(start, end)
            elif i == 2:  # 第3维：正态分布
                labels[:, i] = self.generate_normal_labels(start, end)
            elif i == 3:  # 第4维：分层长尾
                labels[:, i] = self.generate_long_tail_labels(start, end)
            elif i == 4:  # 第5维：幂律分布
                labels[:, i] = self.generate_power_law_labels(start, end)
            elif i == 5:  # 第6维：选择性维度
                labels[:, i] = self.generate_selective_labels(start, end)
            elif i == 6:  # 第7维：关联维度
                labels[:, i] = self.generate_correlated_labels(labels[:, 0], start, end)
            elif i == 7:  # 第8维：稀疏维度
                labels[:, i] = self.generate_sparse_labels(start)
            else:  # 第9-15维：均匀分布
                labels[:, i] = self.generate_uniform_labels(start, end)
        
        return labels.astype(int)
    
    def save_labels(self, labels, filename):
        """保存标签到文本文件"""
        np.savetxt(filename, labels, fmt='%d', delimiter=' ')

class QueryGenerator:
    def __init__(self, dataset_labels, label_ranges):
        self.dataset_labels = dataset_labels
        self.label_ranges = label_ranges
        
    def generate_all_queries(self, n_queries):
        """生成所有查询集"""
        queries = []
        
        # 查询集1：基本实验，单属性构建并搜索，使用属性1
        q1 = np.random.choice(self.dataset_labels[:, 0], n_queries)
        queries.append(("query_set_1", q1))
        
        # 查询集2-1：多属性构建（均匀分布），使用属性1
        q2_1 = np.random.choice(self.dataset_labels[:, 0], n_queries)
        queries.append(("query_set_2_1", q2_1))
        
        # 查询集2-2：混合多属性构建，使用属性1
        q2_2 = np.random.choice(self.dataset_labels[:, 0], n_queries)
        queries.append(("query_set_2_2", q2_2))
        
        # 查询集3：选择性实验 (3-1到3-4)
        percentiles = [0.01, 0.25, 0.50, 0.75]
        range_size = self.label_ranges[5][1] - self.label_ranges[5][0] + 1
        ranges = [
            (self.label_ranges[5][0], self.label_ranges[5][0] + range_size//10),                    # 1-10: 查询集3-1
            (self.label_ranges[5][0] + range_size//10, self.label_ranges[5][0] + range_size//4),    # 11-25: 查询集3-2
            (self.label_ranges[5][0] + range_size//4, self.label_ranges[5][0] + 3*range_size//4),   # 26-75: 查询集3-3
            (self.label_ranges[5][0] + 3*range_size//4, self.label_ranges[5][1])                    # 76-99: 查询集3-4
        ]
        
        for i, (start, end) in enumerate(ranges, 1):
            q = np.random.randint(start, end, n_queries)
            queries.append((f"query_set_3_{i}", q))
        
        # 查询集4：稀疏属性实验
        q4 = np.zeros(n_queries) + self.label_ranges[7][0]
        mask = np.random.random(n_queries) < 0.01
        q4[mask] = self.label_ranges[7][1]
        queries.append(("query_set_4", q4))
        
        # 查询集5：不同分布实验
        # 长尾分布
        q5_1 = np.random.choice(self.dataset_labels[:, 1], n_queries)
        queries.append(("query_set_5_1", q5_1))
        
        # 正态分布
        q5_2 = np.random.choice(self.dataset_labels[:, 2], n_queries)
        queries.append(("query_set_5_2", q5_2))
        
        # 幂律分布
        q5_3 = np.random.choice(self.dataset_labels[:, 4], n_queries)
        queries.append(("query_set_5_3", q5_3))
        
        # 均匀分布
        q5_4 = np.random.choice(self.dataset_labels[:, 0], n_queries)
        queries.append(("query_set_5_4", q5_4))
        
        # 查询集6：多标签，三个均匀分布标签
        q6 = np.zeros((n_queries, 3))
        q6[:, 0] = np.random.choice(self.dataset_labels[:, 0], n_queries)  # 属性1
        q6[:, 1] = np.random.choice(self.dataset_labels[:, 8], n_queries)  # 属性9
        q6[:, 2] = np.random.choice(self.dataset_labels[:, 9], n_queries)  # 属性10
        queries.append(("query_set_6", q6))
        
        # 查询集7：多标签选择性实验 (7-1到7-4)
        ranges = [
            (self.label_ranges[5][0], self.label_ranges[5][0] + range_size//10),                    # 1-10
            (self.label_ranges[5][0] + range_size//10, self.label_ranges[5][0] + range_size//4),    # 11-25
            (self.label_ranges[5][0] + range_size//4, self.label_ranges[5][0] + 3*range_size//4),   # 26-75
            (self.label_ranges[5][0] + 3*range_size//4, self.label_ranges[5][1])                    # 76-99
        ]
        
        for i, (start, end) in enumerate(ranges, 1):
            q = np.zeros((n_queries, 3))
            # 两个均匀分布属性
            q[:, 0] = np.random.choice(self.dataset_labels[:, 0], n_queries)  # 属性1
            q[:, 1] = np.random.choice(self.dataset_labels[:, 8], n_queries)  # 属性9
            # 选择性属性
            q[:, 2] = np.random.randint(start, end, n_queries)
            queries.append((f"query_set_7_{i}", q))
        
        return queries
    
    def save_queries(self, queries, base_filename):
        """保存查询集到文本文件"""
        for name, query in queries:
            filename = f"{base_filename}_{name}.txt"
            if query.ndim == 1:
                np.savetxt(filename, query.reshape(-1, 1), fmt='%d', delimiter=' ')
            else:
                np.savetxt(filename, query, fmt='%d', delimiter=' ')

# 使用示例
if __name__ == "__main__":
    # 生成数据集标签
    n_samples = 100000
    generator = LabelGenerator(n_samples)
    dataset_labels = generator.generate_all_labels()
    
    # 保存数据集标签
    generator.save_labels(dataset_labels, "dataset_labels.txt")
    
    # 生成并保存所有查询集
    n_queries = 1000
    query_generator = QueryGenerator(dataset_labels, generator.label_ranges)
    queries = query_generator.generate_all_queries(n_queries)
    query_generator.save_queries(queries, "query")
    
    # 打印统计信息
    print("数据集标签统计信息：")
    for i, (start, end) in enumerate(generator.label_ranges):
        unique, counts = np.unique(dataset_labels[:, i], return_counts=True)
        print(f"\n维度 {i+1}:")
        print(f"范围: [{start}, {end}]")
        print(f"不同类别数: {len(unique)}")
        print(f"分布: {np.bincount(dataset_labels[:, i].astype(int) - start)[:5]}...")
