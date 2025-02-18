import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time
from tqdm import tqdm
import psutil  # 用于监控CPU使用情况

def read_fvecs(filename, mmap=False):
    """
    读取fvecs文件，返回(N, D)的float32数组
    参数:
        filename: fvecs文件路径
        mmap: 是否使用内存映射（对于大文件有用）
    """
    if mmap:
        with open(filename, 'rb') as f:
            # 读取第一个向量的维度
            dim = np.fromfile(f, dtype=np.int32, count=1)[0]
            vec_size = 4 * (dim + 1)  # 每个向量的字节数：4*（维度+1）
            # 估算向量数量
            filesize = os.path.getsize(filename)
            num_vecs = filesize // vec_size
            
            # 使用内存映射
            mm = np.memmap(filename, dtype=np.float32, mode='r', 
                           offset=0, shape=(num_vecs, dim + 1))
            # 提取所有向量（跳过每个向量的第一个元素）
            vecs = np.zeros((num_vecs, dim), dtype=np.float32)
            for i in range(num_vecs):
                vecs[i] = mm[i, 1:dim+1]
            return vecs
    else:
        with open(filename, 'rb') as f:
            data = f.read()
        
        arr = np.frombuffer(data, dtype=np.int32)
        offset = 0
        vectors = []
        data_bytes = memoryview(data)
        
        while offset < len(arr):
            D = arr[offset]  # 获取维度
            offset += 1
            float_data = np.frombuffer(data_bytes[offset * 4:(offset + D) * 4], dtype=np.float32)  # 获取向量数据
            offset += D
            vectors.append(float_data)
        
        return np.vstack(vectors)  # 返回(N, D)的numpy数组

def write_ivecs(filename, data):
    """
    将data写为ivecs格式：
    data为(N, K)的int32数组，每行: K(int32) + K个int32值
    """
    N, K = data.shape
    with open(filename, 'wb') as f:
        for i in range(N):
            row = data[i]
            f.write(np.int32(K).tobytes())
            f.write(row.astype(np.int32).tobytes())

def compute_distances(query_vec, base_vecs):
    """
    计算 query_vec 与 base_vecs 中所有向量的 L2距离^2（欧式距离平方）。
    返回一个长度为 base_vecs.shape[0] 的 float32 数组
    """
    diff = base_vecs - query_vec
    dist = np.sum(diff * diff, axis=1)  # 欧几里得距离的平方
    return dist

def find_top_k(distances, k=100):
    """
    找到距离最小的 k 个索引
    """
    n = len(distances)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n < k:
        k = n
    idx = np.argpartition(distances, k)[:k]
    idx = idx[np.argsort(distances[idx])]
    return idx

def query_worker(args):
    """
    子进程工作函数，用于处理单条查询
    """
    (query_vals, query_attr_indices, filtered_base_attrs, filtered_base_vecs, global_indices, query_vec, top_k, worker_id) = args

    try:
        # 创建初始掩码
        mask = np.ones(filtered_base_attrs.shape[0], dtype=bool)
        
        # 对每个查询属性进行过滤
        for i, (val, attr_idx) in enumerate(zip(query_vals, query_attr_indices)):
            # 确保查询值是整数
            val = int(val)
            # 获取基础数据中对应属性列的值
            base_vals = filtered_base_attrs[:, attr_idx]
            # 更新掩码
            mask &= (base_vals == val)
            
            # 打印调试信息
            valid_count = np.sum(mask)
            # print(f"[DEBUG] Worker {worker_id} - 属性 {attr_idx} 过滤后剩余: {valid_count} 个点")

        valid_indices = np.where(mask)[0]
        # print(f"[INFO] Worker {worker_id} - 查询 {query_vals} 最终过滤后的有效点数: {len(valid_indices)}")

        if len(valid_indices) == 0:
            return worker_id, np.array([], dtype=np.int32)

        # 计算向量距离并获取top_k
        valid_vecs = filtered_base_vecs[valid_indices]
        dist = compute_distances(query_vec, valid_vecs)
        top_idx = find_top_k(dist, top_k)

        # 返回全局索引
        result = global_indices[valid_indices[top_idx]].astype(np.int32)
        return worker_id, result

    except Exception as e:
        print(f"[ERROR] Worker {worker_id} failed: {str(e)}")
        return worker_id, np.array([], dtype=np.int32)

def process_query_batch(args):
    """
    每个进程处理一批查询，更细粒度的任务分配
    """
    (base_attrs, base_vecs, queries, query_vecs, label_mapping, top_k, batch_indices) = args
    results = {}

    Nb = base_vecs.shape[0]
    
    for i in batch_indices:
        q_vals = queries[i]
        q_vec = query_vecs[i]
        q_attr_indices = label_mapping[:len(q_vals)]
        
        _, result = query_worker((q_vals, q_attr_indices, base_attrs, base_vecs, np.arange(Nb), q_vec, top_k, 0))
        
        if len(result) < top_k:
            padded = np.ones((top_k,), dtype=np.int32) * (-1)
            padded[:len(result)] = result
            results[i] = padded
        else:
            results[i] = result[:top_k]
    
    return results

def compute_ground_truth_with_knn(
    base_attr_file,
    base_fvecs_file,
    query_attr_file,
    query_fvecs_file,
    output_ivecs_file,
    top_k=100,
    label_mapping=None,
    batch_size=None,
    num_processes=None,
    use_mmap=False,
    chunk_size=20
):
    start_time = time.time()
    
    # 设置进程数
    if num_processes is None:
        num_processes = cpu_count()

    print(f"[INFO] 系统CPU核心数: {cpu_count()}")
    print(f"[INFO] 设置使用进程数: {num_processes}")
    
    # 1) 读取base数据
    print("[INFO] 正在读取基础数据...")
    base_attrs = load_base_attributes(base_attr_file)
    base_vecs = read_fvecs(base_fvecs_file, mmap=use_mmap)
    Nb = base_vecs.shape[0]
    print(f"[INFO] 基础数据读取完成: {Nb} 个向量, 维度 {base_vecs.shape[1]}")
    
    # 2) 读取query
    print("[INFO] 正在读取查询数据...")
    Nq, L, query_data = load_queries(query_attr_file)
    query_vecs = read_fvecs(query_fvecs_file, mmap=use_mmap)
    if query_vecs.shape[0] != Nq:
        raise ValueError(f"Query vectors count ({query_vecs.shape[0]}) does not match query lines count ({Nq})")
    print(f"[INFO] 查询数据读取完成: {Nq} 个查询, {L} 个标签")
    
    # 3) 验证标签映射
    if label_mapping is None:
        print("[WARNING] 未提供标签映射，默认使用所有标签")
        label_mapping = list(range(L))

    if max(label_mapping) >= base_attrs.shape[1]:
        raise ValueError(f"标签映射 {label_mapping} 超出基础数据标签列数 {base_attrs.shape[1]}")
    
    print(f"[INFO] 使用标签映射: {label_mapping}")
    
    # 4) 自动设置批处理大小
    if batch_size is None:
        queries_per_process = max(10, Nq // (num_processes * 2))
        batch_size = queries_per_process * num_processes
        batch_size = min(batch_size, Nq)
    
    print(f"[INFO] 设置批处理大小: {batch_size}")
    
    # 5) 重新设计任务分配，增加并行度
    results = {}  # 使用字典存储结果，键为查询索引
    
    with Pool(num_processes) as pool:
        # 监控CPU使用率
        def monitor_cpu_usage():
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            avg_usage = sum(cpu_percent) / len(cpu_percent)
            print(f"[INFO] 当前CPU平均使用率: {avg_usage:.2f}%, 每核使用率: {cpu_percent}")
        
        # 将查询划分为更小的批次，实现更细粒度的任务分配
        for batch_start in range(0, Nq, batch_size):
            batch_end = min(batch_start + batch_size, Nq)
            print(f"[INFO] 处理查询批次 {batch_start+1} 到 {batch_end} (共 {Nq} 个)")

            monitor_cpu_usage()

            batch_indices = list(range(batch_start, batch_end))
            chunks = [batch_indices[i:i+chunk_size] for i in range(0, len(batch_indices), chunk_size)]
            
            print(f"[INFO] 批次拆分为 {len(chunks)} 个子任务，每个任务处理 {chunk_size} 个查询")
            
            chunk_args = [
                (base_attrs, base_vecs, query_data, query_vecs, label_mapping, top_k, chunk)
                for chunk in chunks
            ]
            
            chunk_results_list = list(tqdm(pool.imap_unordered(process_query_batch, chunk_args), total=len(chunks), desc="处理子任务进度"))
            
            for chunk_results in chunk_results_list:
                results.update(chunk_results)

            monitor_cpu_usage()

    final_results = [results.get(i, np.ones((top_k,), dtype=np.int32) * (-1)) for i in range(Nq)]
    out_data = np.vstack(final_results)
    write_ivecs(output_ivecs_file, out_data)
    
    end_time = time.time()
    print(f"[INFO] 真值计算完成，结果保存在: {output_ivecs_file}")
    print(f"[INFO] 总耗时: {end_time - start_time:.2f} 秒")

    # 添加调试信息
    print(f"[DEBUG] 基础属性数据形状: {base_attrs.shape}")
    print(f"[DEBUG] 基础向量数据形状: {base_vecs.shape}")
    print(f"[DEBUG] 查询属性示例:\n{query_data[:3]}")
    print(f"[DEBUG] 查询向量形状: {query_vecs.shape}")
    print(f"[DEBUG] 标签映射: {label_mapping}")

def load_base_attributes(filename):
    """
    读取基础数据的属性文件
    返回numpy数组，每行包含该向量的所有属性值
    """
    attrs = []
    with open(filename, 'r') as f:
        for line in f:
            # 将每行的属性值转换为整数列表
            values = [int(x) for x in line.strip().split()]
            attrs.append(values)
    return np.array(attrs, dtype=np.int32)

def load_queries(filename):
    """
    读取查询文件
    返回：(查询数量，标签数量，查询数据列表)
    每行包含要查询的属性值
    """
    queries = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # 确保所有值都转换为整数
        values = [int(x) for x in line.strip().split()]
        queries.append(values)
        
    if not queries:
        raise ValueError("查询文件为空")
        
    Nq = len(queries)
    L = len(queries[0])
    
    return Nq, L, queries

def get_query_config():
    """
    返回所有数据集和查询集的配置信息
    """
    # 定义数据集配置
    datasets = {
        "sift": {
            "base_dir": "/data/zjxdata/data/sift",
            "label_dir": "/data/filter_data/label/sift_label",
            "gt_dir": "/data/filter_data/gt/sift",
            "base_file": "sift_base.fvecs",
            "query_file": "sift_query.fvecs",
        },
       
        "msong": {
            "base_dir": "/data/zjxdata/data/msong",
            "label_dir": "/data/filter_data/label/msong_label",
            "gt_dir": "/data/filter_data/gt/msong",
            "base_file": "msong_base.fvecs",
            "query_file": "msong_query.fvecs",
        },
        "audio": {
            "base_dir": "/data/zjxdata/data/audio",
            "label_dir": "/data/filter_data/label/audio_label",
            "gt_dir": "/data/filter_data/gt/audio",
            "base_file": "audio_base.fvecs",
            "query_file": "audio_query.fvecs",
        },
        "enron": {
            "base_dir": "/data/zjxdata/data/enron",
            "label_dir": "/data/filter_data/label/enron_label",
            "gt_dir": "/data/filter_data/gt/enron",
            "base_file": "enron_base.fvecs",
            "query_file": "enron_query.fvecs",
        },
        "gist": {
            "base_dir": "/data/zjxdata/data/gist",
            "label_dir": "/data/filter_data/label/gist_label",
            "gt_dir": "/data/filter_data/gt/gist",
            "base_file": "gist_base.fvecs",
            "query_file": "gist_query.fvecs",
        },
        "glove-100": {
            "base_dir": "/data/zjxdata/data/glove-100",
            "label_dir": "/data/filter_data/label/glove-100_label",
            "gt_dir": "/data/filter_data/gt/glove-100",
            "base_file": "glove-100_base.fvecs",
            "query_file": "glove-100_query.fvecs",
        },
       

    }
    
    # 定义查询集配置
    query_sets = {
        "1": {
            "name": "基本实验单属性",
            "attrs": [0],
            "suffix": "query_set_1"
        },
        "2-1": {
            "name": "多属性构建单标签搜索",
            "attrs": [0],
            "suffix": "query_set_2_1"
        },
        "2-1": {
            "name": "多属性构建单标签搜索",
            "attrs": [0],
            "suffix": "query_set_2_2"
        },
         "3-1": {
            "name": "1%选择性实验",
            "attrs": [5],
            "suffix": "query_set_3_1"
        },
        "3-2": {
            "name": "25%选择性实验",
            "attrs": [5],
            "suffix": "query_set_3_2"
        },
        "3-3": {
            "name": "50%选择性实验",
            "attrs": [5],
            "suffix": "query_set_3_3"
        },
        "3-4": {
            "name": "75%选择性实验",
            "attrs": [5],
            "suffix": "query_set_3_4"
        },
        "4": {
            "name": "1%选择性实验",
            "attrs": [7],
            "suffix": "query_set_4"
        },
        "5-1": {
            "name": "长尾分布标签实验",
            "attrs": [1],
            "suffix": "query_set_5_1"
        },
        "5-2": {
            "name": "5正态分布标签实验",
            "attrs": [2],
            "suffix": "query_set_5_2"
        },
        "5-3": {
            "name": "幂律分布标签实验",
            "attrs": [4],
            "suffix": "query_set_5_3"
        },
        "5-4": {
            "name": "均匀分布标签实验",
            "attrs": [0],
            "suffix": "query_set_5_4"
        },
        
        
        
        "6": {
            "name": "三标签搜索",
            "attrs": [0, 8, 9],
            "suffix": "query_set_6"
        },
        "7-1": {
            "name": "多标签1%选择性",
            "attrs": [0, 8, 5],
            "suffix": "query_set_7_1"
        },
        "7-2": {
            "name": "多标签25%选择性",
            "attrs": [0, 8, 5],
            "suffix": "query_set_7_2"
        },
        "7-3": {
            "name": "多标签50%选择性",
            "attrs": [0, 8, 5],
            "suffix": "query_set_7_3"
        },
        "7-4": {
            "name": "多标签75%选择性",
            "attrs": [0, 8, 5],
            "suffix": "query_set_7_4"
        },
    #    添加其他查询集配置...
    }
    
    return datasets, query_sets

def process_all_datasets():
    """
    处理所有数据集和查询集
    """
    datasets, query_sets = get_query_config()
    
    for dataset_name, dataset_config in datasets.items():
        print(f"\n[INFO] 处理数据集: {dataset_name}")
        
        base_fvecs = os.path.join(dataset_config["base_dir"], dataset_config["base_file"])
        query_fvecs = os.path.join(dataset_config["base_dir"], dataset_config["query_file"])
        base_attr = os.path.join(dataset_config["label_dir"], "labels.txt")
        
        for query_id, query_config in query_sets.items():
            print(f"\n[INFO] 处理查询集: {query_config['name']} (ID: {query_id})")
            
            # 构建查询文件路径
            query_attr = os.path.join(dataset_config["label_dir"], 
                                    f"query_{query_config['suffix']}.txt")
            
            # 构建输出文件路径
            output_dir = os.path.join(dataset_config["gt_dir"])
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"gt-{query_config['suffix']}.ivecs")
            
            try:
                # 检查文件是否存在
                if not all(os.path.exists(f) for f in [base_fvecs, query_fvecs, base_attr, query_attr]):
                    print(f"[WARNING] 跳过 {dataset_name}-{query_id}: 部分文件不存在")
                    continue
                
                print(f"[INFO] 开始计算真值: {output_file}")
                compute_ground_truth_with_knn(
                    base_attr_file=base_attr,
                    base_fvecs_file=base_fvecs,
                    query_attr_file=query_attr,
                    query_fvecs_file=query_fvecs,
                    output_ivecs_file=output_file,
                    top_k=100,
                    label_mapping=query_config["attrs"],
                    num_processes=96,
                    use_mmap=True,
                    chunk_size=20
                )
                print(f"[INFO] 完成查询集 {query_id}")
                
            except Exception as e:
                print(f"[ERROR] 处理 {dataset_name}-{query_id} 时出错: {str(e)}")
                continue

if __name__ == "__main__":
    process_all_datasets()
