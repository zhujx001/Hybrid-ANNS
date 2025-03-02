import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from matplotlib.font_manager import FontProperties

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 使用通用字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 设置图表样式
plt.style.use('ggplot')

# 定义数据路径
data_path = "/data/result"

# 定义颜色映射
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 获取所有结果文件
def get_all_result_files():
    # 新路径格式为 /data/result/{算法名}/result/{数据集名}_{查询集编号}_results.csv
    result_files = []
    # 获取所有算法目录
    algorithm_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for alg in algorithm_dirs:
        # 构建算法的result目录路径
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
            # 获取该算法下的所有结果文件
            files = glob.glob(os.path.join(alg_result_path, "*_results.csv"))
            for file in files:
                result_files.append((file, alg))
    
    return result_files

# 提取数据集名称和查询集编号
def extract_file_info(filename, algorithm):
    base = os.path.basename(filename)
    match = re.match(r'(.+?)_(\d+(?:_\d+)?)_results\.csv', base)
    if match:
        dataset = match.group(1)
        queryset = match.group(2)
        return dataset, queryset, algorithm
    return None, None, None

# 加载所有数据
def load_all_data():
    all_data = {}
    files_with_algs = get_all_result_files()
    
    for file, algorithm in files_with_algs:
        dataset, queryset, algorithm = extract_file_info(file, algorithm)
        if dataset and queryset and algorithm:
            try:
                df = pd.read_csv(file)
                
                # 检查必要的列是否存在（至少需要Recall和QPS列）
                required_columns = ['Recall', 'QPS']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                # 添加算法名列
                df['Algorithm'] = algorithm
                
                key = (dataset, queryset)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return all_data

# 计算帕累托前沿
def compute_pareto_frontier(df, x_col, y_col, maximize_x=True, maximize_y=True):
    """
    计算帕累托前沿
    
    参数:
    df (DataFrame): 包含数据的DataFrame
    x_col (str): x轴的列名
    y_col (str): y轴的列名
    maximize_x (bool): 是否最大化x
    maximize_y (bool): 是否最大化y
    
    返回:
    DataFrame: 帕累托前沿点的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # 获取x和y的numpy数组
    x = df[x_col].values
    y = df[y_col].values
    
    # 对于QPS vs Recall，我们希望y (QPS)越高越好，所以要排序时考虑-y
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]
    
    for i in range(1, len(x_sorted)):
        if x_sorted[i] > pareto_front_x[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])
    
    # 创建帕累托前沿的DataFrame
    indices = []
    for px, py in zip(pareto_front_x, pareto_front_y):
        # 找到原始DataFrame中与当前点匹配的行
        matches = df[(df[x_col] == px) & (df[y_col] == py)].index
        if not matches.empty:
            indices.append(matches[0])
    
    # 返回帕累托前沿点的DataFrame
    if indices:
        return df.loc[indices].sort_values(by=x_col, ascending=True)
    else:
        return pd.DataFrame()

# 绘制QPS vs Recall图（使用帕累托前沿）
def plot_qps_vs_recall(data_list, title, save_path, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    
    legend_handles = []
    legend_labels = []
    
    for i, df in enumerate(data_list):
        # 获取算法名称
        if 'Algorithm' in df.columns:
            algorithm = df['Algorithm'].iloc[0]
        else:
            algorithm = f"Algorithm {i+1}"
        
        # 确保Recall和QPS列存在
        if 'Recall' in df.columns and 'QPS' in df.columns:
            # 计算帕累托前沿
            # 对于QPS vs Recall，我们希望Recall越高越好，QPS越高越好
            # 但当Recall增加时，QPS通常会降低，所以我们计算"反向"的帕累托前沿
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            
            if not pareto_df.empty:
                # 绘制帕累托前沿的折线图
                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', 
                        color=colors[i % len(colors)])
                
                # 单独添加图例
                legend_handles.append(line)
                legend_labels.append(algorithm)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('QPS', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 使用手动添加的图例
    if legend_handles:
        plt.legend(legend_handles, legend_labels, fontsize=10)
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 创建输出目录
def create_output_dirs():
    dirs = [
        "plots/1_single_label",
        "plots/2_multi_label_effect",
        "plots/3_label_distribution",
        "plots/4_multi_label_performance",
        "plots/5_selectivity",
        "plots/6_dataset_effect",
        "plots/7_memory_usage"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs

# 1. 简单单标签搜索实验比较
def plot_single_label_search(all_data, output_dir):
    query_set = "1"
    
    # 对每个数据集，将不同算法的结果绘制在同一图中
    for dataset in set(k[0] for k in all_data.keys()):
        key = (dataset, query_set)
        if key in all_data:
            title = f"{dataset} - Comparison of different algorithms for single label search - query set {query_set}"
            save_path = os.path.join(output_dir, f"{dataset}_queryset_{query_set}_algorithms_comparison.png")
            plot_qps_vs_recall(all_data[key], title, save_path)

# 2. 多标签索引构建对单标签搜索的影响
def plot_multi_label_effect(all_data, output_dir):
    query_sets = ["1", "2_1", "2_2"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 1. 将所有查询集和所有算法绘制在一张大图上
        plt.figure(figsize=(15, 10))
        
        legend_handles = []
        legend_labels = []
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分查询集
                    linestyle = ['-', '--', '-.'][query_set_idx % 3]
                    # 使用不同的颜色区分算法
                    color = colors[alg_idx % len(colors)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle=linestyle, 
                                color=color)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} - query set {query_set}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{dataset} - The impact of multi-label index construction on single-label search", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 使用手动添加的图例
        if legend_handles:
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='best')
        
        save_path = os.path.join(output_dir, f"{dataset}_all_querysets_all_algorithms.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 为每个算法绘制一张图，比较其在三个查询集上的表现
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        for alg in algorithms:
            plt.figure(figsize=(10, 6))
            
            legend_handles = []
            legend_labels = []
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', 
                                        color=colors[i % len(colors)])
                                legend_handles.append(line)
                                legend_labels.append(f"query set {query_set}")
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('QPS', fontsize=12)
            plt.title(f"{dataset} - {alg} - Comparison of three query sets (1/2_1/2_2)", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 使用手动添加的图例
            if legend_handles:
                plt.legend(legend_handles, legend_labels, fontsize=10)
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_querysets_comparison.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

# 3. 标签分布对算法的影响
def plot_label_distribution_effect(all_data, output_dir):
    query_sets = ["5_1", "5_2", "5_3", "5_4"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 同一算法在不同标签分布下的表现
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        # 1. 所有标签分布和所有算法组合在一张大图上
        plt.figure(figsize=(15, 10))
        
        legend_handles = []
        legend_labels = []
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分标签分布
                    linestyle = ['-', '--', '-.', ':'][query_set_idx % 4]
                    # 使用不同的颜色区分算法
                    color = colors[alg_idx % len(colors)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle=linestyle, 
                                color=color)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} - label distribution {query_set}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{dataset} - The impact of label distribution on algorithms", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 使用手动添加的图例
        if legend_handles:
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='best')
        
        save_path = os.path.join(output_dir, f"{dataset}_all_distributions_all_algorithms.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 为每个算法创建对比图，显示它在不同标签分布下的表现
        for alg in algorithms:
            plt.figure(figsize=(10, 6))
            
            legend_handles = []
            legend_labels = []
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', 
                                        color=colors[i % len(colors)])
                                legend_handles.append(line)
                                legend_labels.append(f"label distribution {query_set}")
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('QPS', fontsize=12)
            plt.title(f"{dataset} - {alg} - Comparison of different label distributions (5_1 to 5_4)", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 使用手动添加的图例
            if legend_handles:
                plt.legend(legend_handles, legend_labels, fontsize=10)
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_label_distributions_comparison.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 同一标签分布下不同算法的表现对比
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                title = f"{dataset} - Different algorithms in label distribution {query_set} "
                save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_distribution_{query_set}.png")
                plot_qps_vs_recall(all_data[key], title, save_path)

# 4. 多标签算法表现
def plot_multi_label_performance(all_data, output_dir):
    query_sets = ["6", "7_2"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 1. 所有查询集和所有算法组合在一张大图上
        plt.figure(figsize=(15, 10))
        
        legend_handles = []
        legend_labels = []
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分查询集
                    linestyle = ['-', '--'][query_set_idx % 2]
                    # 使用不同的颜色区分算法
                    color = colors[alg_idx % len(colors)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle=linestyle, 
                                color=color)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} - query set {query_set}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{dataset} - Multi-label algorithm performance", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 使用手动添加的图例
        if legend_handles:
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='best')
        
        save_path = os.path.join(output_dir, f"{dataset}_all_querysets_all_algorithms_multilabel.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 不同算法在查询集6上的表现对比
        key = (dataset, "6")
        if key in all_data:
            title = f"{dataset} - Comparison of different algorithms for multi-label search - query set 6"
            save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_queryset_6.png")
            plot_qps_vs_recall(all_data[key], title, save_path)
        
        # 3. 同一算法在查询集6和7_2上的表现对比
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        for alg in algorithms:
            plt.figure(figsize=(10, 6))
            
            legend_handles = []
            legend_labels = []
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', 
                                        color=colors[i % len(colors)])
                                legend_handles.append(line)
                                legend_labels.append(f"query set {query_set}")
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('QPS', fontsize=12)
            plt.title(f"{dataset} - {alg} - Comparison of query set 6 and 7_2", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 使用手动添加的图例
            if legend_handles:
                plt.legend(legend_handles, legend_labels, fontsize=10)
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_querysets_6_7_2_comparison.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 不同算法在查询集7_2下的表现对比
        key = (dataset, "7_2")
        if key in all_data:
            title = f"{dataset} - Comparison of different algorithms for multi-label search - query set 7_2"
            save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_queryset_7_2.png")
            plot_qps_vs_recall(all_data[key], title, save_path)

# 5. 选择性实验比较
def plot_selectivity_experiments(all_data, output_dir):
    single_label_selectivity = ["3_1", "3_2", "3_3", "3_4"]  # 1%, 25%, 50%, 75%
    multi_label_selectivity = ["7_1", "7_2", "7_3", "7_4"]   # 1%, 25%, 50%, 75%
    
    selectivity_mapping = {
        "3_1": "1%", "3_2": "25%", "3_3": "50%", "3_4": "75%",
        "7_1": "1%", "7_2": "25%", "7_3": "50%", "7_4": "75%"
    }
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 1. 单标签：所有选择性和所有算法组合在一张大图上
        plt.figure(figsize=(15, 10))
        
        legend_handles = []
        legend_labels = []
        
        for query_set_idx, query_set in enumerate(single_label_selectivity):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分选择性
                    linestyle = ['-', '--', '-.', ':'][query_set_idx % 4]
                    # 使用不同的颜色区分算法
                    color = colors[alg_idx % len(colors)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle=linestyle, 
                                color=color)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} - selectivity {selectivity_mapping[query_set]}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{dataset} - Single label search selectivity experiment", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 使用手动添加的图例
        if legend_handles:
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='best')
        
        save_path = os.path.join(output_dir, f"{dataset}_all_single_selectivity_all_algorithms.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 多标签：所有选择性和所有算法组合在一张大图上
        plt.figure(figsize=(15, 10))
        
        legend_handles = []
        legend_labels = []
        
        for query_set_idx, query_set in enumerate(multi_label_selectivity):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分选择性
                    linestyle = ['-', '--', '-.', ':'][query_set_idx % 4]
                    # 使用不同的颜色区分算法
                    color = colors[alg_idx % len(colors)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle=linestyle, 
                                color=color)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} - selectivity {selectivity_mapping[query_set]}")
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{dataset} - Multi-label search selectivity experiment", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 使用手动添加的图例
        if legend_handles:
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='best')
        
        save_path = os.path.join(output_dir, f"{dataset}_all_multi_selectivity_all_algorithms.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 不同算法在相同选择性下的表现对比
        # 单标签选择性
        for query_set in single_label_selectivity:
            key = (dataset, query_set)
            if key in all_data:
                title = f"{dataset} - Comparison of different algorithms for single label search - selectivity {selectivity_mapping[query_set]}"
                save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_single_label_selectivity_{query_set}.png")
                plot_qps_vs_recall(all_data[key], title, save_path)
        
        # 多标签选择性
        for query_set in multi_label_selectivity:
            key = (dataset, query_set)
            if key in all_data:
                title = f"{dataset} - Comparison of different algorithms for multi-label search - selectivity {selectivity_mapping[query_set]}"
                save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_multi_label_selectivity_{query_set}.png")
                plot_qps_vs_recall(all_data[key], title, save_path)
        
        # 4. 同一算法在不同选择性下的表现
        algorithms = set()
        for query_set in single_label_selectivity + multi_label_selectivity:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        # 为每个算法创建单标签选择性对比图
        for alg in algorithms:
            # 单标签选择性
            plt.figure(figsize=(10, 6))
            
            legend_handles = []
            legend_labels = []
            
            for i, query_set in enumerate(single_label_selectivity):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', 
                                        color=colors[i % len(colors)])
                                legend_handles.append(line)
                                legend_labels.append(f"selectivity {selectivity_mapping[query_set]}")
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('QPS', fontsize=12)
            plt.title(f"{dataset} - {alg} - Comparison of single label selectivity (1%/25%/50%/75%)", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 使用手动添加的图例
            if legend_handles:
                plt.legend(legend_handles, legend_labels, fontsize=10)
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_single_label_selectivity_comparison.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 多标签选择性
            plt.figure(figsize=(10, 6))
            
            legend_handles = []
            legend_labels = []
            
            for i, query_set in enumerate(multi_label_selectivity):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', 
                                        color=colors[i % len(colors)])
                                legend_handles.append(line)
                                legend_labels.append(f"selectivity {selectivity_mapping[query_set]}")
            
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('QPS', fontsize=12)
            plt.title(f"{dataset} - {alg} - Comparison of multi-label selectivity (1%/25%/50%/75%)", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 使用手动添加的图例
            if legend_handles:
                plt.legend(legend_handles, legend_labels, fontsize=10)
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_multi_label_selectivity_comparison.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
def main():
    # 创建输出目录
    output_dirs = create_output_dirs()
    
    # 加载所有数据
    print("加载数据文件...")
    all_data = load_all_data()
    
    if not all_data:
        print("未找到有效的数据文件，请检查路径和文件格式。")
        return
    
    print(f"成功加载数据，共有 {sum(len(data_list) for data_list in all_data.values())} 个数据集。")
    
    # 运行各个实验的图表绘制
    print("1. 绘制简单单标签搜索实验比较图...")
    plot_single_label_search(all_data, output_dirs[0])
    
    print("2. 绘制多标签索引构建对单标签搜索的影响图...")
    plot_multi_label_effect(all_data, output_dirs[1])
    
    print("3. 绘制标签分布对算法的影响图...")
    plot_label_distribution_effect(all_data, output_dirs[2])
    
    print("4. 绘制多标签算法表现图...")
    plot_multi_label_performance(all_data, output_dirs[3])
    
    print("5. 绘制选择性实验比较图...")
    plot_selectivity_experiments(all_data, output_dirs[4])
    
    print("6. 绘制数据集对算法的影响图...")
    plot_dataset_effect(all_data, output_dirs[5])
    
   
    
    print("所有图表绘制完成！")
if __name__ == "__main__":
    main()
