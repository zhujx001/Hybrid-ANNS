import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 使用通用字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 设置图表样式为白色背景
plt.style.use('default')  # 使用默认风格，白色背景

# 定义数据路径
data_path = "/data/result"

# 为每个算法定义唯一的颜色（标准版本和16位版本使用相同颜色）
ALGORITHM_COLORS = {
    'ACORN-1': '#1f77b4',       # 蓝色
    'ACORN-1-16': '#1f77b4',    # 蓝色
    'ACORN-gama': '#ff7f0e',    # 橙色
    'ACORN-gama-16': '#ff7f0e', # 橙色
    'DiskANN-f': '#2ca02c',     # 绿色
    'DiskANN-f-16': '#2ca02c',  # 绿色
    'DiskANN-s': '#d62728',     # 红色
    'DiskANN-s-16': '#d62728',  # 红色
    'NHQ': '#9467bd',           # 紫色
    'parlayivf-16': '#8c564b',     # 棕色
    # 'puck': '#e377c2',          # 粉色
    'puck-16': '#e377c2',       # 粉色 (添加puck-16)
    'UNG': '#7f7f7f',           # 灰色
    'UNG-16': '#7f7f7f',        # 灰色
}

# 为不同的数据集定义不同的线型
DATASET_LINESTYLES = {
    'sift_1': '-',      # 实线
    'sift_2_1': '--',   # 虚线
    'sift_2_2': '-.'    # 点划线
}

# 标记样式
DATASET_MARKERS = {
    'sift_1': 'o',      # 圆形
    'sift_2_1': 's',    # 方形
    'sift_2_2': '^'     # 三角形
}

# 获取实验结果文件
def get_result_files():
    # 目标算法列表，添加了puck-16
    target_algorithms = [
        'ACORN-1', 'ACORN-1-16', 
        'ACORN-gama', 'ACORN-gama-16', 
        'DiskANN-f', 'DiskANN-f-16',
        'DiskANN-s', 'DiskANN-s-16',
        'NHQ', 'parlayivf-16', 'puck-16',
        'UNG', 'UNG-16'
    ]
    
    result_files = []
    
    for alg in target_algorithms:
        # 构建算法的result目录路径
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
            # 获取所有三种数据集的结果文件
            for pattern in ["sift_1_results.csv", "sift_2_1_results.csv", "sift_2_2_results.csv"]:
                files = glob.glob(os.path.join(alg_result_path, pattern))
                for file in files:
                    # 提取数据集名称（sift_1, sift_2_1, or sift_2_2）
                    dataset = os.path.basename(file).split("_results.csv")[0]
                    result_files.append((file, alg, dataset))
    
    return result_files

# 加载数据
def load_data():
    # 分为单线程和16线程两组数据
    all_data = {
        'single_thread': [],
        'multi_thread': []
    }
    
    files_with_algs = get_result_files()
    
    for file, algorithm, dataset in files_with_algs:
        try:
            df = pd.read_csv(file)
            
            # 检查必要的列是否存在
            required_columns = ['Recall', 'QPS']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: File {file} missing required columns. Skipping.")
                continue
            
            # 添加算法名列和数据集列
            df['Algorithm'] = algorithm
            df['Dataset'] = dataset
            
            # 根据算法名称分组
            if algorithm.endswith('-16'):
                all_data['multi_thread'].append(df)
            else:
                all_data['single_thread'].append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return all_data

# 计算帕累托前沿
def compute_pareto_frontier(df, x_col, y_col):
    """计算帕累托前沿"""
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

# ...existing code...

def plot_comparison_charts(all_data):
    os.makedirs("comparison_plots", exist_ok=True)
    
    # 绘制单线程算法的对比图
    if all_data['single_thread']:
        plt.figure(figsize=(14, 10))
        legend_handles = []
        legend_labels = []
        
        # 按算法和数据集分组
        grouped_data = {}
        for df in all_data['single_thread']:
            algorithm = df['Algorithm'].iloc[0]
            dataset = df['Dataset'].iloc[0]
            
            if algorithm not in grouped_data:
                grouped_data[algorithm] = {}
            
            grouped_data[algorithm][dataset] = df
        
        # 按算法绘制线条
        for algorithm in sorted(grouped_data.keys()):
            for dataset in ['sift_1', 'sift_2_1', 'sift_2_2']:
                if dataset in grouped_data[algorithm]:
                    df = grouped_data[algorithm][dataset]
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], 
                                        marker=DATASET_MARKERS.get(dataset, 'o'),
                                        linestyle=DATASET_LINESTYLES.get(dataset, '-'), 
                                        color=ALGORITHM_COLORS.get(algorithm, '#000000'),
                                        linewidth=2)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} ({dataset})")
        
        # 在x=0.95处添加灰色虚线
        plt.axvline(x=0.95, color='#aaaaaa', linestyle='--', linewidth=1, alpha=0.7)
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('QPS', fontsize=14)
        plt.title('SIFT Dataset - Experiment 2 - Single Thread Algorithms', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数尺度
        plt.ylim(bottom=6)  # 从6开始
        plt.xlim(right=1.05)  # X轴最大值设为1.05
        
        if legend_handles:
            # 创建具有多列的图例，使其更紧凑
            plt.legend(legend_handles, legend_labels, fontsize=9, loc='best', ncol=3)
        
        # 保存为矢量图格式（SVG）
        save_path_svg = os.path.join("comparison_plots", "sift_exp2_single_thread.svg")
        plt.savefig(save_path_svg, format='svg', dpi=300, bbox_inches='tight')
        
        # 也可以同时保存PNG格式作为备用
        save_path_png = os.path.join("comparison_plots", "sift_exp2_single_thread.png") 
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"Single thread algorithms comparison chart saved as vector graphic to {save_path_svg}")
    
    # 绘制多线程算法的对比图
    if all_data['multi_thread']:
        plt.figure(figsize=(14, 10))
        legend_handles = []
        legend_labels = []
        
        # 按算法和数据集分组
        grouped_data = {}
        for df in all_data['multi_thread']:
            algorithm = df['Algorithm'].iloc[0]
            dataset = df['Dataset'].iloc[0]
            
            if algorithm not in grouped_data:
                grouped_data[algorithm] = {}
            
            grouped_data[algorithm][dataset] = df
        
        # 按算法绘制线条
        for algorithm in sorted(grouped_data.keys()):
            for dataset in ['sift_1', 'sift_2_1', 'sift_2_2']:
                if dataset in grouped_data[algorithm]:
                    df = grouped_data[algorithm][dataset]
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
                    
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], 
                                        marker=DATASET_MARKERS.get(dataset, 'o'),
                                        linestyle=DATASET_LINESTYLES.get(dataset, '-'), 
                                        color=ALGORITHM_COLORS.get(algorithm, '#000000'),
                                        linewidth=2)
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} ({dataset})")
        
        # 在x=0.95处添加灰色虚线
        plt.axvline(x=0.95, color='#aaaaaa', linestyle='--', linewidth=1, alpha=0.7)
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('QPS', fontsize=14)
        plt.title('SIFT Dataset - Experiment 2 - 16-Thread Algorithms', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数尺度
        plt.ylim(bottom=6)
        plt.xlim(right=1.05)  # X轴最大值设为1.05
        
        if legend_handles:
            # 创建具有多列的图例，使其更紧凑
            plt.legend(legend_handles, legend_labels, fontsize=9, loc='best', ncol=3)
        
        # 保存为矢量图格式（SVG）
        save_path_svg = os.path.join("comparison_plots", "sift_exp2_16_thread.svg")
        plt.savefig(save_path_svg, format='svg', dpi=300, bbox_inches='tight')
        
        # 也可以同时保存PNG格式作为备用
        save_path_png = os.path.join("comparison_plots", "sift_exp2_16_thread.png")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"16-thread algorithms comparison chart saved as vector graphic to {save_path_svg}")

# ...existing code...

def main():
    print("Loading data files for SIFT dataset experiments...")
    all_data = load_data()
    
    if not all_data['single_thread'] and not all_data['multi_thread']:
        print("No valid data files found. Please check the path and file format.")
        return
    
    single_thread_count = len(all_data['single_thread'])
    multi_thread_count = len(all_data['multi_thread'])
    print(f"Successfully loaded data: {single_thread_count} single-thread algorithm datasets and {multi_thread_count} 16-thread algorithm datasets.")
    
    print("Generating comparison charts...")
    plot_comparison_charts(all_data)
    
    print("All charts generated successfully!")

if __name__ == "__main__":
    main()