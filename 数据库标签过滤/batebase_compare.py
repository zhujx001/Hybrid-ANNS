import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import colorsys

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 使用通用字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 设置图表样式
plt.style.use('ggplot')

# 定义数据路径
data_path = "/data/result"

# 修改颜色定义：使用相同的基础颜色，无论是标准版本还是16位版本
ALGORITHM_COLORS = {
    'vbase': '#1f77b4',    # 蓝色
    'vbase-16': '#1f77b4', # 同样是蓝色
    'pase': '#ff7f0e',     # 橙色
    'pase-16': '#ff7f0e',  # 同样是橙色
    'milvus': '#2ca02c',   # 绿色
    'milvus-16': '#2ca02c' # 同样是绿色
}

# 线型定义：用于区分标准版本和16位版本
ALGORITHM_LINESTYLES = {
    'vbase': '-',      # 实线
    'pase': '-',       # 实线
    'milvus': '-',     # 实线
    'vbase-16': '--',  # 虚线
    'pase-16': '--',   # 虚线
    'milvus-16': '--'  # 虚线
}

# 获取实验1的结果文件
def get_result_files():
    # 我们只关注vbase, pase, milvus (以及它们的16位版本)
    target_algorithms = ['vbase', 'pase', 'milvus', 'vbase-16', 'pase-16', 'milvus-16']
    result_files = []
    
    for alg in target_algorithms:
        # 构建算法的result目录路径
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
            # 获取SIFT数据集、实验1的结果文件
            files = glob.glob(os.path.join(alg_result_path, "sift_1_results.csv"))
            for file in files:
                result_files.append((file, alg))
    
    return result_files

# 加载数据
def load_data():
    all_data = {}
    files_with_algs = get_result_files()
    
    for file, algorithm in files_with_algs:
        try:
            df = pd.read_csv(file)
            
            # 检查必要的列是否存在
            required_columns = ['Recall', 'QPS']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: File {file} missing required columns. Skipping.")
                continue
            
            # 添加算法名列
            df['Algorithm'] = algorithm
            
            # 所有算法都放在同一组中，不再区分standard和16bit
            if 'all_algorithms' not in all_data:
                all_data['all_algorithms'] = []
            all_data['all_algorithms'].append(df)
            
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

# 绘制对比图
def plot_comparison_charts(all_data):
    os.makedirs("comparison_plots", exist_ok=True)
    
    # 绘制所有算法的对比图
    if 'all_algorithms' in all_data and all_data['all_algorithms']:
        plt.figure(figsize=(10, 6))
        legend_handles = []
        legend_labels = []
        
        for df in all_data['all_algorithms']:
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            
            if not pareto_df.empty:
                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], 
                                marker='o', linestyle=ALGORITHM_LINESTYLES.get(algorithm, '-'), 
                                color=ALGORITHM_COLORS.get(algorithm, '#000000'))
                legend_handles.append(line)
                legend_labels.append(algorithm)
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('QPS', fontsize=14)
        plt.title('SIFT Dataset - Experiment 1 - Algorithms Comparison', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        if legend_handles:
            plt.legend(legend_handles, legend_labels, fontsize=12)
        
        save_path = os.path.join("comparison_plots", "sift_exp1_algorithms.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Algorithms comparison chart saved to {save_path}")

def main():
    print("Loading data files for SIFT dataset experiment 1...")
    all_data = load_data()
    
    if not all_data:
        print("No valid data files found. Please check the path and file format.")
        return
    
    all_algorithms_count = len(all_data.get('all_algorithms', []))
    print(f"Successfully loaded data: {all_algorithms_count} algorithms.")
    
    print("Generating comparison charts...")
    plot_comparison_charts(all_data)
    
    print("All charts generated successfully!")

if __name__ == "__main__":
    main()