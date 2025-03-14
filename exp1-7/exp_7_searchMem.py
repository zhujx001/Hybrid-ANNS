import os
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 定义数据路径
data_path = "/data/searchMem"  # 请修改为实际路径

# 基础颜色列表
colors = [
    '#ff8d00', '#006400', '#6696ec', '#f08181', '#727272', 
    '#80ffd5', '#8c008c', '#aeff2b', '#8c0000', '#0000ce',
    '#ffface', '#008c8c', '#ff4300', '#662f9a', '#ffffff',
    '#66CCCC', '#CC99FF', '#FF66B2', '#99CC00', '#47d147'
]

# 全局字典存储所有算法的样式信息
ALGORITHM_STYLES = {}

# 获取所有结果文件
def get_all_result_files():
    result_files = []
    algorithm_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for alg in algorithm_dirs:
        alg_result_path = os.path.join(data_path, alg)
        if os.path.exists(alg_result_path):
            files = glob.glob(os.path.join(alg_result_path, "*.csv"))
            for file in files:
                result_files.append((file, alg))
    
    return result_files

# 提取数据集名称
def extract_file_info(filename, algorithm):
    base = os.path.basename(filename)
    match = re.match(r'(.+?).csv', base)
    if match:
        dataset = match.group(1)
        return dataset, algorithm
    return None, None

# 加载所有数据
def load_all_data():
    all_data = {}
    files_with_algs = get_all_result_files()
    
    for file, algorithm in files_with_algs:
        dataset, algorithm = extract_file_info(file, algorithm)
        if dataset and algorithm:
            try:
                df = pd.read_csv(file)
                if 'Memory(MB)' not in df.columns or 'Dataset' not in df.columns:
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                key = (dataset, algorithm)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return all_data

def plot_memory_comparison(all_data, figsize=(30, 5)):
    """
    对各算法在不同数据集上的内存使用情况构建柱状图比较
    
    参数:
    all_data: 加载的所有数据
    figsize: 图像大小
    """
    # 收集所有唯一的数据集
    all_datasets = set()
    for key in all_data.keys():
        for df_list in all_data.values():
            for df in df_list:
                all_datasets.update(df['Dataset'].values)
    
    datasets = sorted(list(all_datasets))
    
    # 收集所有唯一的算法
    algorithms = sorted(list(set([key[1] for key in all_data.keys()])))
    
    # 为每个算法分配一种颜色
    for i, alg in enumerate(algorithms):
        if alg not in ALGORITHM_STYLES:
            ALGORITHM_STYLES[alg] = {'color': colors[i % len(colors)]}
    
    # 创建一个子图
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    # 设置柱状图宽度
    bar_width = 0.8 / len(algorithms)
    
    # 收集所有内存值用于设置Y轴范围
    all_values = []
    
    # 为每个算法创建数据集到内存值的映射
    alg_data = {}
    for alg in algorithms:
        alg_data[alg] = {}
        for ds in datasets:
            # 初始化为NaN
            alg_data[alg][ds] = np.nan
    
    # 填充实际数据
    for (dataset, alg), df_list in all_data.items():
        for df in df_list:
            for _, row in df.iterrows():
                ds_name = row['Dataset']
                memory = row['Memory(MB)']
                if ds_name in datasets:
                    alg_data[alg][ds_name] = memory
                    if not np.isnan(memory):
                        all_values.append(memory)
    
    # 绘制每个算法的柱子
    for alg_idx, alg in enumerate(algorithms):
        x = np.arange(len(datasets))
        values = [alg_data[alg][ds] for ds in datasets]
        
        # 设置每个柱的位置
        offset = bar_width * (alg_idx - len(algorithms) / 2 + 0.5)
        
        # 绘制柱状图
        bars = ax.bar(x + offset, values, bar_width, 
               label=alg, 
               color=ALGORITHM_STYLES[alg]['color'],
               edgecolor='black', linewidth=2)
        
        # 添加数值标签（只显示值较大的）
        # for i, bar in enumerate(bars):
        #     if not np.isnan(values[i]):
        #         height = bar.get_height()

        #         value_text = f"{values[i]:.2f}"
                
        #         ax.text(bar.get_x() + bar.get_width()/2., height+20,
        #                 value_text, ha='center', va='bottom',
        #                 rotation=90, fontsize=8)
    
    # 设置对数刻度
    if all_values:
        ax.set_yscale('log')
        
        # 设置Y轴显示范围，留出一点空间显示标签
        min_val = min([v for v in all_values if v > 0]) * 0.8
        max_val = max(all_values) * 1.2
        ax.set_ylim(min_val, max_val)

    # 设置轴刻度标签字体大小
    ax.tick_params(axis='both', labelsize=20)
    
    # 设置X轴刻度和标签
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    
    # 添加标题和轴标签
    plt.ylabel('Memory Usage (MB)', fontsize=22)
    
    # 添加图例
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(algorithms),
        fontsize=17,
        frameon=False)
    
    # 调整布局和显示网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    output_dir = "/home/ykw/study/Hybrid-ANNS/faiss/plots/searchMem"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/memory_comparison.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/memory_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    print("Loading data files...")
    all_data = load_all_data()
    
    if not all_data:
        print("No valid data files found. Please check path and file format.")
        return
    
    print(f"Successfully loaded data with {sum(len(data_list) for data_list in all_data.values())} datasets.")
    plot_memory_comparison(all_data=all_data)
    print("Plot generated successfully.")

if __name__ == "__main__":
    main()