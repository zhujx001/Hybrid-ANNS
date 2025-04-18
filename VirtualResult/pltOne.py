import os
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# 定义数据路径
data_path = "/data/result"  # 请修改为实际路径

# 基础颜色列表
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#FF4444', '#47D147', '#AA44FF', '#FF9933'
]

# 将算法分组（单线程和16线程）
def group_algorithms(algorithms):
    alg_16Thread = []
    alg_other = []
    for alg in algorithms:
        if alg.endswith('-16'):
            alg_16Thread.append(alg)
        else:
            alg_other.append(alg)
    return alg_16Thread, alg_other

# 获取所有结果文件
def get_all_result_files():
    result_files = []
    algorithm_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for alg in algorithm_dirs:
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
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
                
                if 'Recall' not in df.columns or 'QPS' not in df.columns:
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                df['Algorithm'] = algorithm
                
                key = (dataset, queryset)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return all_data

# 计算帕累托前沿
def compute_pareto_frontier(df, x_col, y_col):
    if df.empty:
        return pd.DataFrame()
    
    x = df[x_col].values
    y = df[y_col].values
    
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]
    
    for i in range(1, len(x_sorted)):
        if x_sorted[i] > pareto_front_x[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])
    
    indices = []
    for px, py in zip(pareto_front_x, pareto_front_y):
        matches = df[(df[x_col] == px) & (df[y_col] == py)].index
        if not matches.empty:
            indices.append(matches[0])
    
    if indices:
        return df.loc[indices].sort_values(by=x_col, ascending=True)
    else:
        return pd.DataFrame()

# 获取y轴范围和刻度
def get_y_range_and_ticks(y_data_list):
    if not y_data_list:
        return 10**2, 10**5, [10**2, 10**3, 10**4, 10**5], ["10²", "10³", "10⁴", "10⁵"]
    
    # 获取所有数据的最小值和最大值
    all_y_values = [y for sublist in y_data_list for y in sublist]
    
    if not all_y_values:
        return 10**2, 10**5, [10**2, 10**3, 10**4, 10**5], ["10²", "10³", "10⁴", "10⁵"]
    
    min_y = min(all_y_values)
    max_y = max(all_y_values)
    
    # 确定最小和最大的幂次
    min_power = max(0, np.floor(np.log10(min_y)))
    max_power = np.ceil(np.log10(max_y))
    
    # 确保至少有2个主刻度
    if max_power - min_power < 1:
        min_power = max(0, max_power - 2)
    
    # 设置y轴范围，稍微扩展一点
    y_min = 10 ** min_power / 1.5
    y_max = 10 ** max_power * 1.5
    
    # 生成刻度值和标签
    ticks = [10 ** i for i in range(int(min_power), int(max_power) + 1)]
    tick_labels = [f"10{superscript(i)}" for i in range(int(min_power), int(max_power) + 1)]
    
    return y_min, y_max, ticks, tick_labels

# 辅助函数：将数字转换为上标
def superscript(n):
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'
    }
    return ''.join(superscript_map[digit] for digit in str(n))

# 绘制所有数据集的QPS vs Recall对比图
def plot_all_datasets_comparison(all_data):
    query_set = "1"  # 这里假设我们只关心查询集1
    datasets = sorted(set(k[0] for k in all_data.keys() if k[1] == query_set))
    
    # 如果数据集少于6个，补充到6个
    if len(datasets) < 6:
        datasets = datasets + [None] * (6 - len(datasets))
    # 如果数据集多于6个，只取前6个
    elif len(datasets) > 6:
        datasets = datasets[:6]
    
    # 创建全局颜色字典，确保同一算法无论单线程还是16线程都使用相同颜色
    global_algorithms = set()
    for key in all_data:
        if key[1] == query_set:
            for df in all_data[key]:
                # 提取基础名称
                base_name = df['Algorithm'].iloc[0].replace('-16', '')
                global_algorithms.add(base_name)
    global_color_dict = {alg: colors[i % len(colors)] for i, alg in enumerate(sorted(global_algorithms))}
    
    # 创建一个2行6列的大图，使用更宽的比例
    fig = plt.figure(figsize=(30, 10))
    
    # 调整上下间距，给图例留出空间
    gs = GridSpec(2, 6, figure=fig, wspace=0.15, hspace=0.4, height_ratios=[1, 1], top=0.85, bottom=0.1)
    
    # 为构造统一图例，使用集合记录在当前图中出现的基础算法名称
    plotted_algs = set()
    
    # 遍历所有数据集，绘制单线程和16线程的图
    for col, dataset in enumerate(datasets):
        if dataset is None:
            continue
        
        key = (dataset, query_set)
        if key not in all_data:
            continue
        
        data_list = all_data[key]
        
        # 绘制单线程图 (第一行)
        ax_single = fig.add_subplot(gs[0, col])
        
        # 收集单线程图的Y轴数据，为每个算法单独收集
        single_thread_y_data = []
        
        # 绘制单线程算法数据
        for df in [d for d in data_list if not d['Algorithm'].iloc[0].endswith('-16')]:
            algorithm = df['Algorithm'].iloc[0]
            # 提取基础名称
            base_name = algorithm
            color = global_color_dict.get(base_name, '#000000')
            
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            if not pareto_df.empty:
                # 过滤 Recall 小于0.7以及等于1.01的数据
                filtered_df = pareto_df[(pareto_df['Recall'] >= 0.7) & (pareto_df['Recall'] != 1.01)]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    single_thread_y_data.append(y_data)
                    ax_single.plot(x_data, y_data, 'o-', label=base_name, color=color, markersize=5)
                    plotted_algs.add(base_name)
        
        # 设置单线程图的y轴范围和刻度
        y_min_single, y_max_single, y_ticks_single, y_tick_labels_single = get_y_range_and_ticks(single_thread_y_data)
        
        ax_single.set_yscale('log')
        ax_single.set_ylim(y_min_single, y_max_single)
        ax_single.yaxis.set_major_locator(plt.FixedLocator(y_ticks_single))
        ax_single.yaxis.set_minor_locator(plt.NullLocator())
        ax_single.set_yticklabels(y_tick_labels_single)
        
        # 设置x轴范围为0.7到1.01，但仅显示刻度至1.0
        ax_single.set_xlim(0.7, 1.01)
        ax_single.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.7, 0.8, 0.9, 1.0]])
        
        ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_single.grid(True, linestyle=':', alpha=0.6)
        
        ax_single.set_title(f"{dataset} - Single Thread", fontsize=12, pad=5)
        ax_single.set_xlabel("")
        ax_single.set_ylabel("QPS", fontsize=10)
        
        # 绘制16线程图 (第二行)
        ax_16 = fig.add_subplot(gs[1, col])
        
        thread_16_y_data = []
        for df in [d for d in data_list if d['Algorithm'].iloc[0].endswith('-16')]:
            algorithm = df['Algorithm'].iloc[0]
            base_name = algorithm.replace('-16', '')
            color = global_color_dict.get(base_name, '#000000')
            
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            if not pareto_df.empty:
                filtered_df = pareto_df[(pareto_df['Recall'] >= 0.7) & (pareto_df['Recall'] != 1.01)]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    thread_16_y_data.append(y_data)
                    ax_16.plot(x_data, y_data, 'o-', label=base_name, color=color, markersize=5)
                    plotted_algs.add(base_name)
        
        y_min_16, y_max_16, y_ticks_16, y_tick_labels_16 = get_y_range_and_ticks(thread_16_y_data)
        
        ax_16.set_yscale('log')
        ax_16.set_ylim(y_min_16, y_max_16)
        ax_16.yaxis.set_major_locator(plt.FixedLocator(y_ticks_16))
        ax_16.yaxis.set_minor_locator(plt.NullLocator())
        ax_16.set_yticklabels(y_tick_labels_16)
        
        ax_16.set_xlim(0.7, 1.01)
        ax_16.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax_16.set_xticklabels([f"{tick:.1f}" for tick in [0.7, 0.8, 0.9, 1.0]])
        
        ax_16.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_16.grid(True, linestyle=':', alpha=0.6)
        
        ax_16.set_title(f"{dataset} - 16 Threads", fontsize=12, pad=5)
        ax_16.set_xlabel("Recall", fontsize=10)
        ax_16.set_ylabel("QPS", fontsize=10)
    
    # 创建统一图例（按照全局颜色字典中出现过的算法）
    legend_elements = []
    for alg in sorted(plotted_algs):
        legend_elements.append(plt.Line2D([0], [0], color=global_color_dict[alg], marker='o', linestyle='-',
                                            markersize=6, label=alg))
    
    if legend_elements:
        leg = fig.legend(handles=legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, 0.98), ncol=min(15, len(legend_elements)),
                         fontsize=10, frameon=True, title="Algorithms (Single & 16 Threads)")
        leg.get_title().set_fontweight('bold')
    
    plt.suptitle("QPS vs Recall Performance Comparison Across Datasets", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.0, 1, 1.0])
    
    return fig

# 在Jupyter环境中使用的代码

print("加载数据文件...")
all_data = load_all_data()

if not all_data:
    print("未找到有效的数据文件，请检查路径和文件格式。")
else:
    total_files = sum(len(data_list) for data_list in all_data.values())
    print(f"成功加载数据，共有 {total_files} 个数据文件。")
    
    print("创建对比图...")
    fig = plot_all_datasets_comparison(all_data)
    
    plt.savefig("算法性能对比图.svg", format="svg", bbox_inches='tight')
    plt.show()
    
    print("图表已创建并保存！")
