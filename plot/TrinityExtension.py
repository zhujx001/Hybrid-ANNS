import os
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

# 定义数据路径
data_path = "/data/result"  # 请修改为实际路径

libertine_font = fm.FontProperties(
    fname='/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')

# 定义每个数据集的TOP3算法（针对多模态数据集）
TOP3_ALGORITHMS = {
    "text2image-mix": ["UNG", "NHQ", "CAPS"],
    "YT-Audio-Real85": ["UNG", "StitchedVamana", "NHQ"],
    "deep100M": ["UNG", "NHQ", "Faiss+HQI_Batch"]
}

colors = [
  # 主色系（高饱和度）
  '#9B59B6', # 宝石紫（Gem Purple）
  
  '#D35400', # 南瓜深橙（Pumpkin Orange）
  '#E74C3C', # 警报红（Alert Red）
  '#FF6B6B', # 珊瑚红（Coral Red）
  '#34495E', # 钢蓝灰（Steel Blue Gray）
  
  # 扩展色（高冲击力）
  
  '#A569BD', # 紫水晶（Amethyst Purple）
  '#2ECC71', # 翡翠绿（Jade Green）
  '#48D1CC', # 土耳其蓝（Turquoise）
  '#F39C12', # 亮橙（Vivid Orange）
  '#3498DB', # 荧光蓝（Fluorescent Blue）
  
  # 新增补充色（保持高对比）
  '#1ABC9C', # 加勒比绿（Caribbean Green）
  '#6EC1E0', # 电光冰蓝（Electric Ice Blue）
  
  '#E91E63', # 霓虹粉（Neon Pink）
  '#2C3E50', # 午夜蓝（Midnight Blue）
  '#27AE60', # 翡翠深绿（Deep Emerald）
]

plot_params = {
    'markersize': 6,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

plot_legend_params = {
    'markersize': 10,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '*', '+', 'x', '|', '1', '2', '3', '4']

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
    # all_data = {('audio', '1'): [df1, df2, ...], ('audio', '2_1'): [df1, df2, ...], ...}
    return all_data

def compute_pareto_frontier(df, x_col, y_col, threshold_x=0.025, threshold_y_ratio=2.5):
    """
    计算帕累托前沿，并进行稀疏化处理，确保点在x轴和y轴（对数空间）的间距不小于给定阈值
    
    参数:
    threshold_x: x轴的绝对阈值
    threshold_y_ratio: y轴对数空间的相对阈值，表示相邻点的y值比例变化至少为(1+threshold_y_ratio)
    """
    if df.empty:
        return pd.DataFrame()
    
    x = df[x_col].values
    y = df[y_col].values
    
    # 按QPS从高到低排序
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # 初始化帕累托前沿点集合
    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]
    
    # 构建帕累托前沿
    for i in range(1, len(x_sorted)):
        if x_sorted[i] > pareto_front_x[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])
    
    # 应用稀疏化处理
    if threshold_x > 0 or threshold_y_ratio > 0:
        sparse_x = [pareto_front_x[0]]
        sparse_y = [pareto_front_y[0]]
        
        for i in range(1, len(pareto_front_x)):
            x_diff = pareto_front_x[i] - sparse_x[-1]
            # 由于y是对数刻度，使用比例变化而非绝对差值
            y_ratio = sparse_y[-1] / pareto_front_y[i] if pareto_front_y[i] < sparse_y[-1] else 1
            
            # 保留点的条件：x差距超过阈值 或 y值的比例变化超过阈值
            if x_diff >= threshold_x or y_ratio >= (1 + threshold_y_ratio):
                sparse_x.append(pareto_front_x[i])
                sparse_y.append(pareto_front_y[i])
            # 特例：如果是最后一个点，使用更宽松的条件
            elif i == len(pareto_front_x) - 1 and (x_diff >= threshold_x * 0.2 or y_ratio >= (1 + threshold_y_ratio * 0.2)):
                sparse_x.append(pareto_front_x[i])
                sparse_y.append(pareto_front_y[i])
        
        # 使用稀疏化后的点集
        pareto_front_x = sparse_x
        pareto_front_y = sparse_y
    
    # 找回原始索引
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

def fun1_attribute_multimodel(all_data):
    query_set = '1'
    datasets = ['text2image-mix']
    alg_set = ['ACORN-1','ACORN-γ', 'CAPS', 'Faiss', 'Faiss+HQI_Batch', 'FilteredVamana', 'Milvus', 'NHQ', 'Puck', 'StitchedVamana', 'UNG', 'VBASE']

    dataset_display_names = {
        'text2image-mix': 'Text2Image-Mix',
    }

    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, query in enumerate(alg_set):
        algorithm_colors[query] = colors[i % len(colors)]
        algorithm_line_styles[query] = line_styles[i % len(line_styles)]
        algorithm_markers[query] = markers[i % len(markers)]

    fig = plt.figure(figsize=(5, 4.4))

    gs = GridSpec(1, 1, figure=fig, wspace=0.15, hspace=0.25, height_ratios=[1], top=0.80, bottom=0.1)

    for idx_dataset, dataset in enumerate(datasets):
        single_thread_y_data = []
        row = 0
        col = 0
        ax_single = fig.add_subplot(gs[row, col])
        for alg in alg_set:
            key = (dataset, query_set)
            if key not in all_data:
                continue            
            data_list = all_data[key]
            df = None
            for d in data_list:
                if d['Algorithm'].iloc[0] == alg:
                    df = d
                    break  # 找到匹配的算法后跳出循环
            
            if df is None:  # 如果没有找到匹配的算法，跳过
                continue
            color = algorithm_colors[alg]
            linestyle = algorithm_line_styles[alg]
            marker = algorithm_markers[alg]    
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            if not pareto_df.empty:
                # 过滤 Recall 小于0.4的数据
                filtered_df = pareto_df[(pareto_df['Recall'] >= 0.5)]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    single_thread_y_data.append(y_data)
                    ax_single.plot(x_data, y_data, marker=marker, linestyle=linestyle, 
                                  label=alg, color=color, **plot_params)  # 使用alg作为标签

        # 设置单线程图的y轴范围和刻度
        y_min_single, y_max_single, y_ticks_single, y_tick_labels_single = get_y_range_and_ticks(single_thread_y_data)
        ax_single.set_yscale('log')
        ax_single.set_ylim(y_min_single, y_max_single)
         # 只显示主刻度
        ax_single.yaxis.set_major_locator(plt.FixedLocator(y_ticks_single))
        ax_single.yaxis.set_minor_locator(plt.NullLocator())
        ax_single.set_yticklabels(y_tick_labels_single)
        ax_single.tick_params(axis='both', labelsize=16)

        # 设置x轴范围为0.7到1.01，但仅显示刻度至1.0
        ax_single.set_xlim(0.5, 1.01)
        ax_single.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_single.grid(True, linestyle=':', alpha=0.6)

        # 添加Recall@10标签
        # 设置x轴标签
        # 使用格式化后的数据集名称
        formatted_dataset = dataset_display_names[dataset]
        formatted_label = f"Recall@10 ({formatted_dataset})"
        ax_single.set_xlabel(formatted_label, 
                        fontproperties=libertine_font,
                        fontsize=20,
                        fontweight='bold')
        # 仅第一显示y轴标签
        if col == 0:
            ax_single.set_ylabel("QPS", fontsize=16)
        else:
            ax_single.set_ylabel("")
        
        # 添加TOP3标注
        if dataset in TOP3_ALGORITHMS:
            top3_algs = TOP3_ALGORITHMS[dataset]
            # 构建要显示的文本
            top_text = "Top 3: " + ", ".join([f"{alg}" for alg in top3_algs])
            # 使用红色添加文本，位置在左上角
            ax_single.text(0.03, 0.92, top_text, 
                    transform=ax_single.transAxes, 
                    fontsize=12.6, 
                    color='red')
                    
    # 调整布局，为底部的数据集标签留出空间
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig      

def fun2_attribute_85(all_data):
    query_set = '1'
    datasets = ['YT-Audio-Real85']
    alg_set = ['ACORN-1','ACORN-γ', 'CAPS', 'Faiss', 'Faiss+HQI_Batch', 'FilteredVamana', 'Milvus', 'NHQ', 'Puck', 'StitchedVamana', 'UNG', 'VBASE']

    dataset_display_names = {
        'text2image-mix': 'Text2Image-Mix',
        'YT-Audio-Real85': 'YT-Audio-Real'
    }

    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, query in enumerate(alg_set):
        algorithm_colors[query] = colors[i % len(colors)]
        algorithm_line_styles[query] = line_styles[i % len(line_styles)]
        algorithm_markers[query] = markers[i % len(markers)]

    fig = plt.figure(figsize=(5, 4.4))

    gs = GridSpec(1, 1, figure=fig, wspace=0.15, hspace=0.25, height_ratios=[1], top=0.80, bottom=0.1)

    for idx_dataset, dataset in enumerate(datasets):
        single_thread_y_data = []
        row = 0
        col = 0
        ax_single = fig.add_subplot(gs[row, col])
        for alg in alg_set:
            key = (dataset, query_set)
            if key not in all_data:
                continue            
            data_list = all_data[key]
            df = None
            for d in data_list:
                if d['Algorithm'].iloc[0] == alg:
                    df = d
                    break  # 找到匹配的算法后跳出循环
            
            if df is None:  # 如果没有找到匹配的算法，跳过
                continue
            color = algorithm_colors[alg]
            linestyle = algorithm_line_styles[alg]
            marker = algorithm_markers[alg]    
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            if not pareto_df.empty:
                # 过滤 Recall 小于0.4的数据
                filtered_df = pareto_df[(pareto_df['Recall'] >= 0.5)]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    single_thread_y_data.append(y_data)
                    ax_single.plot(x_data, y_data, marker=marker, linestyle=linestyle, 
                                  label=alg, color=color, **plot_params)  # 使用alg作为标签

        # 设置单线程图的y轴范围和刻度
        y_min_single, y_max_single, y_ticks_single, y_tick_labels_single = get_y_range_and_ticks(single_thread_y_data)
        ax_single.set_yscale('log')
        ax_single.set_ylim(y_min_single, y_max_single)
         # 只显示主刻度
        ax_single.yaxis.set_major_locator(plt.FixedLocator(y_ticks_single))
        ax_single.yaxis.set_minor_locator(plt.NullLocator())
        ax_single.set_yticklabels(y_tick_labels_single)
        ax_single.tick_params(axis='both', labelsize=16)

        # 设置x轴范围为0.7到1.01，但仅显示刻度至1.0
        ax_single.set_xlim(0.5, 1.01)
        ax_single.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_single.grid(True, linestyle=':', alpha=0.6)

        # 添加Recall@10标签
        # 设置x轴标签
        # 使用格式化后的数据集名称
        formatted_dataset = dataset_display_names[dataset]
        formatted_label = f"Recall@10 ({formatted_dataset})"
        ax_single.set_xlabel(formatted_label, 
                        fontproperties=libertine_font,
                        fontsize=20,
                        fontweight='bold')
        # 仅第一显示y轴标签
        ax_single.set_ylabel("")
        # 添加TOP3标注
        if dataset in TOP3_ALGORITHMS:
            top3_algs = TOP3_ALGORITHMS[dataset]
            # 构建要显示的文本
            top_text = "Top 3: " + ", ".join([f"{alg}" for alg in top3_algs])
            # 使用红色添加文本，位置在左上角
            ax_single.text(0.03, 0.92, top_text, 
                    transform=ax_single.transAxes, 
                    fontsize=12.6, 
                    color='red')
                    
    # 调整布局，为底部的数据集标签留出空间
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig

def fun3_attribute_100M(all_data):
    query_set = '1'
    datasets = ['deep100M']
    alg_set = ['ACORN-1','ACORN-γ', 'CAPS', 'Faiss', 'Faiss+HQI_Batch', 'FilteredVamana', 'Milvus', 'NHQ', 'Puck', 'StitchedVamana', 'UNG', 'VBASE']

    dataset_display_names = {
        'text2image-mix': 'Text2Image-Mix',
        'YT-Audio-Real85': 'YT-Audio-Real',
        'deep100M': 'Deep100M'
    }

    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, query in enumerate(alg_set):
        algorithm_colors[query] = colors[i % len(colors)]
        algorithm_line_styles[query] = line_styles[i % len(line_styles)]
        algorithm_markers[query] = markers[i % len(markers)]

    fig = plt.figure(figsize=(5, 4.4))

    gs = GridSpec(1, 1, figure=fig, wspace=0.15, hspace=0.25, height_ratios=[1], top=0.80, bottom=0.1)

    for idx_dataset, dataset in enumerate(datasets):
        single_thread_y_data = []
        row = 0
        col = 0
        ax_single = fig.add_subplot(gs[row, col])
        for alg in alg_set:
            key = (dataset, query_set)
            if key not in all_data:
                continue            
            data_list = all_data[key]
            df = None
            for d in data_list:
                if d['Algorithm'].iloc[0] == alg:
                    df = d
                    break  # 找到匹配的算法后跳出循环
            
            if df is None:  # 如果没有找到匹配的算法，跳过
                continue
            color = algorithm_colors[alg]
            linestyle = algorithm_line_styles[alg]
            marker = algorithm_markers[alg]    
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            if not pareto_df.empty:
                # 过滤 Recall 小于0.4的数据
                filtered_df = pareto_df[(pareto_df['Recall'] >= 0.5)]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    single_thread_y_data.append(y_data)
                    ax_single.plot(x_data, y_data, marker=marker, linestyle=linestyle, 
                                  label=alg, color=color, **plot_params)  # 使用alg作为标签

        # 设置单线程图的y轴范围和刻度
        y_min_single, y_max_single, y_ticks_single, y_tick_labels_single = get_y_range_and_ticks(single_thread_y_data)
        ax_single.set_yscale('log')
        ax_single.set_ylim(y_min_single, y_max_single)
         # 只显示主刻度
        ax_single.yaxis.set_major_locator(plt.FixedLocator(y_ticks_single))
        ax_single.yaxis.set_minor_locator(plt.NullLocator())
        ax_single.set_yticklabels(y_tick_labels_single)
        ax_single.tick_params(axis='both', labelsize=16)

        # 设置x轴范围为0.7到1.01，但仅显示刻度至1.0
        ax_single.set_xlim(0.5, 1.01)
        ax_single.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_single.grid(True, linestyle=':', alpha=0.6)

        # 添加Recall@10标签
        # 设置x轴标签
        # 使用格式化后的数据集名称
        formatted_dataset = dataset_display_names[dataset]
        formatted_label = f"Recall@10 ({formatted_dataset})"
        ax_single.set_xlabel(formatted_label, 
                        fontproperties=libertine_font,
                        fontsize=20,
                        fontweight='bold')
        # 仅第一显示y轴标签
        ax_single.set_ylabel("")
        
        # 添加TOP3标注
        if dataset in TOP3_ALGORITHMS:
            top3_algs = TOP3_ALGORITHMS[dataset]
            # 构建要显示的文本
            top_text = "Top 3: " + ", ".join([f"{alg}" for alg in top3_algs])
            # 使用红色添加文本，位置在左上角
            ax_single.text(0.03, 0.92, top_text, 
                    transform=ax_single.transAxes, 
                    fontsize=12.6, 
                    color='red')
                    
    # 调整布局，为底部的数据集标签留出空间
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig                    

def plot_attribute_legend_only():
    """创建一个单独的图例图表"""
    alg_set = ['ACORN-1','ACORN-γ', 'CAPS', 'Faiss', 'Faiss+HQI_Batch', 'FilteredVamana', 'Milvus', 'NHQ', 'Puck', 'StitchedVamana', 'UNG', 'VBASE']
    
    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, alg in enumerate(alg_set):
        algorithm_colors[alg] = colors[i % len(colors)]
        algorithm_line_styles[alg] = line_styles[i % len(line_styles)]
        algorithm_markers[alg] = markers[i % len(markers)]  
    
    # 创建一个更紧凑的图形，稍微增加高度以容纳两行图例
    fig = plt.figure(figsize=(10, 1.8))
    
    # 创建算法图例
    legend_elements = []
    for alg in alg_set:
        legend_elements.append(plt.Line2D([0], [0], 
                                         color=algorithm_colors[alg], 
                                         marker=algorithm_markers[alg], 
                                         linestyle=algorithm_line_styles[alg],
                                         label=alg, 
                                         **plot_legend_params))
    
    if legend_elements:
        # 调整图例位置和间距，设置为两行，每行6个
        leg = fig.legend(handles=legend_elements, loc='center',
                         bbox_to_anchor=(0.5, 0.5), ncol=6,
                         fontsize=20, frameon=False,
                         columnspacing=1.0,  # 减少列间距
                         handletextpad=0.5,  # 减少图例标记和文本之间的距离
                         borderaxespad=0.1)  # 减少图例和轴边界的距离
        leg.get_title().set_fontweight('bold')
    
    # 隐藏坐标轴并调整边距
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    
    # 调整整个图的边距，进一步减少留白
    plt.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.22)
    
    return fig

print("加载数据文件...")
all_data = load_all_data()
if not all_data:
    print("未找到有效的数据文件，请检查路径和文件格式。")
else:
    print("创建attribute_multimodel对比图...")
    fig1 = fun1_attribute_multimodel(all_data)
    save_path_svg = os.path.join("/data/plots/exp","attribute_multimodel.svg")
    save_path_pdf = os.path.join("/data/plots/exp","attribute_multimodel.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

    print("创建attribute_85对比图...")
    fig1 = fun2_attribute_85(all_data)
    save_path_svg = os.path.join("/data/plots/exp","attribute_85.svg")
    save_path_pdf = os.path.join("/data/plots/exp","attribute_85.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

    print("创建attribute_100M对比图...")
    fig1 = fun3_attribute_100M(all_data)
    save_path_svg = os.path.join("/data/plots/exp","attribute_100M.svg")
    save_path_pdf = os.path.join("/data/plots/exp","attribute_100M.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

    print("创建图例图...")
    fig_legend = plot_attribute_legend_only()
    save_path_svg = os.path.join("/data/plots/exp","attribute_legend.svg")
    save_path_pdf = os.path.join("/data/plots/exp","attribute_legend.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')