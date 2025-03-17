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
# 基础颜色列表
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#FF4444', '#47D147', '#AA44FF', '#FF9933'
]

plot_params = {
    'markersize': 4,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

plot_legend_params = {
    'markersize': 6,                # 标记大小
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

# 绘制所有数据集的QPS vs Recall对比图
def plot_all_datasets_comparison(all_data):
    query_set = ['5_1', '5_2', '5_3', '5_4'] # 这里假设我们只关心查询集1
    datasets = 'sift'
    
    # 创建全局颜色字典，确保同一算法无论单线程还是16线程都使用相同颜色
   
    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, query in enumerate(query_set):
        algorithm_colors[query] = colors[i % len(colors)]
        algorithm_line_styles[query] = line_styles[i % len(line_styles)]
        algorithm_markers[query] = markers[i % len(markers)]  

    # 创建一个2行6列的大图，使用更宽的比例
    fig = plt.figure(figsize=(30, 8))
    
    # 调整上下间距，给图例留出空间
    gs = GridSpec(2, 6, figure=fig, wspace=0.15, hspace=0.25, height_ratios=[1, 1], top=0.85, bottom=0.1)

    plotted_algs = set()
    
    # 为构造统一图例，使用集合记录在当前图中出现的基础算法名称
    single_thread_algs = ['UNG', 'NHQ', 'StitchedVamana', 'CAPS', 'FilteredVamana', 'faiss+HQI_Batch', 'ACORN-gama', 'ACORN-1', 'faiss','milvus', 'vbase', 'pase']

    # print(all_data[('sift', '5_1')][0]['Algorithm'].iloc[0])
    # 遍历所有算法
    for col, alg in enumerate(single_thread_algs):

        # 绘制单线程图 (第一行)
        row = col // 6
        col = col % 6
        ax_single = fig.add_subplot(gs[row, col])

        # 收集单线程图的Y轴数据，为每个算法单独收集
        single_thread_y_data = []

        for query in query_set:
            key = (datasets, query)
            if key not in all_data:
                continue
            # (sift, 5_1) 对应的df列表
            data_list = all_data[key]
            df = None
            for d in data_list:
                if d['Algorithm'].iloc[0] == alg:
                    df = d
            color = algorithm_colors[query]
            linestyle = algorithm_line_styles[query]
            marker = algorithm_markers[query]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            if not pareto_df.empty:
                # 过滤 Recall 小于0.7以及等于1.01的数据
                filtered_df = pareto_df[(pareto_df['Recall'] >= 0.7)]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    single_thread_y_data.append(y_data)
                    ax_single.plot(x_data, y_data, marker=marker,linestyle=linestyle, label=datasets, color=color, **plot_params)
                    plotted_algs.add(datasets)

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
        ax_single.set_xlim(0.7, 1.01)
        ax_single.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.7, 0.8, 0.9, 1.0]])

        ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_single.grid(True, linestyle=':', alpha=0.6)

        # 设置x轴标签
        label_index = chr(97 + row*6 + col)  # 97是ASCII码中'a'的值
        formatted_label = f"({label_index}) Recall@10 ({alg})"
        ax_single.set_xlabel(formatted_label, 
                         #fontfamily='Linux Libertine O',  # 字体家族
                         fontproperties=libertine_font,
                         fontsize=20,         # 字体大小
                         fontweight='bold')   # 字体粗细
        
        
        # 仅第一显示y轴标签
        if col == 0:
            ax_single.set_ylabel("QPS", fontsize=16)
        else:
            ax_single.set_ylabel("")

    distribution_mapping = {
        '5_1': 'Long-tailed distribution',
        '5_2': 'Normal distribution', 
        '5_3': 'Power-law distribution',
        '5_4': 'Uniform distribution'
    }

    legend_elements = []
    for query in query_set:
        legend_elements.append(plt.Line2D([0], [0], color=algorithm_colors[query], marker=algorithm_markers[query], linestyle=algorithm_line_styles[query],
                                            label=distribution_mapping[query], **plot_legend_params))
    
    if legend_elements:
        leg = fig.legend(handles=legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, 0.92), ncol=min(15, len(legend_elements)),
                         fontsize=17, frameon=False)
        leg.get_title().set_fontweight('bold')
    
    # plt.suptitle("QPS vs Recall Performance Comparison Across Datasets", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.0, 1, 1.0])

    return fig

# 在Jupyter环境中使用的代码

# 创建一个新的函数用于绘制多线程算法图表
def plot_multi_thread_comparison(all_data):
    query_set = ['5_1', '5_2', '5_3', '5_4']
    datasets = 'sift'
    
    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, query in enumerate(query_set):
        algorithm_colors[query] = colors[i % len(colors)]
        algorithm_line_styles[query] = line_styles[i % len(line_styles)]
        algorithm_markers[query] = markers[i % len(markers)]  
    
    # 创建一个2行6列的大图，使用更宽的比例
    fig = plt.figure(figsize=(30, 8))
    
    # 调整上下间距，给图例留出空间
    gs = GridSpec(2, 6, figure=fig, wspace=0.15, hspace=0.25, height_ratios=[1, 1], top=0.85, bottom=0.1)

    plotted_algs = set()
    
    # 多线程算法列表
    multi_thread_algs = ['puck-16', 'UNG-16', 'parlayivf-16', 'CAPS', 'DiskANN-s-16', 'DiskANN-f-16', 'ACORN-gama-16', 'ACORN-1-16', 'faiss-16', 'milvus-16', 'vbase-16', 'pase-16']
    
    # 遍历所有算法
    for col, alg in enumerate(multi_thread_algs):
        # 绘制多线程图
        row = col // 6
        col = col % 6
        ax_multi = fig.add_subplot(gs[row, col])

        # 收集多线程图的Y轴数据
        multi_thread_y_data = []

        for query in query_set:
            key = (datasets, query)
            if key not in all_data:
                continue
            
            data_list = all_data[key]
            df = None
            for d in data_list:
                if d['Algorithm'].iloc[0] == alg:
                    df = d
            
            color = algorithm_colors[query]
            linestyle = algorithm_line_styles[query]
            marker = algorithm_markers[query]
            if df is not None:
                pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
                if not pareto_df.empty:
                    # 过滤 Recall 小于0.7以及等于1.01的数据
                    filtered_df = pareto_df[(pareto_df['Recall'] >= 0.7)]
                    if not filtered_df.empty:
                        x_data = filtered_df['Recall'].tolist()
                        y_data = filtered_df['QPS'].tolist()
                        multi_thread_y_data.append(y_data)
                        ax_multi.plot(x_data, y_data, marker=marker,linestyle=linestyle, label=datasets, color=color, **plot_params)
                        plotted_algs.add(datasets)

        # 设置多线程图的y轴范围和刻度
        y_min_multi, y_max_multi, y_ticks_multi, y_tick_labels_multi = get_y_range_and_ticks(multi_thread_y_data)
        ax_multi.set_yscale('log')
        ax_multi.set_ylim(y_min_multi, y_max_multi)
        ax_multi.yaxis.set_major_locator(plt.FixedLocator(y_ticks_multi))
        ax_multi.yaxis.set_minor_locator(plt.NullLocator())
        ax_multi.set_yticklabels(y_tick_labels_multi)
        ax_multi.tick_params(axis='both', labelsize=16)

        # 设置x轴范围为0.7到1.01，但仅显示刻度至1.0
        ax_multi.set_xlim(0.7, 1.01)
        ax_multi.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax_multi.set_xticklabels([f"{tick:.1f}" for tick in [0.7, 0.8, 0.9, 1.0]])
        ax_multi.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_multi.grid(True, linestyle=':', alpha=0.6)
        # 设置x轴标签
        label_index = chr(97 + row*6 + col)  # 97是ASCII码中'a'的值
        display_alg = alg.replace('-16', '') if alg.endswith('-16') else alg
        formatted_label = f"({label_index}) Recall@10 ({display_alg})"
        ax_multi.set_xlabel(formatted_label, 
                         #fontfamily='Linux Libertine O',  # 字体家族
                         fontproperties=libertine_font,
                         fontsize=20,         # 字体大小
                         fontweight='bold')   # 字体粗细
        
        
        # 仅第一显示y轴标签
        if col == 0:
            ax_multi.set_ylabel("QPS", fontsize=16)
        else:
            ax_multi.set_ylabel("")


        distribution_mapping = {
            '5_1': 'Long-tailed distribution',
            '5_2': 'Normal distribution', 
            '5_3': 'Power-law distribution',
            '5_4': 'Uniform distribution'
        }   
    # 创建统一图例
    legend_elements = []
    for query in query_set:
        legend_elements.append(plt.Line2D([0], [0], color=algorithm_colors[query], marker=algorithm_markers[query], linestyle=algorithm_line_styles[query],
                                            label=distribution_mapping[query], **plot_legend_params))
    
    if legend_elements:
        leg = fig.legend(handles=legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, 0.92), ncol=min(15, len(legend_elements)),
                         fontsize=17, frameon=False)
        leg.get_title().set_fontweight('bold')
    
    # plt.suptitle("QPS vs Recall Performance Comparison Across Datasets (Multi-Thread)", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.0, 1, 1.0])

    return fig

print("加载数据文件...")
all_data = load_all_data()

if not all_data:
    print("未找到有效的数据文件，请检查路径和文件格式。")
else:
    total_files = sum(len(data_list) for data_list in all_data.values())
    print(f"成功加载数据，共有 {total_files} 个数据文件。")
    
    print("创建单线程对比图...")
    fig1 = plot_all_datasets_comparison(all_data)
    save_path_svg = os.path.join("/data/plots/exp","exp_3_1.svg")
    save_path_pdf = os.path.join("/data/plots/exp","exp_3_1.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig1)
    
    print("创建多线程对比图...")
    fig2 = plot_multi_thread_comparison(all_data)
    save_path_svg = os.path.join("/data/plots/exp","exp_3_2.svg")
    save_path_pdf = os.path.join("/data/plots/exp","exp_3_2.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.show()
    
    print("图表已创建并保存！")
