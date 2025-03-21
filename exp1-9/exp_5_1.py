'''实验五绘图代码'''
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

# 创建一个字体对象，指定字体文件的路径
libertine_font = fm.FontProperties(
    fname='/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')

# 基础颜色列表
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#FF4444', '#CC79A7', '#AA44FF', '#FF9933',
    '#66CCCC', '#CC99FF', '#FF66B2', '#99CC00', '#47d147'
]


# 扩展线型和标记列表
line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '*', '+', 'x', '|', '1', '2', '3', '4']

plot_params = {
    'markersize': 4,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'alpha': 0.9,                 # 透明度
    'linewidth': 1.2           # 线条粗细
}

plot_params_Legend = {
    'markersize': 6,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

selectivity_mapping = {
    '3_1': 'selectivity 1%',
    '3_2': 'selectivity 25%',
    '3_3': 'selectivity 50%',
    '3_4': 'selectivity 75%',
    '7_1': 'selectivity 1%',
    '7_2': 'selectivity 25%',
    '7_3': 'selectivity 50%',
    '7_4': 'selectivity 75%',
}

# 全局字典存储所有算法的样式信息
ALGORITHM_STYLES = {}

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

# 计算稀疏化的帕累托前沿
# threshold_y_ratio是相对阈值，表示相邻点的y值比例变化至少为(1+threshold_y_ratio)
# def compute_pareto_frontier(df, x_col, y_col, threshold_x=0.02, threshold_y_ratio=0.5):     # 多标签参数
def compute_pareto_frontier(df, x_col, y_col, threshold_x=0.025, threshold_y_ratio=5):   # 单标签参数
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
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '-': '⁻'
    }
    return ''.join(superscript_map[digit] for digit in str(n))

# 初始化算法样式（颜色、线型、标记）
def initialize_algorithm_styles(all_data, query_set="1"):
    """为所有算法分配一致的视觉样式（颜色、线型、标记）"""
    global ALGORITHM_STYLES
    
    # 收集所有基本算法名称（移除"-16"后缀）
    base_algorithms = set()
    for key, data_list in all_data.items():
        if key[1] == query_set:
            for df in data_list:
                if 'Algorithm' in df.columns:
                    alg_name = df['Algorithm'].iloc[0]
                    # 对于16线程算法，提取基本算法名（移除-16后缀）
                    base_name = alg_name.replace('-16', '')
                    base_algorithms.add(base_name)
    
    # 为每个算法分配一致的视觉样式
    for i, alg in enumerate(sorted(base_algorithms, key=str.lower)):
        ALGORITHM_STYLES[alg] = {
            'color': colors[i % len(colors)],
            'linestyle': line_styles[i % len(line_styles)],
            'marker': markers[i % len(markers)]
        }
    
    return ALGORITHM_STYLES

# 通用绘图函数
def plot_dataset_comparison(all_data, thread_mode="single", _query_set="3", figsize=(20, 5.8)):
    """
    绘制单线程或16线程的QPS vs Recall性能对比图
    
    参数:
    all_data: 所有数据的字典
    thread_mode: "single"表示单线程图，"multi"表示16线程图
    query_set: 查询集编号
    figsize: 图表尺寸
    
    返回:
    fig: matplotlib图表对象
    """
    # datasets = sorted(set(k[0] for k in all_data.keys() if k[1].startswith(query_set + "_") or k[1] == query_set))
    dataset = "sift"
    query_sets = sorted(set(k[1] for k in all_data.keys() if k[1].startswith(_query_set + "_") or k[1] == _query_set))
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 调整间距
    gs = GridSpec(1, len(query_sets), figure=fig, wspace=0.15, top=0.72, bottom=0.2)
    
    # 收集当前图表中的算法
    current_algorithms = {}
    
    # 遍历所有数据集
    for col, query_set in enumerate(query_sets):
        if query_set is None or (dataset, query_set) not in all_data:
            continue
        
        # 获取数据列表
        data_list = all_data[(dataset, query_set)]
        
        # 分组算法
        all_algorithms = set(df['Algorithm'].iloc[0] for df in data_list if 'Algorithm' in df.columns)
        alg_16Thread, alg_single = group_algorithms(all_algorithms)
        
        # 根据线程模式选择要绘制的算法
        selected_algorithms = alg_single if thread_mode == "single" else alg_16Thread
        
        # 创建子图
        ax = fig.add_subplot(gs[0, col])
        
        # 收集Y轴数据
        y_data_list = []
        
        # 绘制算法数据
        for df in [d for d in data_list if d['Algorithm'].iloc[0] in selected_algorithms]:
            algorithm = df['Algorithm'].iloc[0]
            
            # 处理算法名和样式
            if thread_mode == "multi":
                base_name = algorithm.replace('-16', '')
                label = base_name
            else:
                base_name = algorithm
                label = algorithm
            
            # 获取算法样式
            style = ALGORITHM_STYLES.get(base_name, {
                'color': colors[0],
                'linestyle': line_styles[0],
                'marker': markers[0]
            })
            
            # 记录此算法已在图中
            current_algorithms[label] = style['color']
            
            # 计算并绘制帕累托前沿
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
            
            if not pareto_df.empty:
                filtered_df = pareto_df[pareto_df['Recall'] >= 0.7]
                if not filtered_df.empty:
                    x_data = filtered_df['Recall'].tolist()
                    y_data = filtered_df['QPS'].tolist()
                    y_data_list.append(y_data)
                    
                    # 绘制线条
                    ax.plot(x_data, y_data,
                            marker=style['marker'], 
                            linestyle=style['linestyle'], 
                            label=label, 
                            color=style['color'],
                            **plot_params)
                    
        # 设置轴刻度标签字体大小
        ax.tick_params(axis='both', labelsize=16)
        
        # 设置y轴范围和刻度
        y_min, y_max, y_ticks, y_tick_labels = get_y_range_and_ticks(y_data_list)
        
        # 设置y轴为对数刻度并调整范围
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        
        # 只显示主刻度
        ax.yaxis.set_major_locator(plt.FixedLocator(y_ticks))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_yticklabels(y_tick_labels)
        
        # 设置x轴范围
        ax.set_xlim(0.7, 1.01)
        # ax.set_box_aspect(1)
        
        # 参考线和网格
        ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 设置x轴标签
        label_index = chr(97 + col)  # 97是ASCII码中'a'的值

        formatted_label = f"({label_index}) Recall@10 ({selectivity_mapping[query_set]})"

        ax.set_xlabel(formatted_label, 
                     fontproperties=libertine_font,
                     fontsize=20,
                     fontweight='bold')
        
        # 仅第一列显示y轴标签
        if col == 0:
            ax.set_ylabel("QPS", fontsize=18)
        else:
            ax.set_ylabel("")
    
    # 添加图例
    legend_elements = []
    for alg, color in sorted(current_algorithms.items(), key=lambda x: x[0].lower()):
        base_alg = alg.replace('-16', '') if thread_mode == "multi" else alg
        style = ALGORITHM_STYLES.get(base_alg, {
            'color': color,
            'linestyle': line_styles[0],
            'marker': markers[0]
        })
        
        legend_elements.append(plt.Line2D([0], [0], 
                                        color=style['color'], 
                                        marker=style['marker'], 
                                        linestyle=style['linestyle'], 
                                        label=alg,
                                        **plot_params_Legend))
    
    if legend_elements:
        leg = fig.legend(handles=legend_elements, 
                        loc='upper center', 
                        bbox_to_anchor=(0.515, 0.88),
                        ncol=min(6, len(legend_elements)),
                        fontsize=16,
                        frameon=False)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.0, 1, 1.0])
    
    return fig

# 主程序
def main():
    # 加载数据
    print("加载数据文件...")
    all_data = load_all_data()

    if not all_data:
        print("未找到有效的数据文件，请检查路径和文件格式。")
        return
    
    print(f"成功加载数据，共有 {sum(len(data_list) for data_list in all_data.values())} 个数据集。")
    
    # 初始化所有算法的样式
    print("初始化算法样式...")
    initialize_algorithm_styles(all_data)

    query_set = "3"  # 绘制查询集1的图表
    save_path = "/home/ykw/study/Hybrid-ANNS/faiss/plots/labelFilter/exp/exp_5_1_"  # 请修改为实际路径

    if query_set == "3":
        label_mode = "_SingleLabel"
        index = 1
    elif query_set == "7":  
        label_mode = "_MultiLabel"
        index = 3
    
    
    # 创建单线程对比图
    print("创建单线程对比图...")
    fig_single = plot_dataset_comparison(all_data, thread_mode="single", _query_set= query_set)
    plt.savefig(save_path + str(index) + label_mode + "_1thread.svg", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + str(index) + label_mode + "_1thread.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_single)
    
    # 创建16线程对比图
    print("创建16线程对比图...")
    fig_multi = plot_dataset_comparison(all_data, thread_mode="multi", _query_set= query_set)    
    plt.savefig(save_path + str(index+1) + label_mode + "_16thread.svg", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + str(index+1) + label_mode + "_16thread.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_multi)
    
    print("图表已创建并保存！")

if __name__ == "__main__":
    main()