'''实验一/四绘图代码'''
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
data_path = "/data/ood/result"  # 请修改为实际路径

# 创建一个字体对象，指定字体文件的路径
libertine_font = fm.FontProperties(
    fname='/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')

# 基础颜色列表
colors = [ # 主色系（增强饱和度）
 '#F39C12', # 深邃蓝（原#5E81AC提纯）
 '#6EC1E0', # 电光冰蓝（原#88C0D0去灰）
 '#F39C12', # 亮橙
 '#E74C3C', # 警报红（原#BF616A加深）
 
 # 扩展高冲击力颜色
 '#34495E', # 钢蓝灰（原#4C566A微调）

 '#2ECC71', # 翡翠绿

 # 辅助色（强化对比）
 '#48D1CC', # 土耳其蓝
 '#9B59B6', # 宝石紫（原#B48EAD增饱和）
 '#E67E22', # 南瓜橙（替换原#D08770）
 '#8FCB6B', # 苹果绿（原#A3BE8C增艳）
 '#3498DB', # 荧光蓝（原#81A1C1提亮）
]


# 扩展线型和标记列表
line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '*', '+', 'x', '|', '1', '2', '3', '4']

plot_params = {
    'markersize': 6,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

plot_params_Legend = {
    'markersize': 8,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
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
        alg_result_path = os.path.join(data_path, alg)
        if os.path.exists(alg_result_path):
            files = glob.glob(os.path.join(alg_result_path, "*_results.csv"))
            for file in files:
                result_files.append((file, alg))
    
    return result_files

# 提取数据集名称和查询集编号
def extract_file_info(filename, algorithm):
    base = os.path.basename(filename)
    match = re.match(r'(.+?)_results\.csv', base)
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
                
                if 'Recall' not in df.columns or 'QPS' not in df.columns:
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                key = (dataset, algorithm)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return all_data

# 计算稀疏化的帕累托前沿
def compute_pareto_frontier(df, x_col, y_col, threshold_x=0.05, threshold_y_ratio=5):   # 单标签参数
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

# 通用绘图函数
def plot_dataset_comparison(all_data, thread_mode="multi", figsize=(10, 4.5)):
    """
    绘制QPS vs Recall性能对比图，每个数据集一个子图，左右并排
    
    参数:
    all_data: 所有数据的字典
    thread_mode: "single"表示单线程图，"multi"表示16线程图
    figsize: 图表尺寸
    
    返回:
    fig: matplotlib图表对象
    """
    # 收集所有唯一的数据集
    datasets = sorted(set(k[0] for k in all_data.keys()))
    
    # 收集所有唯一的算法并按线程模式筛选
    all_algorithms = set(key[1] for key in all_data.keys())

    # 为每个算法分配一种颜色
    for i, alg in enumerate(all_algorithms):
        alg = alg.replace('-16', '')
        if alg not in ALGORITHM_STYLES:
                ALGORITHM_STYLES[alg] = {
                'color': colors[i % len(colors)],
                'linestyle': line_styles[i % len(line_styles)],
                'marker': markers[i % len(markers)]
            }
    
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    
    # 调整间距
    gs = GridSpec(1, len(datasets), figure=fig, wspace=0.2, top=0.85, bottom=0.15)
    
    # 记录当前图表中的算法以便创建统一图例
    current_algorithms = {}
    
    # 收集所有Y轴数据以统一刻度
    all_y_values = []
    
    # 预处理数据并收集Y值
    for dataset_idx, dataset in enumerate(datasets):
        for algorithm in all_algorithms:
            key = (dataset, algorithm)
            if key in all_data and all_data[key]:
                for df in all_data[key]:
                    if not df.empty and 'QPS' in df.columns and 'Recall' in df.columns:
                        all_y_values.extend(df['QPS'].values)
    
    
    # 遍历所有数据集并创建子图
    for dataset_idx, dataset in enumerate(datasets):
        # 创建子图
        ax = fig.add_subplot(gs[0, dataset_idx])

        # 收集Y轴数据
        y_data_list = []
        
        # 为每个算法绘制曲线
        for algorithm in all_algorithms:
            key = (dataset, algorithm)
            if key in all_data and all_data[key]:
                for df in all_data[key]:
                    if not df.empty and 'QPS' in df.columns and 'Recall' in df.columns:
                        # 处理算法名和样式
                        base_name = algorithm.replace('-16', '')
                        label = base_name
                        
                        # 获取算法样式
                        style = ALGORITHM_STYLES.get(base_name, {
                            'color': colors[0],
                            'linestyle': line_styles[0],
                            'marker': markers[0]
                        })
                        
                        # 记录此算法已在图中
                        current_algorithms[label] = style

                        # 计算并绘制帕累托前沿
                        pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')

                        if not pareto_df.empty:
                            filtered_df = pareto_df[pareto_df['Recall'] >= 0.3]
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
        ax.set_xlim(0.3, 1.01)
        # ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 网格线
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 设置x轴标签
        label_index = chr(97 + dataset_idx)  # 97是ASCII码中'a'的值
        formatted_label = f"({label_index}) Recall@10 ({dataset})"
        ax.set_xlabel(formatted_label, fontproperties=libertine_font, fontweight='bold', fontsize=20)
        
        # 仅第一列显示y轴标签
        if dataset_idx == 0:
            ax.set_ylabel("QPS", fontsize=18)
        else:
            ax.set_ylabel("")
    
    legend_elements = []
    # Create a mapping for algorithm name display
    alg_display_names = {
        'ParDiskANN': 'parlaydiskann',
        'ParHNSW': 'parlayhnsw',
        'ParHCNNG': 'parlayhcnng',
        'ParPyNNDescent': 'parlaypynn'
    }
    
    for alg, style in sorted(current_algorithms.items(), key=lambda x: x[0].lower()):
        # Use the mapped name if available, otherwise use the original name
        display_name = alg_display_names.get(alg, alg)
        
        legend_elements.append(plt.Line2D([0], [0], 
                                       color=style['color'], 
                                       marker=style['marker'], 
                                       linestyle=style['linestyle'], 
                                       label=display_name,
                                       **plot_params_Legend))
    
    if legend_elements:
        leg = fig.legend(handles=legend_elements, 
                        loc='upper center', 
                        bbox_to_anchor=(0.51, 1.02),
                        ncol=min(len(legend_elements), 5),
                        fontsize=12.5,
                        frameon=False)

    
    # 调整布局
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    
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

    save_path = ""  # 请修改为实际路径
    
    # 创建16线程对比图
    print("创建16线程对比图...")
    fig_multi = plot_dataset_comparison(all_data, thread_mode="multi")
    plt.savefig(save_path + "exp_9_16thread.svg", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + "exp_9_16thread.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_multi)
    
    print("图表已创建并保存！")

if __name__ == "__main__":
    main()