import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.font_manager as fm
import re

# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 使用通用字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 设置图表样式为白色背景
plt.style.use('default')  # 使用默认风格，白色背景

# 定义数据路径
data_path = "/data/result"

# 创建一个字体对象，指定字体文件的路径
libertine_font = fm.FontProperties(
    fname='/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')

# 为每个算法定义唯一的颜色（标准版本和16位版本使用相同颜色）
ALGORITHM_COLORS = {
    'FilteredVamana': '#F39C12',     # 橙色
    'StitchedVamana': '#316bea',     # 蓝色
    'NHQ': '#00bf00',           # 绿色
    'Puck': '#E74C3C',          # 粉色
    'UNG': '#bfbf00',           # 灰色
    'CAPS': '#AE73D0',          # 紫色
}

# 为不同的数据集定义不同的线型
DATASET_LINESTYLES = {
    'sift_1': '-',      # 实线
    'sift_2_1': '--',   # 虚线
}

# 标记样式
DATASET_MARKERS = {
    'sift_1': 'o',      # 圆形
    'sift_2_1': '^',    # 方形
}

plot_params = {
    'markersize': 12,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 2,         # 标记边缘宽度
    'linewidth': 1.4        # 线条粗细
}

# 获取实验结果文件
def get_result_files():
    # 目标算法列表，添加了puck-16
    target_algorithms = [
        'FilteredVamana',
        'StitchedVamana',
        'NHQ', 'Puck',
        'UNG', 'CAPS'
    ]
    
    result_files = []
    
    for alg in target_algorithms:
        # 构建算法的result目录路径
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
            # 获取所有三种数据集的结果文件
            for pattern in ["sift_1_results.csv", "sift_2_1_results.csv"]:
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
            
            all_data['single_thread'].append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
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
def get_y_range_and_ticks(y_data_list):
    """获取y轴范围和刻度，与exp_1.py相同"""
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

def superscript(n):
    """将数字转换为上标，与exp_1.py相同"""
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'
    }
    return ''.join(superscript_map[digit] for digit in str(n))

def plot_comparison_charts(all_data):
    # NEW: Create sets to collect all unique algorithms and datasets
    all_algorithms = set()
    all_datasets = set(['sift_1', 'sift_2_1'])
    
    # NEW: First pass - collect all algorithms
    for group in ['single_thread',]:
        for df in all_data[group]:
            all_algorithms.add(df['Algorithm'].iloc[0])
    
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
        
        
        # 收集y轴数据用于自动确定范围
        y_data_list = []
        
        # 按算法绘制线条
        for algorithm in sorted(grouped_data.keys()):
            for dataset in ['sift_1', 'sift_2_1']:
                if dataset in grouped_data[algorithm]:
                    df = grouped_data[algorithm][dataset]
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
                    
                    if not pareto_df.empty:
                        # 过滤 Recall 小于0.6以及等于1.01的数据，与exp_1.py保持一致
                        filtered_df = pareto_df[(pareto_df['Recall'] >= 0.7) & (pareto_df['Recall'] != 1.01)]
                        if not filtered_df.empty:
                            y_data_list.append(filtered_df['QPS'].tolist())
                            
                            line, = plt.plot(filtered_df['Recall'], filtered_df['QPS'], 
                                            marker=DATASET_MARKERS.get(dataset, 'o'),
                                            linestyle=DATASET_LINESTYLES.get(dataset, '-'), 
                                            color=ALGORITHM_COLORS.get(algorithm, '#000000'),
                                            **plot_params)
                            legend_handles.append(line)
                            legend_labels.append(f"{algorithm} ({dataset})")
        
        # 自动设置y轴范围和刻度
        y_min, y_max, y_ticks, y_tick_labels = get_y_range_and_ticks(y_data_list)
        
        # 在x=0.95处添加灰色虚线，与exp_1.py保持一致的样式
        plt.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', labelsize=36)
        plt.xlabel('Recall@10 (SIFT1M)', fontsize=50, fontproperties=libertine_font)
        plt.ylabel('QPS', fontsize=38)
        # plt.title('SIFT Dataset - Experiment 2 - Single Thread Algorithms', fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 设置对数刻度和自动计算的范围
        plt.yscale('log')
        plt.ylim(y_min, y_max)
        plt.gca().yaxis.set_major_locator(plt.FixedLocator(y_ticks))
        plt.gca().yaxis.set_minor_locator(plt.NullLocator())
        plt.gca().set_yticklabels(y_tick_labels)
        
        # 设置x轴范围为0.6到1.01，但仅显示刻度至1.0，与exp_1.py保持一致
        plt.xlim(0.7, 1.01)
        plt.xticks([0.7, 0.8, 0.9, 1.0])
        plt.gca().set_xticklabels([f"{tick:.1f}" for tick in [0.7, 0.8, 0.9, 1.0]])
        
        # if legend_handles:
        #     # 创建具有多列的图例，使其更紧凑
        #     plt.legend(legend_handles, legend_labels, fontsize=12, loc='best', ncol=3)
        
        # 保存为矢量图格式（SVG）
        save_path_svg = os.path.join("/data/plots/exp","exp_2_1.svg")
        save_path_pdf = os.path.join("/data/plots/exp","exp_2_1.pdf")
        plt.savefig(save_path_svg, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        
        
        plt.close()
        print(f"Single thread algorithms comparison chart saved as vector graphic to {save_path_svg}")
    


# ...existing code...

def create_standalone_legend():
    """Create a standalone legend figure showing all algorithms with their dataset markers"""
    from matplotlib.legend_handler import HandlerLine2D
    
    # Create a wider figure to ensure columns don't overlap
    fig, ax = plt.subplots(figsize=(20, 2.5))
    ax.set_axis_off()  # Hide axes
    
    datasets = {
        'sift_1': 'S',     # S for SIFT-1
        'sift_2_1': 'M'    # M for SIFT-2
    }
    
    # Define the three columns explicitly for 4+4+2 layout
    column1_items = [
        ('FilteredVamana', 'sift_1'),
        ('FilteredVamana', 'sift_2_1'),
        ('StitchedVamana', 'sift_1'),
        ('StitchedVamana', 'sift_2_1')
    ]
    
    column2_items = [
        ('NHQ', 'sift_1'),
        ('NHQ', 'sift_2_1'),
        ('Puck', 'sift_1'),
        ('Puck', 'sift_2_1')
    ]
    
    column3_items = [
        ('UNG', 'sift_1'),
        ('UNG', 'sift_2_1'),
        ('CAPS', 'sift_1'),
        ('CAPS', 'sift_2_1')
    ]
    
    # Create three separate legend objects with much wider spacing
    columns = [column1_items, column2_items, column3_items]
    # Use very distinct positions to ensure no overlap
    legend_positions = [0.2, 0.5, 0.75] 
    
    for col_idx, (column_items, x_pos) in enumerate(zip(columns, legend_positions)):
        # Create handles and labels for this column
        handles = []
        labels = []
        
        for alg, dataset in column_items:
            shortname = datasets[dataset]
            line = plt.Line2D([0], [0], 
                          marker=DATASET_MARKERS.get(dataset, 'o'),
                          linestyle=DATASET_LINESTYLES.get(dataset, '-'), 
                          color=ALGORITHM_COLORS.get(alg, '#000000'),
                          **plot_params)
            handles.append(line)
            labels.append(f"{alg} ({shortname})")
        
        # Use consistent alignment for all columns
        # This helps with predictable positioning
        alignment = 'center'
            
        # Add this column's legend to the figure
        leg = ax.legend(handles, labels, 
                      loc=alignment,
                      bbox_to_anchor=(x_pos, 0.5),
                      frameon=False,
                      fontsize=24,
                      handlelength=3,
                      handletextpad=1.5,
                      handler_map={plt.Line2D: HandlerLine2D(numpoints=1)})
        
        # Add the legend to the figure
        ax.add_artist(leg)
    
    # Save the legend as separate files with tight bounding box
    legend_path_svg = os.path.join("/data/plots/exp", "exp_2_legend.svg")
    legend_path_pdf = os.path.join("/data/plots/exp", "exp_2_legend.pdf")
    plt.savefig(legend_path_svg, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(legend_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Standalone legend saved to {legend_path_svg} with 4+4+2 layout")

def main():
    print("Loading data files for SIFT dataset experiments...")
    all_data = load_data()
    
    if not all_data['single_thread']:
        print("No valid data files found. Please check the path and file format.")
        return
    
    single_thread_count = len(all_data['single_thread'])
    
    print("Generating comparison charts...")
    plot_comparison_charts(all_data)
    
    # Add call to create the standalone legend
    create_standalone_legend()
    
    print("All charts generated successfully!")



def main():
    print("Loading data files for SIFT dataset experiments...")
    all_data = load_data()
    
    if not all_data['single_thread'] and not all_data['multi_thread']:
        print("No valid data files found. Please check the path and file format.")
        return
    
    single_thread_count = len(all_data['single_thread'])
    
    print("Generating comparison charts...")
    plot_comparison_charts(all_data)

    create_standalone_legend()


    
    print("All charts generated successfully!")

if __name__ == "__main__":
    main()