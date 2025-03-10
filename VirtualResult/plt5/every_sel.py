import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import colorsys
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.scale import FuncScale
# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = 'DejaVu Sans'  # 统一字体
plt.style.use('ggplot')  # 设定绘图风格

# 数据路径
data_path = "/data/result"

# 预定义一组颜色（保证相同算法颜色一致）
BASE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#FF4444', '#47D147', '#AA44FF', '#FF9933', '#33CCFF'
]

def generate_distinct_colors(n):
    """生成n个高对比度的颜色"""
    colors = BASE_COLORS.copy()
    if n <= len(colors):
        return colors[:n]
    
    additional_colors_needed = n - len(colors)
    golden_ratio = 0.618033988749895
    hue = 0.
    
    for i in range(additional_colors_needed):
        hue = (hue + golden_ratio) % 1.0
        saturation = 0.7 + (i % 3) * 0.1  # 0.7-0.9 变化
        value = 0.8 + (i % 2) * 0.1       # 0.8-0.9 变化
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

# 生成颜色
colors = generate_distinct_colors(100)

def get_all_result_files():
    """获取所有匹配的结果文件"""
    result_files = []
    algorithm_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for alg in algorithm_dirs:
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
            files = glob.glob(os.path.join(alg_result_path, "sift*_results.csv"))
            for file in files:
                result_files.append((file, alg))
    
    return result_files

def extract_file_info(filename, algorithm):
    """解析文件名提取数据集和查询集编号"""
    base = os.path.basename(filename)
    match = re.match(r'(.+?)_(\d+(?:_\d+)?)_results\.csv', base)
    if match:
        dataset = match.group(1)
        queryset = match.group(2)
        return dataset, queryset, algorithm
    return None, None, None

def load_all_data():
    """加载所有数据"""
    all_data = {}
    files_with_algs = get_all_result_files()
    
    for file, algorithm in files_with_algs:
        dataset, queryset, algorithm = extract_file_info(file, algorithm)
        if dataset and queryset and algorithm:
            try:
                df = pd.read_csv(file)
                if not all(col in df.columns for col in ['Recall', 'QPS']):
                    print(f"Warning: {file} 缺少必要列，跳过。")
                    continue
                
                df['Algorithm'] = algorithm
                
                key = (dataset, queryset)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"加载错误 {file}: {e}")
    
    return all_data

# 将算法分组（16位和非16位）
def group_algorithms(algorithms):
    alg_16bit = []
    alg_other = []
    for alg in algorithms:
        if alg.endswith('-16'):
            alg_16bit.append(alg)
        else:
            alg_other.append(alg)
    return alg_16bit, alg_other

def compute_pareto_frontier(df, x_col, y_col, maximize_x=True, maximize_y=True):
    """计算帕累托前沿"""
    if df.empty:
        return pd.DataFrame()
    
    x = df[x_col].values
    y = df[y_col].values
    
    # 根据 maximize_y 排序 (默认是最大化 y)
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k] if maximize_y else y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]
    
    for i in range(1, len(x_sorted)):
        if (x_sorted[i] > pareto_front_x[-1]) if maximize_x else (x_sorted[i] < pareto_front_x[-1]):
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])
    
    indices = []
    for px, py in zip(pareto_front_x, pareto_front_y):
        matches = df[(df[x_col] == px) & (df[y_col] == py)].index
        if not matches.empty:
            indices.append(matches[0])
    
    return df.loc[indices].sort_values(by=x_col, ascending=not maximize_x)
def create_output_dirs():
    """创建输出目录"""
    dirs = ["plots/5_selectivity/every_sel"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs
def plot_qps_vs_recall(data_list, title, save_path, figsize=(12, 8)):
    # 获取所有算法
    all_algorithms = set()
    for df in data_list:
        if 'Algorithm' in df.columns:
            all_algorithms.add(df['Algorithm'].iloc[0])
    
    # 分组算法
    alg_16, alg_single = group_algorithms(all_algorithms)
    
    def x_transform(x):
        """自定义 x 轴变换: 让 0.0~0.6 变短，使 0.6~1.0 显示正常"""
        return np.where(x < 0.6, x / 10, x - 0.54)

    def x_inverse_transform(x):
        """逆变换，恢复原始 x 值"""
        return np.where(x < (0.6 / 10), x * 10, x + 0.54)

    # 如果两组都有算法，则绘制两张图
    if alg_16 and alg_single:
        # 绘制16thread算法图
        plt.figure(figsize=figsize)
        legend_handles = []
        legend_labels = []

        for i, df in enumerate([d for d in data_list if d['Algorithm'].iloc[0] in alg_16]):
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            if not pareto_df.empty:
                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', color=plt.cm.tab10(i % 10))
                legend_handles.append(line)
                legend_labels.append(algorithm)

        # 设置背景为白色
        plt.gcf().set_facecolor('white')
        plt.gca().set_facecolor('white')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{title} (16-Thread algorithms)", fontsize=14)
        plt.grid(True, alpha=0.3)

        # 设置 y 轴为对数刻度
        plt.yscale('log', base=10)
        plt.ylim(bottom=10**1)

        # 使用自定义的 x 轴变换
        plt.gca().set_xscale(FuncScale(plt.gca(), (x_transform, x_inverse_transform)))

        # 设置 x 轴刻度
        xticks = np.append([0.0], np.arange(0.6, 1.05, 0.05))  # 0.0 独立，0.6 到 1.0 以 0.05 递增
        plt.xticks(xticks)

        # 添加垂直参考线
        plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.7, label='Recall=0.95')

        # 自定义 y 轴刻度标签为 10^n 格式
        def y_label_formatter(value, pos):
            """自定义格式化器：将 y 轴刻度显示为 10^k 形式"""
            if value == 0:
                return "0"
            return f"$10^{{{int(np.log10(value))}}}$"

        plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_label_formatter))

        if legend_handles:
            # 将参考线添加到图例中
            legend_handles.append(plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.7))
            legend_labels.append('Recall=0.95')
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='upper left', facecolor='white',bbox_to_anchor=(1, 1))

        # 保存16thread算法图
        save_path_16 = save_path.replace('.png', '_16thread.png')
        os.makedirs(os.path.dirname(save_path_16), exist_ok=True)
        plt.savefig(save_path_16, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制single thread算法图
        plt.figure(figsize=figsize)
        legend_handles = []
        legend_labels = []

        for i, df in enumerate([d for d in data_list if d['Algorithm'].iloc[0] in alg_single]):
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            if not pareto_df.empty:
                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', color=plt.cm.tab10(i % 10))
                legend_handles.append(line)
                legend_labels.append(algorithm)

        # 设置背景为白色
        plt.gcf().set_facecolor('white')
        plt.gca().set_facecolor('white')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(f"{title} (single thread algorithms)", fontsize=14)
        plt.grid(True, alpha=0.3)

        # 设置 y 轴为对数刻度
        plt.yscale('log', base=10)
        plt.ylim(bottom=10**1)

        # 使用自定义的 x 轴变换
        plt.gca().set_xscale(FuncScale(plt.gca(), (x_transform, x_inverse_transform)))

        # 设置 x 轴刻度
        plt.xticks(xticks)

        # 添加垂直参考线
        plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.7, label='Recall=0.95')

        # 自定义 y 轴刻度标签为 10^n 格式
        plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_label_formatter))

        if legend_handles:
            # 将参考线添加到图例中
            legend_handles.append(plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.7))
            legend_labels.append('Recall=0.95')
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='upper left',facecolor='white', bbox_to_anchor=(1, 1))

        # 保存other算法图
        save_path_other = save_path.replace('.png', '_1thread.png')
        os.makedirs(os.path.dirname(save_path_other), exist_ok=True)
        plt.savefig(save_path_other, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # 如果只有一组算法，则绘制单张图
        plt.figure(figsize=figsize)
        legend_handles = []
        legend_labels = []
        
        for i, df in enumerate(data_list):
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            if not pareto_df.empty:
                line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker='o', linestyle='-', color=plt.cm.tab10(i % 10))
                legend_handles.append(line)
                legend_labels.append(algorithm)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('QPS', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 设置y轴为对数刻度，从10开始
        plt.yscale('log', base=10)
        plt.ylim(bottom=10**1)
        
        # 设置x轴范围从0.6开始，步长0.05
        plt.xlim(0.6, 1.0)
        plt.xticks(np.arange(0.6, 1.05, 0.05))
        
        # 添加0.95垂直参考线
        plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.7, label='Recall=0.95')
        
        # 自定义y轴刻度标签为10^n格式
        plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(y_label_formatter))
        
        if legend_handles:
            # 将参考线添加到图例中
            legend_handles.append(plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.7))
            legend_labels.append('Recall=0.95')
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='upper left', facecolor='white',bbox_to_anchor=(1, 1))
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

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
    
    
    print("5. 绘制选择性实验比较图...")
    plot_selectivity_experiments(all_data, output_dirs[0])
    
   
    
   
    
    print("所有图表绘制完成！")
if __name__ == "__main__":
    main() 