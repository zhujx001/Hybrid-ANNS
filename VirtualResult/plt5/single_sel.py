import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import colorsys
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter
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

def compute_pareto_frontier(df, x_col, y_col):
    """计算帕累托前沿"""
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
    
    return df.loc[indices].sort_values(by=x_col, ascending=True)

def create_output_dirs():
    """创建输出目录"""
    dirs = ["plots/5_selectivity/single_sel"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def plot_selectivity_experiments(all_data, output_dir):
    """绘制选择性实验图"""
    single_label_selectivity = ["3_1", "3_2", "3_3", "3_4"]
    selectivity_mapping = {
        "3_1": "1%", "3_2": "25%", "3_3": "50%", "3_4": "75%"
    }
    
    datasets = set(k[0] for k in all_data.keys())

    def x_transform(x):
        """自定义 x 轴变换: 让 0.0~0.6 变短，使 0.6~1.0 显示正常"""
        return np.where(x < 0.6, x / 10, x - 0.54)  

    def x_inverse_transform(x):
        """逆变换，恢复原始 x 值"""
        return np.where(x < (0.6 / 10), x * 10, x + 0.54)

    for dataset in datasets:
        for thread_type in ["single_thread", "16_threads"]:
            plt.figure(figsize=(15, 10))
            legend_handles, legend_labels = [], []
            algorithm_data = {}
            
            for query_set in single_label_selectivity:
                key = (dataset, query_set)
                if key in all_data:
                    for df in all_data[key]:
                        algorithm = df['Algorithm'].iloc[0]
                        
                        if (thread_type == "single_thread" and "16" in algorithm) or (
                                thread_type == "16_threads" and "16" not in algorithm):
                            continue
                        
                        if algorithm not in algorithm_data:
                            algorithm_data[algorithm] = []
                        algorithm_data[algorithm].append((query_set, df))
            
            for alg_idx, (algorithm, df_list) in enumerate(sorted(algorithm_data.items())):
                color = plt.cm.tab10(alg_idx % 10)  # 使用颜色映射
                linestyles = ['-', '--', ':', '-']
                markers = ['o', 'o', 'o', 's']

                for query_set_idx, (query_set, df) in enumerate(sorted(df_list, key=lambda x: x[0])):
                    linestyle = linestyles[query_set_idx % 4]
                    marker = markers[query_set_idx % 4]

                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
                    if not pareto_df.empty:
                        line, = plt.plot(pareto_df['Recall'], pareto_df['QPS'], marker=marker, linestyle=linestyle,
                                        color=color, label=f"{algorithm} - {selectivity_mapping[query_set]}")
                        legend_handles.append(line)
                        legend_labels.append(f"{algorithm} - {selectivity_mapping[query_set]}")
                
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')

            # # 设置纵坐标 log 轴
            plt.yscale('log', base=10)
            
        
            # 设置非线性 X 轴
            plt.gca().set_xscale(FuncScale(plt.gca(), (x_transform, x_inverse_transform)))

            # 设置 x 轴刻度
            xticks = np.append([0.0], np.arange(0.6, 1.05, 0.05))  # 0.0 独立，0.6 到 1.0 以 0.05 递增
            plt.xticks(xticks)

            # 添加垂直参考线
            plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.7, label='Recall=0.95')
            
            # 设置 y 轴刻度格式
            def y_label_formatter(value, pos):
                """自定义格式化器：将y轴刻度显示为 10^k 形式"""
                if value == 0:
                    return "0"
                return f"$10^{{{int(np.log10(value))}}}$"

            # 设置 y 轴定位器和格式化器
            plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
            plt.gca().yaxis.set_major_formatter(FuncFormatter(y_label_formatter))

            # 设置 y 轴范围
            plt.ylim(1e1, 1e6)  # 确保y轴从10^2到10^5

            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('QPS', fontsize=12)
            plt.title(f"{dataset} - {thread_type.replace('_', ' ')} Selectivity Experiment", fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 显示图例
            plt.legend(legend_handles, legend_labels, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', framealpha=1.0)
            
            save_path = os.path.join(output_dir, f"{dataset}_{thread_type}_selectivity.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.gca().set_facecolor('white')  
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white') 
            plt.close()

def main():
    output_dirs = create_output_dirs()
    all_data = load_all_data()
    plot_selectivity_experiments(all_data, output_dirs[0])
    print("图表绘制完成！")

if __name__ == "__main__":
    main()