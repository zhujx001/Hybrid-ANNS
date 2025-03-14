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
data_path = "/data/indexdata"  # 请修改为实际路径

# 创建一个字体对象，指定字体文件的路径
libertine_font = fm.FontProperties(
    fname='/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')

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

# 提取数据集名称和查询集编号
def extract_file_info(filename, algorithm):
    base = os.path.basename(filename)
    match = re.match(r'(.+?).csv', base)
    if match:
        dataset = match.group(0)
        dataset = dataset.replace(".csv", "")
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
                if 'query_set' not in df.columns or 'memory_mb' not in df.columns or 'build_time' not in df.columns or 'index_size_mb' not in df.columns:
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                
                key = (dataset, algorithm)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return all_data

def plot_dataset_comparison(all_data, plot_col=None, query_set="1", figsize=(30, 5)):
    """
    对指定查询集的数据构建柱状图比较
    
    参数:
    all_data: 加载的所有数据
    plot_col: 要绘制的列，可以是 'memory_mb', 'build_time', 'index_size_mb' 或 None(绘制所有列)
    query_set: 查询集ID，默认为"1"
    figsize: 图像大小
    """
    # 确定要绘制的列
    cols_to_plot = [plot_col] if plot_col else ['memory_mb', 'build_time', 'index_size_mb']
    
    # 收集所有唯一的数据集
    datasets = sorted(list(set([key[0] for key in all_data.keys()])))
    
    # 收集所有唯一的算法
    algorithms = sorted(list(set([key[1] for key in all_data.keys()])))
    
    # 为每个算法分配一种颜色
    for i, alg in enumerate(algorithms):
        if alg not in ALGORITHM_STYLES:
            ALGORITHM_STYLES[alg] = {'color': colors[i % len(colors)]}
    
    # 对于每个要绘制的列创建一个子图
    for col_idx, column in enumerate(cols_to_plot):
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        
        # Y轴标签映射
        y_labels = {
            'memory_mb': 'Memory Usage (MB)',
            'build_time': 'Build Time (s)',
            'index_size_mb': 'Index Size (MB)'
        }
        
        bar_width = 0.8 / len(algorithms)
        
        # 收集每个数据集和算法的值
        all_values = []
        
        # 绘制每个算法的柱子
        for alg_idx, alg in enumerate(algorithms):
            values = []
            
            for ds in datasets:
                key = (ds, alg)
                if key in all_data and all_data[key]:
                    # 筛选指定查询集的数据
                    filtered_dfs = []
                    for df in all_data[key]:
                        if query_set in df['query_set'].values:
                            filtered_df = df[df['query_set'] == query_set]
                            if not filtered_df.empty:
                                filtered_dfs.append(filtered_df)
                    
                    # 如果有数据，计算平均值
                    if filtered_dfs:
                        value = pd.concat(filtered_dfs)[column].mean()
                        values.append(value)
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            
            # 记录所有非NaN值
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                all_values.extend(valid_values)
            
            # 设置每个柱的位置
            x = np.arange(len(datasets))
            offset = bar_width * (alg_idx - len(algorithms) / 2 + 0.5)
            
            # 绘制柱状图
            bars = ax.bar(x + offset, values, bar_width, 
                   label=alg, 
                   color=ALGORITHM_STYLES[alg]['color'],
                   edgecolor='black', linewidth=2)
            
            # # 添加数值标签（只显示值较大的）
            # for i, bar in enumerate(bars):
            #     if not np.isnan(values[i]):
            #         height = bar.get_height()
            #         # 为了避免标签拥挤，可以设定一个阈值
            #         if column == 'build_time':  # 时间通常需要更精确的表示
            #             value_text = f"{values[i]:.2f}"
            #         else:
            #             value_text = f"{values[i]:.1f}"
                    
            #         ax.text(bar.get_x() + bar.get_width()/2., height,
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
        plt.ylabel(y_labels[column],  fontsize=20)
        
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
        output_dir = "/home/ykw/study/Hybrid-ANNS/faiss/plots/indexData"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{column}_comparison_query{query_set}.svg", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/{column}_comparison_query{query_set}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    


def main():
    # 加载数据
    print("加载数据文件...")
    all_data = load_all_data()
    
    if not all_data:
        print("未找到有效的数据文件，请检查路径和文件格式。")
        return
    
    print(f"成功加载数据，共有 {sum(len(data_list) for data_list in all_data.values())} 个数据集。")
    plot_dataset_comparison(all_data=all_data)

if __name__ == "__main__":
    main()