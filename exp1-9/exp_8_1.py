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
data_path = "/data/rangefilter/result"  # 请修改为实际路径

libertine_font = fm.FontProperties(
    fname='/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')
# 基础颜色列表
colors = [ # 主色系（增强饱和度）
 '#F39C12', # 深邃蓝（原#5E81AC提纯）
 '#6EC1E0', # 电光冰蓝（原#88C0D0去灰）
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

plot_params = {
    'markersize': 6,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

plot_legend_params = {
    'markersize': 8,                # 标记大小
    'markerfacecolor': (1, 1, 1, 0.8),     # 标记填充颜色（白色）
    'markeredgewidth': 1,         # 标记边缘宽度
    'linewidth': 1.2           # 线条粗细
}

line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '*', '+', 'x', '|', '1', '2', '3', '4']


def get_all_result_files():
    result_files = []
    # Get all range directories (range_2, range_4, range_6, range_8)
    range_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith("range_")]
    
    for range_dir in range_dirs:
        # Extract range value (2, 4, 6, 8)
        range_value = range_dir.split("_")[1]
        range_path = os.path.join(data_path, range_dir)
        
        # Get algorithm directories in this range directory
        algorithm_dirs = [d for d in os.listdir(range_path) if os.path.isdir(os.path.join(range_path, d))]
        
        for alg in algorithm_dirs:
            alg_result_path = os.path.join(range_path, alg, "result")
            if os.path.exists(alg_result_path):
                # Find all CSV files
                files = glob.glob(os.path.join(alg_result_path, "*.csv"))
                for file in files:
                    result_files.append((file, range_value, alg))
            else:
                # If there's no "result" subdirectory, look for CSV files directly in the algorithm directory
                files = glob.glob(os.path.join(range_path, alg, "*.csv"))
                for file in files:
                    result_files.append((file, range_value, alg))
    
    return result_files

# 提取数据集名称和查询集编号
def extract_file_info(filename):
    base = os.path.basename(filename)
    match = re.match(r'(.+?)\.csv', base)
    if match:
        dataset = match.group(1)
        return dataset
    return None

# 加载所有数据
def load_all_data():
    all_data = {}
    files_with_ranges_algs = get_all_result_files()
    
    for file, range_value, alg in files_with_ranges_algs:
        dataset = extract_file_info(file)
        if dataset and range_value and alg:
            try:
                df = pd.read_csv(file)
                
                if 'Recall' not in df.columns or 'QPS' not in df.columns:
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                df['Algorithm'] = alg
                df['Range'] = range_value
                
                key = (dataset, f"range_{range_value}")
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    # all_data = {('deep', 'range_2'): [df1, df2, ...], ('deep', 'range_4'): [df1, df2, ...], ...}
    return all_data

def compute_pareto_frontier(df, x_col, y_col, threshold_x=0.025, threshold_y_ratio=2.5):
    """
    计算帕累托前沿，并进行稀疏化处理，确保点在x轴和y轴（对数空间）的间距不小于给定阈值
    
    参数:
    threshold_x: x轴的绝对阈值
    threshold_y_ratio: y轴对数空间的相对阈值，表示相邻点的y值比例变化至少为(1+threshold_y_ratio)
    """
    if df is None or df.empty:
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
    # max_y 只要是1000多，max_power就是4
    max_power = np.ceil(np.log10(max_y))
    
    # 确保至少有2个主刻度
    if max_power - min_power < 1:
        min_power = max(0, max_power - 2)
    
    # 设置y轴范围，稍微扩展一点
    y_min = 10 ** min_power / 1.5
    y_max = 10 ** max_power * 1.5
    
    # 生成刻度值和标签
    # ticks = [10, 100, 1000, 10000]
    ticks = [10 ** i for i in range(int(min_power), int(max_power) + 1)]
    # tick_labels = ["10²", "10³", "10⁴", "10⁵"]
    tick_labels = [f"10{superscript(i)}" for i in range(int(min_power), int(max_power) + 1)]
    
    return y_min, y_max, ticks, tick_labels

# 辅助函数：将数字转换为上标
def superscript(n):
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'
    }
    return ''.join(superscript_map[digit] for digit in str(n))

def plot_3_4_comparison(all_data):
    range_set = ['range_2', 'range_4', 'range_6', 'range_8'] # 这里假设我们只关心查询集1
    alg_set = ['DSG', 'IRange', 'SeRF', 'UNIFY', 'WinFilter']
    datasets = ['wit', 'deep', 'yt8m']
    
    # 创建全局颜色字典，确保同一算法无论单线程还是16线程都使用相同颜色
   
    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, query in enumerate(alg_set):
        algorithm_colors[query] = colors[i % len(colors)]
        algorithm_line_styles[query] = line_styles[i % len(line_styles)]
        algorithm_markers[query] = markers[i % len(markers)]  

    # 创建一个3行4列的大图，单位是英寸
    fig = plt.figure(figsize=(30, 22))
    
    # 调整上下间距，给图例留出空间
    # wspace = 0.15表示列间距为子图宽度的15%
    # hspace = 0.4表示行间距为子图高度的40%
    # height_ratios: 这个参数用于指定网格中每一行的高度比例
    # 顶部的 "range fraction: 2⁻²" 等标题（在 top=0.85 以上的空间内）
    gs = GridSpec(3, 4, figure=fig, wspace=0.15, hspace=0.4, height_ratios=[1, 1, 1], top=0.85, bottom=0.15)

    # 定义每列的range fraction标签
    range_labels = [r"range fraction: $2^{-2}$", r"range fraction: $2^{-4}$", 
                   r"range fraction: $2^{-6}$", r"range fraction: $2^{-8}$"]
    
    # 遍历所有算法
    for idx_dataset, dataset in enumerate(datasets):
        for idx_range, range_val in enumerate(range_set):
            row = idx_dataset
            col = idx_range
            ax_single = fig.add_subplot(gs[row, col])

            single_thread_y_data = []
            for alg in alg_set:
                key = (dataset, range_val)
                if key not in all_data:
                    continue
                    
                data_list = all_data[key]
                df = None
                for d in data_list:
                    if d['Algorithm'].iloc[0] == alg:
                        df = d
                color = algorithm_colors[alg]
                linestyle = algorithm_line_styles[alg]
                marker = algorithm_markers[alg]    
                pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS')
                if not pareto_df.empty:
                    # 过滤 Recall 小于0.7的数据
                    filtered_df = pareto_df[(pareto_df['Recall'] >= 0.4)]
                    if not filtered_df.empty:
                        x_data = filtered_df['Recall'].tolist()
                        y_data = filtered_df['QPS'].tolist()
                        single_thread_y_data.append(y_data)
                        # 画df这条线
                        ax_single.plot(x_data, y_data, marker=marker,linestyle=linestyle, label=datasets, color=color, **plot_params)
            
            # 添加range fraction标签到子图顶部
            # pad 的值是以**点（points）**为单位的。1 点 (point) 等于 1/72 英寸
            ax_single.set_title(range_labels[idx_range], fontsize=18, fontweight='normal', pad=10)
            # 设置x轴和y轴标签的刻度字体大小
            ax_single.tick_params(axis='both', labelsize=18)
                        
            # 设置单线程图的y轴范围和刻度
            y_min_single, y_max_single, y_ticks_single, y_tick_labels_single = get_y_range_and_ticks(single_thread_y_data)
            # 以 10 为底的对数
            ax_single.set_yscale('log')
            ax_single.set_ylim(y_min_single, y_max_single)
             # 只显示主刻度
            ax_single.yaxis.set_major_locator(plt.FixedLocator(y_ticks_single))
            ax_single.yaxis.set_minor_locator(plt.NullLocator())
            ax_single.set_yticklabels(y_tick_labels_single)
            

            # 设置x轴范围为0.7到1.01，但仅显示刻度至1.0
            ax_single.set_xlim(0.4, 1.01)
            ax_single.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

            ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
            ax_single.grid(True, linestyle=':', alpha=0.6)

            # 添加Recall@10标签
            formatted_label = "Recall@10"
            ax_single.set_xlabel(formatted_label, 
                            fontproperties=libertine_font,
                            fontsize=24,
                            fontweight='bold')
                
            # 仅第一列显示y轴标签
            if idx_range == 0:
                ax_single.set_ylabel("QPS", fontsize=18)
            else:
                ax_single.set_ylabel("")
    
    # 为每行添加数据集标签
    for idx_dataset, dataset in enumerate(datasets):
        label = f"({chr(97+idx_dataset)}) {dataset}"
        
        # 计算行的中心位置
        # 需要在原始坐标系中计算，bbox_to_anchor的坐标是相对于整个图像的
        y_pos = 0.86 - (idx_dataset+1) * 0.26 + 0.02# 根据行间距调整
        
        fig.text(0.5, y_pos, label,
                ha='center', va='center',
                fontsize=20, fontweight='bold')
    
    # 创建算法图例
    legend_elements = []
    for alg in alg_set:
        legend_elements.append(plt.Line2D([0], [0], color=algorithm_colors[alg], marker=algorithm_markers[alg], linestyle=algorithm_line_styles[alg],
                                            label=alg, **plot_legend_params))
    
    if legend_elements:
        leg = fig.legend(handles=legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, 0.9), ncol=min(15, len(legend_elements)),
                         fontsize=20, frameon=False)
        leg.get_title().set_fontweight('bold')
    
    # 调整布局，为底部的数据集标签留出空间
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    return fig

# 在Jupyter环境中使用的代码

# 创建一个新的函数用于绘制多线程算法图表
def plot_1_6_comparison(all_data):
    range_set = ['range_2', 'range_8']
    alg_set = ['DSG', 'IRange', 'SeRF', 'UNIFY', 'WinFilter','ACORN-1', 'ACORN-γ', 'Faiss', 'Milvus', 'VBASE','PASE']
    datasets = ['wit', 'deep', 'yt8m']
    
    # 为图例创建格式化后的数据集名称映射
    dataset_display_names = {
        'wit': 'WIT',
        'deep': 'Deep',
        'yt8m': 'YT-Audio'
    }
    
    algorithm_colors = {}
    algorithm_line_styles = {}
    algorithm_markers = {}
    for i, alg in enumerate(alg_set):
        algorithm_colors[alg] = colors[i % len(colors)]
        algorithm_line_styles[alg] = line_styles[i % len(line_styles)]
        algorithm_markers[alg] = markers[i % len(markers)]  
    
    # 创建一个1行6列的大图，使用更宽的比例
    fig = plt.figure(figsize=(30, 4.4))
    
    # 调整上下间距，给图例留出空间
    gs = GridSpec(1, 6, figure=fig, wspace=0.15, hspace=0.25, height_ratios=[1], top=0.80, bottom=0.1)
    # 定义每列的range fraction标签
    range_labels = [r"$2^{-2}$", r"$2^{-8}$"]
    # 遍历所有算法
    for idx_dataset, dataset in enumerate(datasets):
        for idx_range, range_val in enumerate(range_set):
            row = 0
            col = idx_dataset * 2 + idx_range
            ax_single = fig.add_subplot(gs[row, col])
            single_thread_y_data = []
            for alg in alg_set:
                key = (dataset, range_val)
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
                    filtered_df = pareto_df[(pareto_df['Recall'] >= 0.4)]
                    if not filtered_df.empty:
                        x_data = filtered_df['Recall'].tolist()
                        y_data = filtered_df['QPS'].tolist()
                        single_thread_y_data.append(y_data)
                        ax_single.plot(x_data, y_data, marker=marker, linestyle=linestyle, 
                                      label=alg, color=color, **plot_params)  # 使用alg作为标签
            
            # 添加range fraction标签到子图顶部
            ax_single.set_title(range_labels[idx_range], fontsize=16, fontweight='normal', pad=10)
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
            ax_single.set_xlim(0.4, 1.01)
            ax_single.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax_single.set_xticklabels([f"{tick:.1f}" for tick in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

            ax_single.axvline(x=0.95, color='gray', linestyle='--', alpha=0.7)
            ax_single.grid(True, linestyle=':', alpha=0.6)

            # 添加Recall@10标签
            # 设置x轴标签
            label_index = chr(97 + col)  # 97是ASCII码中'a'的值
            # 使用格式化后的数据集名称
            formatted_dataset = dataset_display_names[dataset]
            formatted_label = f"({label_index}) Recall@10 ({formatted_dataset})"
            ax_single.set_xlabel(formatted_label, 
                            fontproperties=libertine_font,
                            fontsize=20,
                            fontweight='bold')
            # 仅第一显示y轴标签
            if col == 0:
                ax_single.set_ylabel("QPS", fontsize=16)
            else:
                ax_single.set_ylabel("")

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
        leg = fig.legend(handles=legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, 0.999), ncol=min(15, len(legend_elements)),
                         fontsize=17, frameon=False)
        leg.get_title().set_fontweight('bold')
    
    # 调整布局，为底部的数据集标签留出空间
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig

print("加载数据文件...")
all_data = load_all_data()

if not all_data:
    print("未找到有效的数据文件，请检查路径和文件格式。")
else:
    total_files = sum(len(data_list) for data_list in all_data.values())
    print(f"成功加载数据，共有 {total_files} 个数据文件。")
    
    print("创建3x4对比图...")
    fig1 = plot_3_4_comparison(all_data)
    # save_path_svg = os.path.join("exp_8_1.svg")
    save_path_svg = os.path.join("/data/plots/exp","exp_8_1.svg")
    save_path_pdf = os.path.join("/data/plots/exp","exp_8_1.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    # plt.close(fig1)
    
    print("创建1x6对比图...")
    fig2 = plot_1_6_comparison(all_data)
    # save_path_svg = os.path.join("exp_8_2.svg")
    save_path_svg = os.path.join("/data/plots/exp","exp_8_2.svg")
    save_path_pdf = os.path.join("/data/plots/exp","exp_8_2.pdf")
    plt.savefig(save_path_svg, format="svg", bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    # plt.show()
    
    print("图表已创建并保存！")
