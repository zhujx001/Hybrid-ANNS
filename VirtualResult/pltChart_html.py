import os
import pandas as pd
import numpy as np
import glob
import re
import colorsys
from pyecharts import options as opts
from pyecharts.charts import Line, Page
from pyecharts.commons.utils import JsCode
import warnings

warnings.filterwarnings("ignore")

# 定义数据路径
data_path = "/data/result"

# 预定义一组美观且具有高对比度的基础颜色
BASE_COLORS = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf',  # 青色
    '#1a55FF',  # 亮蓝色
    '#FF4444',  # 亮红色
    '#47D147',  # 亮绿色
    '#AA44FF',  # 亮紫色
    '#FF9933',  # 亮橙色
    '#33CCFF',  # 天蓝色
]

def generate_distinct_colors(n):
    """
    生成n个具有高对比度的不同颜色
    
    策略：
    1. 首先使用预定义的基础颜色
    2. 如果需要更多颜色，则通过调整色相、饱和度和明度生成
    3. 确保相邻颜色有足够的区分度
    """
    colors = BASE_COLORS.copy()
    
    if n <= len(colors):
        return colors[:n]
    
    # 需要生成额外的颜色
    additional_colors_needed = n - len(colors)
    
    # 使用黄金分割比来生成分散的色相值
    golden_ratio = 0.618033988749895
    hue = 0.
    
    for i in range(additional_colors_needed):
        hue = (hue + golden_ratio) % 1.0
        # 使用较高的饱和度和明度以确保颜色鲜艳
        saturation = 0.7 + (i % 3) * 0.1  # 在0.7-0.9之间变化
        value = 0.8 + (i % 2) * 0.1       # 在0.8-0.9之间变化
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # 转换为16进制颜色代码
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

# 生成100个不同的颜色
colors = generate_distinct_colors(100)

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

# 获取所有结果文件
def get_all_result_files():
    # 新路径格式为 /data/result/{算法名}/result/{数据集名}_{查询集编号}_results.csv
    result_files = []
    # 获取所有算法目录
    algorithm_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for alg in algorithm_dirs:
        # 构建算法的result目录路径
        alg_result_path = os.path.join(data_path, alg, "result")
        if os.path.exists(alg_result_path):
            # 获取该算法下的所有结果文件
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
                
                # 检查必要的列是否存在（至少需要Recall和QPS列）
                required_columns = ['Recall', 'QPS']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: File {file} missing required columns. Skipping.")
                    continue
                
                # 添加算法名列
                df['Algorithm'] = algorithm
                
                key = (dataset, queryset)
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return all_data

# 计算帕累托前沿
def compute_pareto_frontier(df, x_col, y_col, maximize_x=True, maximize_y=True):
    """
    计算帕累托前沿
    
    参数:
    df (DataFrame): 包含数据的DataFrame
    x_col (str): x轴的列名
    y_col (str): y轴的列名
    maximize_x (bool): 是否最大化x
    maximize_y (bool): 是否最大化y
    
    返回:
    DataFrame: 帕累托前沿点的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # 获取x和y的numpy数组
    x = df[x_col].values
    y = df[y_col].values
    
    # 对于QPS vs Recall，我们希望y (QPS)越高越好，所以要排序时考虑-y
    sorted_indices = sorted(range(len(y)), key=lambda k: -y[k])
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    pareto_front_x = [x_sorted[0]]
    pareto_front_y = [y_sorted[0]]
    
    for i in range(1, len(x_sorted)):
        if x_sorted[i] > pareto_front_x[-1]:
            pareto_front_x.append(x_sorted[i])
            pareto_front_y.append(y_sorted[i])
    
    # 创建帕累托前沿的DataFrame
    indices = []
    for px, py in zip(pareto_front_x, pareto_front_y):
        # 找到原始DataFrame中与当前点匹配的行
        matches = df[(df[x_col] == px) & (df[y_col] == py)].index
        if not matches.empty:
            indices.append(matches[0])
    
    # 返回帕累托前沿点的DataFrame
    if indices:
        return df.loc[indices].sort_values(by=x_col, ascending=True)
    else:
        return pd.DataFrame()

# 使用pyecharts绘制QPS vs Recall图（使用帕累托前沿）
def plot_qps_vs_recall(data_list, title, save_path, width="900px", height="600px"):
    # 获取所有算法
    all_algorithms = set()
    for df in data_list:
        if 'Algorithm' in df.columns:
            all_algorithms.add(df['Algorithm'].iloc[0])
    
    # 分组算法
    alg_16bit, alg_other = group_algorithms(all_algorithms)
    
    # 如果两组都有算法，则绘制两张图
    if alg_16bit and alg_other:
        # 绘制16位算法图
        line_16bit = (
            Line(init_opts=opts.InitOpts(width=width, height=height))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{title} (16-bit algorithms)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        for i, df in enumerate([d for d in data_list if d['Algorithm'].iloc[0] in alg_16bit]):
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            
            if not pareto_df.empty:
                x_data = pareto_df['Recall'].tolist()
                y_data = pareto_df['QPS'].tolist()
                
                line_16bit.add_xaxis(x_data)
                line_16bit.add_yaxis(
                    series_name=algorithm,
                    y_axis=y_data,
                    symbol_size=8,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                )
        
        # 保存16位算法图
        save_path_16 = save_path.replace('.html', '_16bit.html')
        os.makedirs(os.path.dirname(save_path_16), exist_ok=True)
        line_16bit.render(save_path_16)
        
        # 绘制其他算法图
        line_other = (
            Line(init_opts=opts.InitOpts(width=width, height=height))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{title} (other algorithms)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        for i, df in enumerate([d for d in data_list if d['Algorithm'].iloc[0] in alg_other]):
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            
            if not pareto_df.empty:
                x_data = pareto_df['Recall'].tolist()
                y_data = pareto_df['QPS'].tolist()
                
                line_other.add_xaxis(x_data)
                line_other.add_yaxis(
                    series_name=algorithm,
                    y_axis=y_data,
                    symbol_size=8,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                )
        
        # 保存其他算法图
        save_path_other = save_path.replace('.html', '_other.html')
        os.makedirs(os.path.dirname(save_path_other), exist_ok=True)
        line_other.render(save_path_other)
    
    else:
        # 如果只有一组算法，则绘制单张图
        line = (
            Line(init_opts=opts.InitOpts(width=width, height=height))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        for i, df in enumerate(data_list):
            algorithm = df['Algorithm'].iloc[0]
            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
            
            if not pareto_df.empty:
                x_data = pareto_df['Recall'].tolist()
                y_data = pareto_df['QPS'].tolist()
                
                line.add_xaxis(x_data)
                line.add_yaxis(
                    series_name=algorithm,
                    y_axis=y_data,
                    symbol_size=8,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=2),
                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line.render(save_path)

# 创建输出目录
def create_output_dirs():
    dirs = [
        "plots/1_single_label",
        "plots/2_multi_label_effect",
        "plots/3_label_distribution",
        "plots/4_multi_label_performance",
        "plots/5_selectivity",
        "plots/6_dataset_effect",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs


# 1. 简单单标签搜索实验比较
def plot_single_label_search(all_data, output_dir):
    query_set = "1"
    
    # 对每个数据集，将不同算法的结果绘制在同一图中
    for dataset in set(k[0] for k in all_data.keys()):
        key = (dataset, query_set)
        if key in all_data:
            title = f"{dataset} - Comparison of different algorithms for single label search - query set {query_set}"
            save_path = os.path.join(output_dir, f"{dataset}_queryset_{query_set}_algorithms_comparison.html")
            plot_qps_vs_recall(all_data[key], title, save_path)

# 2. 多标签索引构建对单标签搜索的影响
def plot_multi_label_effect(all_data, output_dir):
    query_sets = ["1", "2_1", "2_2"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 1. 将所有查询集和所有算法绘制在一张大图上
        line_all = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{dataset} - The impact of multi-label index construction on single-label search"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分查询集
                    linestyles = ["solid", "dashed", "dotted"]
                    linestyle = linestyles[query_set_idx % len(linestyles)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        x_data = pareto_df['Recall'].tolist()
                        y_data = pareto_df['QPS'].tolist()
                        
                        # 只需添加一次x轴数据
                        if not added_xaxis:
                            line_all.add_xaxis(x_data)
                            added_xaxis = True
                        
                        # 添加数据系列
                        line_all.add_yaxis(
                            series_name=f"{algorithm} - query set {query_set}",
                            y_axis=y_data,
                            symbol_size=8,
                            label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(
                                width=2, 
                                type_=linestyle
                            ),
                            itemstyle_opts=opts.ItemStyleOpts(
                                color=colors[alg_idx % len(colors)]
                            ),
                            is_connect_nones=True,
                        )
        
        save_path = os.path.join(output_dir, f"{dataset}_all_querysets_all_algorithms.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_all.render(save_path)
        
        # 2. 为每个算法绘制一张图，比较其在三个查询集上的表现
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        for alg in algorithms:
            line_alg = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{dataset} - {alg} - Comparison of three query sets (1/2_1/2_2)"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(orient="horizontal", pos_top="5%"),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_alg.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_alg.add_yaxis(
                                    series_name=f"query set {query_set}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_querysets_comparison.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            line_alg.render(save_path)

# 2. 多标签索引构建对单标签搜索的影响
def plot_multi_label_effect(all_data, output_dir):
    query_sets = ["1", "2_1", "2_2"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 1. 将所有查询集和所有算法绘制在一张大图上
        line_all = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{dataset} - The impact of multi-label index construction on single-label search"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分查询集
                    linestyles = ["solid", "dashed", "dotted"]
                    linestyle = linestyles[query_set_idx % len(linestyles)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        x_data = pareto_df['Recall'].tolist()
                        y_data = pareto_df['QPS'].tolist()
                        
                        # 只需添加一次x轴数据
                        if not added_xaxis:
                            line_all.add_xaxis(x_data)
                            added_xaxis = True
                        
                        # 添加数据系列
                        line_all.add_yaxis(
                            series_name=f"{algorithm} - query set {query_set}",
                            y_axis=y_data,
                            symbol_size=8,
                            label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(
                                width=2, 
                                type_=linestyle
                            ),
                            itemstyle_opts=opts.ItemStyleOpts(
                                color=colors[alg_idx % len(colors)]
                            ),
                            is_connect_nones=True,
                        )
        
        save_path = os.path.join(output_dir, f"{dataset}_all_querysets_all_algorithms.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_all.render(save_path)
        
        # 2. 为每个算法绘制一张图，比较其在三个查询集上的表现
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        for alg in algorithms:
            line_alg = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{dataset} - {alg} - Comparison of three query sets (1/2_1/2_2)"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(orient="horizontal", pos_top="5%"),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_alg.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_alg.add_yaxis(
                                    series_name=f"query set {query_set}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_querysets_comparison.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            line_alg.render(save_path)

# 3. 标签分布对算法的影响
def plot_label_distribution_effect(all_data, output_dir):
    query_sets = ["5_1", "5_2", "5_3", "5_4"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 同一算法在不同标签分布下的表现
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        # 1. 所有标签分布和所有算法组合在一张大图上
        line_all = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{dataset} - The impact of label distribution on algorithms"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分标签分布
                    linestyles = ["solid", "dashed", "dotted", "dashdot"]
                    linestyle = linestyles[query_set_idx % len(linestyles)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        x_data = pareto_df['Recall'].tolist()
                        y_data = pareto_df['QPS'].tolist()
                        
                        # 只需添加一次x轴数据
                        if not added_xaxis:
                            line_all.add_xaxis(x_data)
                            added_xaxis = True
                        
                        # 添加数据系列
                        line_all.add_yaxis(
                            series_name=f"{algorithm} - label distribution {query_set}",
                            y_axis=y_data,
                            symbol_size=8,
                            label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(
                                width=2, 
                                type_=linestyle
                            ),
                            itemstyle_opts=opts.ItemStyleOpts(
                                color=colors[alg_idx % len(colors)]
                            ),
                            is_connect_nones=True,
                        )
        
        save_path = os.path.join(output_dir, f"{dataset}_all_distributions_all_algorithms.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_all.render(save_path)
        
        # 2. 为每个算法创建对比图，显示它在不同标签分布下的表现
        for alg in algorithms:
            line_alg = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{dataset} - {alg} - Comparison of different label distributions (5_1 to 5_4)"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(orient="horizontal", pos_top="5%"),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_alg.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_alg.add_yaxis(
                                    series_name=f"label distribution {query_set}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_label_distributions_comparison.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            line_alg.render(save_path)
        
        # 3. 同一标签分布下不同算法的表现对比
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                title = f"{dataset} - Different algorithms in label distribution {query_set}"
                save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_distribution_{query_set}.html")
                plot_qps_vs_recall(all_data[key], title, save_path)

# 4. 多标签算法表现
def plot_multi_label_performance(all_data, output_dir):
    query_sets = ["6", "7_2"]
    
    # 获取所有数据集
    datasets = set(k[0] for k in all_data.keys())
    
    for dataset in datasets:
        # 1. 所有查询集和所有算法组合在一张大图上
        line_all = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{dataset} - Multi-label algorithm performance"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for query_set_idx, query_set in enumerate(query_sets):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分查询集
                    linestyles = ["solid", "dashed"]
                    linestyle = linestyles[query_set_idx % len(linestyles)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        x_data = pareto_df['Recall'].tolist()
                        y_data = pareto_df['QPS'].tolist()
                        
                        # 只需添加一次x轴数据
                        if not added_xaxis:
                            line_all.add_xaxis(x_data)
                            added_xaxis = True
                        
                        # 添加数据系列
                        line_all.add_yaxis(
                            series_name=f"{algorithm} - query set {query_set}",
                            y_axis=y_data,
                            symbol_size=8,
                            label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(
                                width=2, 
                                type_=linestyle
                            ),
                            itemstyle_opts=opts.ItemStyleOpts(
                                color=colors[alg_idx % len(colors)]
                            ),
                            is_connect_nones=True,
                        )
        
        save_path = os.path.join(output_dir, f"{dataset}_all_querysets_all_algorithms_multilabel.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_all.render(save_path)
        
        # 2. 不同算法在查询集6上的表现对比
        key = (dataset, "6")
        if key in all_data:
            title = f"{dataset} - Comparison of different algorithms for multi-label search - query set 6"
            save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_queryset_6.html")
            plot_qps_vs_recall(all_data[key], title, save_path)
        
        # 3. 同一算法在查询集6和7_2上的表现对比
        algorithms = set()
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        for alg in algorithms:
            line_alg = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{dataset} - {alg} - Comparison of query set 6 and 7_2"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(orient="horizontal", pos_top="5%"),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            
            for i, query_set in enumerate(query_sets):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_alg.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_alg.add_yaxis(
                                    series_name=f"query set {query_set}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_querysets_6_7_2_comparison.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            line_alg.render(save_path)
        
        # 4. 不同算法在查询集7_2下的表现对比
        key = (dataset, "7_2")
        if key in all_data:
            title = f"{dataset} - Comparison of different algorithms for multi-label search - query set 7_2"
            save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_queryset_7_2.html")
            plot_qps_vs_recall(all_data[key], title, save_path)

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
        # 1. 单标签：所有选择性和所有算法组合在一张大图上
        line_single = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{dataset} - Single label search selectivity experiment"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for query_set_idx, query_set in enumerate(single_label_selectivity):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分选择性
                    linestyles = ["solid", "dashed", "dotted", "dashdot"]
                    linestyle = linestyles[query_set_idx % len(linestyles)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        x_data = pareto_df['Recall'].tolist()
                        y_data = pareto_df['QPS'].tolist()
                        
                        # 只需添加一次x轴数据
                        if not added_xaxis:
                            line_single.add_xaxis(x_data)
                            added_xaxis = True
                        
                        # 添加数据系列
                        line_single.add_yaxis(
                            series_name=f"{algorithm} - selectivity {selectivity_mapping[query_set]}",
                            y_axis=y_data,
                            symbol_size=8,
                            label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(
                                width=2, 
                                type_=linestyle
                            ),
                            itemstyle_opts=opts.ItemStyleOpts(
                                color=colors[alg_idx % len(colors)]
                            ),
                            is_connect_nones=True,
                        )
        
        save_path = os.path.join(output_dir, f"{dataset}_all_single_selectivity_all_algorithms.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_single.render(save_path)
        
        # 2. 多标签：所有选择性和所有算法组合在一张大图上
        line_multi = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{dataset} - Multi-label search selectivity experiment"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for query_set_idx, query_set in enumerate(multi_label_selectivity):
            key = (dataset, query_set)
            if key in all_data:
                for alg_idx, df in enumerate(all_data[key]):
                    algorithm = df['Algorithm'].iloc[0]
                    # 使用不同的线型区分选择性
                    linestyles = ["solid", "dashed", "dotted", "dashdot"]
                    linestyle = linestyles[query_set_idx % len(linestyles)]
                    
                    # 计算帕累托前沿
                    pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                    
                    if not pareto_df.empty:
                        x_data = pareto_df['Recall'].tolist()
                        y_data = pareto_df['QPS'].tolist()
                        
                        # 只需添加一次x轴数据
                        if not added_xaxis:
                            line_multi.add_xaxis(x_data)
                            added_xaxis = True
                        
                        # 添加数据系列
                        line_multi.add_yaxis(
                            series_name=f"{algorithm} - selectivity {selectivity_mapping[query_set]}",
                            y_axis=y_data,
                            symbol_size=8,
                            label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(
                                width=2, 
                                type_=linestyle
                            ),
                            itemstyle_opts=opts.ItemStyleOpts(
                                color=colors[alg_idx % len(colors)]
                            ),
                            is_connect_nones=True,
                        )
        
        save_path = os.path.join(output_dir, f"{dataset}_all_multi_selectivity_all_algorithms.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_multi.render(save_path)
        
        # 3. 不同算法在相同选择性下的表现对比
        # 单标签选择性
        for query_set in single_label_selectivity:
            key = (dataset, query_set)
            if key in all_data:
                title = f"{dataset} - Comparison of different algorithms for single label search - selectivity {selectivity_mapping[query_set]}"
                save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_single_label_selectivity_{query_set}.html")
                plot_qps_vs_recall(all_data[key], title, save_path)
        
        # 多标签选择性
        for query_set in multi_label_selectivity:
            key = (dataset, query_set)
            if key in all_data:
                title = f"{dataset} - Comparison of different algorithms for multi-label search - selectivity {selectivity_mapping[query_set]}"
                save_path = os.path.join(output_dir, f"{dataset}_algorithms_comparison_multi_label_selectivity_{query_set}.html")
                plot_qps_vs_recall(all_data[key], title, save_path)
        
        # 4. 同一算法在不同选择性下的表现
        algorithms = set()
        for query_set in single_label_selectivity + multi_label_selectivity:
            key = (dataset, query_set)
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns:
                        algorithms.add(df['Algorithm'].iloc[0])
        
        # 为每个算法创建单标签选择性对比图
        for alg in algorithms:
            # 单标签选择性
            line_alg_single = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{dataset} - {alg} - Comparison of single label selectivity (1%/25%/50%/75%)"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(orient="horizontal", pos_top="5%"),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            
            for i, query_set in enumerate(single_label_selectivity):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_alg_single.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_alg_single.add_yaxis(
                                    series_name=f"selectivity {selectivity_mapping[query_set]}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_single_label_selectivity_comparison.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            line_alg_single.render(save_path)
            
            # 多标签选择性
            line_alg_multi = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{dataset} - {alg} - Comparison of multi-label selectivity (1%/25%/50%/75%)"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(orient="horizontal", pos_top="5%"),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            
            for i, query_set in enumerate(multi_label_selectivity):
                key = (dataset, query_set)
                if key in all_data:
                    # 查找该算法的数据
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_alg_multi.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_alg_multi.add_yaxis(
                                    series_name=f"selectivity {selectivity_mapping[query_set]}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[i % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            save_path = os.path.join(output_dir, f"{dataset}_{alg}_multi_label_selectivity_comparison.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            line_alg_multi.render(save_path)

# 6. 数据集对算法的影响
def plot_dataset_effect(all_data, output_dir):
    # 获取所有数据集和查询集
    datasets = set(k[0] for k in all_data.keys())
    query_sets = ["1", "6"]  # 单标签(1)和多标签(6)查询集
    
    # 获取所有算法
    algorithms = set()
    for key in all_data:
        for df in all_data[key]:
            if 'Algorithm' in df.columns:
                algorithms.add(df['Algorithm'].iloc[0])
    
    # 1. 对于每个算法比较其在不同数据集上的单标签搜索表现
    for alg in algorithms:
        # 单标签搜索
        line_single = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{alg} - Performance on different datasets - Single label search"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for dataset_idx, dataset in enumerate(datasets):
            key = (dataset, "1")  # 单标签查询集
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                        # 计算帕累托前沿
                        pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                        
                        if not pareto_df.empty:
                            x_data = pareto_df['Recall'].tolist()
                            y_data = pareto_df['QPS'].tolist()
                            
                            # 只需添加一次x轴数据
                            if not added_xaxis:
                                line_single.add_xaxis(x_data)
                                added_xaxis = True
                            
                            # 添加数据系列
                            line_single.add_yaxis(
                                series_name=f"dataset {dataset}",
                                y_axis=y_data,
                                symbol_size=8,
                                label_opts=opts.LabelOpts(is_show=False),
                                linestyle_opts=opts.LineStyleOpts(width=2),
                                itemstyle_opts=opts.ItemStyleOpts(color=colors[dataset_idx % len(colors)]),
                                is_connect_nones=True,
                            )
        
        save_path = os.path.join(output_dir, f"{alg}_dataset_comparison_single_label.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_single.render(save_path)
        
        # 多标签搜索
        line_multi = (
            Line(init_opts=opts.InitOpts(width="1200px", height="800px"))
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{alg} - Performance on different datasets - Multi-label search"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="Recall",
                    name_location="center",
                    name_gap=35,
                    min_=0,
                    max_=1,
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value", 
                    name="QPS",
                    name_location="center",
                    name_gap=35,
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_right="0%", 
                    pos_top="10%",
                    border_width=1,
                    border_color="#ccc",
                    padding=10,
                    item_gap=10,
                    item_width=25,
                    item_height=14,
                    background_color="rgba(255,255,255,0.8)",  # 半透明背景
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        
        # 用于跟踪已添加的x轴数据
        added_xaxis = False
        
        for dataset_idx, dataset in enumerate(datasets):
            key = (dataset, "6")  # 多标签查询集
            if key in all_data:
                for df in all_data[key]:
                    if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                        # 计算帕累托前沿
                        pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                        
                        if not pareto_df.empty:
                            x_data = pareto_df['Recall'].tolist()
                            y_data = pareto_df['QPS'].tolist()
                            
                            # 只需添加一次x轴数据
                            if not added_xaxis:
                                line_multi.add_xaxis(x_data)
                                added_xaxis = True
                            
                            # 添加数据系列
                            line_multi.add_yaxis(
                                series_name=f"dataset {dataset}",
                                y_axis=y_data,
                                symbol_size=8,
                                label_opts=opts.LabelOpts(is_show=False),
                                linestyle_opts=opts.LineStyleOpts(width=2),
                                itemstyle_opts=opts.ItemStyleOpts(color=colors[dataset_idx % len(colors)]),
                                is_connect_nones=True,
                            )
        
        save_path = os.path.join(output_dir, f"{alg}_dataset_comparison_multi_label.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        line_multi.render(save_path)
    
    # 2. 对于每个数据集和查询集，比较不同算法的表现
    for dataset in datasets:
        for query_set in query_sets:
            key = (dataset, query_set)
            if key in all_data:
                if query_set == "1":
                    title = f"Different algorithms on {dataset} - Single label search"
                else:
                    title = f"Different algorithms on {dataset} - Multi-label search"
                
                save_path = os.path.join(output_dir, f"{dataset}_queryset_{query_set}_algorithms_comparison.html")
                plot_qps_vs_recall(all_data[key], title, save_path)
    
    # 3. 综合比较图：矩阵形式展示所有数据集+查询集组合
    for query_type, query_set_id in [("single_label", "1"), ("multi_label", "6")]:
        # 为每个查询类型创建一个大的比较图
        all_page = Page(page_title=f"Algorithm performance across all datasets - {query_type}")
        
        for alg in algorithms:
            line_comprehensive = (
                Line(init_opts=opts.InitOpts(width="900px", height="600px"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{alg} - Performance on all datasets - {query_type}"
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="Recall",
                        name_location="center",
                        name_gap=35,
                        min_=0,
                        max_=1,
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value", 
                        name="QPS",
                        name_location="center",
                        name_gap=35,
                    ),
                    legend_opts=opts.LegendOpts(
                        orient="vertical", 
                        pos_right="0%", 
                        pos_top="10%",
                        border_width=1,
                        border_color="#ccc",
                        padding=10,
                        item_gap=10,
                        item_width=25,
                        item_height=14,
                        background_color="rgba(255,255,255,0.8)",  # 半透明背景
                    ),
                    datazoom_opts=[
                        opts.DataZoomOpts(range_start=0, range_end=100),
                        opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                    ],
                )
            )
            
            # 用于跟踪已添加的x轴数据
            added_xaxis = False
            has_data = False
            
            for dataset_idx, dataset in enumerate(datasets):
                key = (dataset, query_set_id)
                if key in all_data:
                    for df in all_data[key]:
                        if 'Algorithm' in df.columns and df['Algorithm'].iloc[0] == alg:
                            # 计算帕累托前沿
                            pareto_df = compute_pareto_frontier(df, 'Recall', 'QPS', maximize_x=True, maximize_y=True)
                            
                            if not pareto_df.empty:
                                has_data = True
                                x_data = pareto_df['Recall'].tolist()
                                y_data = pareto_df['QPS'].tolist()
                                
                                # 只需添加一次x轴数据
                                if not added_xaxis:
                                    line_comprehensive.add_xaxis(x_data)
                                    added_xaxis = True
                                
                                # 添加数据系列
                                line_comprehensive.add_yaxis(
                                    series_name=f"dataset {dataset}",
                                    y_axis=y_data,
                                    symbol_size=8,
                                    label_opts=opts.LabelOpts(is_show=False),
                                    linestyle_opts=opts.LineStyleOpts(width=2),
                                    itemstyle_opts=opts.ItemStyleOpts(color=colors[dataset_idx % len(colors)]),
                                    is_connect_nones=True,
                                )
            
            # 只有当有数据时才添加到页面
            if has_data:
                all_page.add(line_comprehensive)
        
        # 保存综合比较页面
        save_path = os.path.join(output_dir, f"all_datasets_comparison_{query_type}.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        all_page.render(save_path)

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
    
    # 运行各个实验的图表绘制
    print("1. 绘制简单单标签搜索实验比较图...")
    plot_single_label_search(all_data, output_dirs[0])
    
    print("2. 绘制多标签索引构建对单标签搜索的影响图...")
    plot_multi_label_effect(all_data, output_dirs[1])
    
    print("3. 绘制标签分布对算法的影响图...")
    plot_label_distribution_effect(all_data, output_dirs[2])
    
    print("4. 绘制多标签算法表现图...")
    plot_multi_label_performance(all_data, output_dirs[3])
    
    print("5. 绘制选择性实验比较图...")
    plot_selectivity_experiments(all_data, output_dirs[4])
    
    print("6. 绘制数据集对算法的影响图...")
    plot_dataset_effect(all_data, output_dirs[5])
    
   
    
    print("所有图表绘制完成！")
if __name__ == "__main__":
    main()
