import numpy as np
import pandas as pd

def fit_curve(x, y, degree=2, plot=False):
    x = np.array(x)
    y = np.array(y)
    mask = x > 0.8  # 规定recall大于0.8的点才参与拟合
    x = x[mask]
    y = y[mask]
    degree = min(degree, len(x)-1)
    # 用numpy的polyfit做多项式拟合，返回多项式系数（从高到低）
    coeffs = np.polyfit(x, y, degree)
    # 根据多项式系数构造一个多项式函数对象，可以像普通函数一样调用
    fit_func = np.poly1d(coeffs)
    return fit_func, coeffs

def calculate_cv_for_files(files, algorithm_name):
    """计算给定文件列表的CV均值"""
    import os
    
    # 存储每个文件的拟合函数和处理后的数据范围
    fit_functions = []
    data_ranges = []
    
    print(f"处理算法: {algorithm_name}")
    
    # 对每个文件进行拟合
    for i, file_path in enumerate(files):
        if not os.path.exists(file_path):
            print(f"  跳过文件 {i+1}: 文件不存在")
            continue
            
        print(f"  处理文件 {i+1}: {file_path.split('/')[-1]}")
        
        # 读取数据
        df = pd.read_csv(file_path)
        if 'Recall' not in df.columns or 'QPS' not in df.columns:
            print(f"    跳过: 缺少必要列")
            continue
            
        x_original = df['Recall'].to_numpy()
        y_original = df['QPS'].to_numpy()
        
        # 应用过滤条件（recall > 0.8）得到拟合用的数据
        mask = x_original > 0.8
        x_filtered = x_original[mask]
        y_filtered = y_original[mask]
        
        if len(x_filtered) < 2:
            print(f"    跳过: 过滤后数据点太少 ({len(x_filtered)})")
            continue
        
        # 拟合曲线
        fit_func, coeffs = fit_curve(x_original, y_original, degree=2)
        fit_functions.append(fit_func)
        
        # 记录过滤后数据的范围
        x_min, x_max = x_filtered.min(), x_filtered.max()
        data_ranges.append((x_min, x_max))
        print(f"    拟合完成，数据范围: [{x_min:.3f}, {x_max:.3f}]")
    
    if len(fit_functions) == 0:
        print(f"  {algorithm_name}: 没有成功处理任何文件")
        return None
    
    print(f"  成功处理了 {len(fit_functions)} 个文件")
    
    # 计算所有文件的公共范围（取交集）
    global_min = max(range_pair[0] for range_pair in data_ranges)
    global_max = min(range_pair[1] for range_pair in data_ranges)
    
    if global_min >= global_max:
        print(f"  {algorithm_name}: 文件之间没有公共的recall范围")
        return None
    
    # 生成5个测试点：最小值、最大值，以及中间平均分布的3个点
    recall_points = np.linspace(global_min, global_max, 5)
    print(f"  公共数据范围: [{global_min:.3f}, {global_max:.3f}]")
    
    # 计算每个recall点的方差
    cv_values = []
    
    for i, recall in enumerate(recall_points):
        qps_values = []
        
        # 计算每条拟合曲线在该recall点的QPS值
        for j, fit_func in enumerate(fit_functions):
            qps = fit_func(recall)
            qps_values.append(qps)
        
        # 计算统计量
        qps_array = np.array(qps_values)
        mean_qps = np.mean(qps_array)
        std_dev = np.std(qps_array)
        
        # 计算变异系数 CV = std / mean
        cv = std_dev / mean_qps if mean_qps != 0 else 0
        cv_values.append(cv)
    
    # 计算CV的均值
    mean_cv = np.mean(cv_values)
    print(f"  CV均值: {mean_cv:.4f}")
    print()
    
    return mean_cv

def calculate_all_algorithms_cv():
    """计算所有算法的CV均值"""
    algorithms = ['ACORN-1','ACORN-γ', 'CAPS', 'Faiss', 'Faiss+HQI_Batch', 'FilteredVamana', 
                 'Milvus', 'NHQ', 'Puck', 'StitchedVamana', 'UNG', 'VBASE']
    
    results = {}
    
    print("开始计算所有算法的CV均值...")
    print("=" * 60)
    
    for algorithm in algorithms:
        base_path = f"/data/result/{algorithm}/result"
        files = [
            f"{base_path}/sift_5_1_results.csv",
            f"{base_path}/sift_5_2_results.csv", 
            f"{base_path}/sift_5_3_results.csv",
            f"{base_path}/sift_5_4_results.csv"
        ]
        
        cv_mean = calculate_cv_for_files(files, algorithm)
        if cv_mean is not None:
            results[algorithm] = cv_mean
    
    print("=" * 60)
    print("所有算法CV均值汇总:")
    for algorithm, cv_mean in results.items():
        print(f"{algorithm:16}: {cv_mean:.4f}")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='单个数据文件（CSV，有表头）')
    parser.add_argument('--degree', type=int, default=2, help='拟合阶数')
    parser.add_argument('--plot', action='store_true', help='是否画拟合图')
    parser.add_argument('--variance', action='store_true', help='计算VBASE单算法方差模式')
    parser.add_argument('--all', action='store_true', help='计算所有算法的CV均值')
    args = parser.parse_args()

    if args.all:
        # 计算所有算法的CV均值
        calculate_all_algorithms_cv()
    elif args.variance:
        # 单算法方差计算模式（VBASE）
        base_path = "/data/result/VBASE/result"
        files = [
            f"{base_path}/sift_5_1_results.csv",
            f"{base_path}/sift_5_2_results.csv", 
            f"{base_path}/sift_5_3_results.csv",
            f"{base_path}/sift_5_4_results.csv"
        ]
        calculate_cv_for_files(files, "VBASE")
    else:
        # 原有的单文件处理模式
        if not args.file:
            print("错误：单文件模式需要指定 --file 参数")
            return
            
        # 用 pandas 读文件，只取 Recall 和 QPS 列
        df = pd.read_csv(args.file)
        if 'Recall' not in df.columns or 'QPS' not in df.columns:
            raise ValueError('CSV文件必须包含"Recall"和"QPS"列！')

        x = df['Recall'].to_numpy()
        y = df['QPS'].to_numpy()

        # 拟合
        fit_func, coeffs = fit_curve(x, y, degree=args.degree, plot=args.plot)
        print('拟合多项式:')
        print(fit_func)
        print('多项式系数:', coeffs)

        # 演示一下拟合效果
        for xi, yi in zip(x, y):
            print(f'Recall={xi:.4f}, QPS真实={yi:.2f}, QPS拟合={fit_func(xi):.2f}')

if __name__ == '__main__':
    main()

# python exp_3_1.py --all
