import os
import pandas as pd

# 获取当前目录下所有的文件
for filename in os.listdir('.'):
    # 排除 Python 脚本文件 (.py) 和没有.csv后缀的文件
    if filename.endswith('.py'):
        continue

    try:
        # 读取CSV文件
        df = pd.read_csv(filename, encoding='utf-8-sig')

        # 去除列名中的多余空格
        df.columns = df.columns.str.strip()

        # 打印列名确认
        print(f"正在处理文件: {filename}")
        print(f"列名: {df.columns}")

        # 只保留后两列
        df_filtered = df[['QPS', 'Recall@k10']]

        # 构建新的文件名
        new_filename = f'{filename}_faiss-ivf.csv'

        # 保存结果到新的CSV文件
        df_filtered.to_csv(new_filename, index=False)

        # 删除原始文件
        os.remove(filename)

        print(f"{filename} 已处理，结果已保存到 {new_filename}")
    
    except Exception as e:
        print(f"处理 {filename} 时出错: {e}")
