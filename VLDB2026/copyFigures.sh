#!/bin/bash
# 检查并复制 PDF 文件到 figures2 目录，保持路径结构

# 目标根目录
TARGET_DIR="figures2"

# 文件列表（可复制你贴的完整清单）
FILES=(
"hybridANNS.pdf"
"figures/exp/attribute_100M.pdf"
"figures/exp/attribute_71.pdf"
"figures/exp/attribute_85.pdf"
"figures/exp/attribute_legend.pdf"
"figures/exp/attribute_multimodel_1.pdf"
"figures/exp/attribute_multimodel.pdf"
"figures/exp/exp_1_1_SingleLabel_1thread.pdf"
"figures/exp/exp_2_1.pdf"
"figures/exp/exp_2_legend.pdf"
"figures/exp/exp_3_1.pdf"
"figures/exp/exp_4_1_MultiLabel_1thread.pdf"
"figures/exp/exp_5_2_1.pdf"
"figures/exp/exp_5_2_2.pdf"
"figures/exp/exp_5_2_3.pdf"
"figures/exp/exp_8_2.pdf"
"figures/exp/range_71.pdf"
"figures/exp/range_85.pdf"
"figures/exp/range_deep100M.pdf"
"figures/exp/range_legend.pdf"
"figures/exp/range_multimodel_1.pdf"
"figures/exp/range_multimodel.pdf"
"figures/graph.pdf"
"figures/indexData/exp_7_build_time_comparison_query1.pdf"
"figures/indexData/exp_7_index_size_mb_comparison_query1.pdf"
"figures/indexData/exp_7_memory_mb_comparison_query1.pdf"
"figures/indexData/legend_only.pdf"
"figures/indexData/rangeFilter_build_time_comparison_query.pdf"
"figures/indexData/rangeFilter_index_size_mb_comparison_query.pdf"
"figures/indexData/rangeFilter_legend_only.pdf"
"figures/indexData/rangeFilter_memory_mb_comparison_query.pdf"
"figures/ivf.pdf"
"figures/searchMem/label_memory_comparison.pdf"
"figures/searchMem/range_memory_comparison.pdf"
)

echo "检查并复制 PDF 文件到 $TARGET_DIR ..."

for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
        # 生成目标路径
        target_path="$TARGET_DIR/$file"
        mkdir -p "$(dirname "$target_path")"
        cp "$file" "$target_path"
        echo "✅ 已复制: $file → $target_path"
    else
        echo "❌ 缺失文件: $file"
    fi
done

echo "✅ 完成！"
