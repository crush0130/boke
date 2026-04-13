import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

# ================= 配置路径 =================
PDB_DIR = "/media/dell/新加卷/zym/foldingdiff/out_dir/sampled_pdb"  # 你的PDB文件夹路径
TM_SCORE_FILE = "/media/dell/新加卷/zym/foldingdiff/out_dir/tm_scores.json"            # 横坐标文件
SCTM_SCORE_FILE = "/media/dell/新加卷/zym/foldingdiff/out_dir/sctm_scores.json"        # 纵坐标文件
LENGTH_OUTPUT_FILE = "sample_lengths.json" # 生成的长度文件
OUTPUT_FIG = "fig4c_colored.png"           # 输出图片文件名
# ===========================================

# ---------- 1. 从PDB文件提取长度 ----------
def get_length_from_pdb(pdb_file):
    """统计PDB文件中CA原子的数量（每个残基一个CA）"""
    count = 0
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                count += 1
    return count

print("正在从PDB文件提取长度信息...")
pdb_files = list(Path(PDB_DIR).glob("*.pdb"))
length_dict = {}
for pdb in pdb_files:
    sample_id = pdb.stem  # 文件名（不含扩展名）
    try:
        length = get_length_from_pdb(pdb)
        length_dict[sample_id] = length
    except Exception as e:
        print(f"警告：处理 {pdb.name} 时出错：{e}")

# 保存长度信息
with open(LENGTH_OUTPUT_FILE, 'w') as f:
    json.dump(length_dict, f, indent=2)
print(f"长度信息已保存至 {LENGTH_OUTPUT_FILE}，共 {len(length_dict)} 个样本")

# ---------- 2. 读取TM分数和scTM分数 ----------
with open(TM_SCORE_FILE) as f:
    tm_scores = json.load(f)      # 横坐标
with open(SCTM_SCORE_FILE) as f:
    sctm_scores = json.load(f)    # 纵坐标

# ---------- 3. 合并数据 ----------
# 取三个文件共有的样本ID
common_ids = set(tm_scores.keys()) & set(sctm_scores.keys()) & set(length_dict.keys())
print(f"三个文件共有的样本数：{len(common_ids)}")

data = []
for sid in common_ids:
    data.append({
        "sample": sid,
        "max_train_tm": tm_scores[sid],
        "scTM": sctm_scores[sid],
        "length": length_dict[sid],
        "length_group": "short (≤70 aa)" if length_dict[sid] <= 70 else "long (>70 aa)"
    })

df = pd.DataFrame(data)
print(f"合并后数据框大小：{df.shape}")

# ---------- 4. 计算Spearman相关系数 ----------
rho, p_value = spearmanr(df["max_train_tm"], df["scTM"])
print(f"Spearman's ρ = {rho:.3f}, p = {p_value:.3e}")

# ---------- 5. 绘图 ----------
plt.figure(figsize=(6, 5))
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 散点图（按长度分组着色）
colors = {"short (≤70 aa)": "blue", "long (>70 aa)": "orange"}
sns.scatterplot(
    data=df,
    x="max_train_tm",
    y="scTM",
    hue="length_group",
    palette=colors,
    alpha=0.6,
    edgecolor=None,
    s=30
)


# 添加两条参考线（0.5）
plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)

# 坐标轴标签
plt.xlabel("Maximum training TM-score")
plt.ylabel("scTM score")
plt.legend(title="Protein length", loc='lower right')

# 在图上标注相关系数
plt.text(
    0.05, 0.95,
    f"Spearman's ρ = {rho:.3f}\np = {p_value:.3e}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"图片已保存为 {OUTPUT_FIG}")
