import matplotlib.pyplot as plt
import wandb
import numpy as np
import seaborn as sns
import re

# 设置全局样式
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 14

api = wandb.Api()

# 定义要使用的run名称（可以增加？）
run_names = [
    # "2（Test）rel_rm_step_302_f1_0.670_acc_0.670",
    # "2（Test）rel_rm_step_602_f1_0.702_acc_0.692",
    # "1（Test）fact_rm_step_728_f1_0.647_acc_0.754",
    # "1（Test）fact_rm_step_926_f1_0.661_acc_0.773",
    "3（Test）comp_rm_step_5130_acc_0.683",
    "3（Test）comp_rm_step_1230_acc_0.690"
]

# 定义要绘制的数据键
keys_to_plot = {
    'reward_raw': "train/train_rm/reward/raw",
    'reward_KL': "train/train_rm/reward/KL",
    'reward_penalized': "train/train_rm/reward/penalized"
}

degree = 3  # 多项式拟合的阶数

colors = sns.color_palette("Set2", len(run_names))  # 根据run数量生成颜色


def extract_info_from_name(name):
    """
    从 run 名称中提取奖励模型的类型、f1 分数和准确度
    """
    if "comp" in name:
        match = re.search(r"acc_(\d+\.\d+)", name)
        if match:
            acc = match.group(1)
            return f"Comp-RM_acc_{acc}"
        else:
            return name

    else:
        match = re.search(r"f1_(\d+\.\d+)_acc_(\d+\.\d+)", name)
        if match:
            f1_score = match.group(1)
            acc = match.group(2)
            return f"Fact-RM_f1_{f1_score}_acc_{acc}"
        else:
            return name

def plot_performance(metric_key, metric_label, runs):
    fig, ax1 = plt.subplots()

    for run, color in zip(runs, colors):
        steps = []
        values = []

        history = run.history(keys=[metric_key, 'train/step'])

        if not history.empty and metric_key in history and 'train/step' in history:
            step_values = history['train/step'].values
            metric_values = history[metric_key].values

            steps.extend(step_values)
            values.extend(metric_values)

        if steps and values:
            sorted_data = sorted(zip(steps, values), key=lambda x: x[0])
            sorted_steps, sorted_values = zip(*sorted_data)

            # 提取信息并修改图例部分
            extracted_info = extract_info_from_name(run.name)

            # 绘制点
            ax1.scatter(sorted_steps, sorted_values, label=extracted_info, s=50, color=color)

    ax1.set_xlabel('Train Step')
    ax1.set_ylabel(f'{metric_label.replace("/", "-").title()}')
    ax1.grid(True)
    plt.legend(loc='upper right')
    plt.title(f'RLHF Training of T5-small on Relevance Task')
    plt.savefig(f"/home/llm/FineGrainedRLHF/Pictures/2D-a/{extracted_info.split('-')[0]}-{metric_label}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


# 获取指定的run，通过查询所有run并匹配名称来找到对应的run对象
project = "T5-small_RM_research_StepTest_StepOnly"
entity = "battam"

# 使用项目和实体名获取所有run，然后筛选出指定的run
all_runs = api.runs(f"{entity}/{project}")
runs = [run for run in all_runs if any(run_name in run.name for run_name in run_names)]

# 分别绘制每个指标的性能图表
for metric_label, metric_key in keys_to_plot.items():
    plot_performance(metric_key, metric_label, runs)
