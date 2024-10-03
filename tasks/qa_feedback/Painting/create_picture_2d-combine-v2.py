import matplotlib.pyplot as plt
import wandb
import re
import numpy as np
import seaborn as sns

# 设置全局样式
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 14

api = wandb.Api()

project = "T5-small_RM_research_StepTest_StepOnly"
# project = "T5-base_RM_research_StepTest_StepOnly"
# project = "T5-large_RM_research_StepTest_StepOnly"
entity = "battam"  # 替换为您的实体名称

keys_prefix_mapping = {
    "eval_rm": {
        "1": "eval/eval_rm/factuality_ratios",
        "2": "eval/eval_rm/relevance_ratios",
        "3": "eval/eval_rm/completeness_rewards"
    },
    "train_rm": {
        "1": "eval/train_rm/factuality_ratios",
        "2": "eval/train_rm/relevance_ratios",
        "3": "eval/train_rm/completeness_rewards"
    }
}

keys_task_mapping = {
    "1": "factuality_ratios",
    "2": "relevance_ratios",
    "3": "completeness_rewards"
}

degree = 3  # 多项式拟合的阶数

# 使用两种不同颜色来区分 EvalRM 和 TrainRM
color_eval = sns.color_palette("Set2", 2)[0]  # 第一个颜色用于 EvalRM
color_train = sns.color_palette("Set2", 2)[1]  # 第二个颜色用于 TrainRM

def plot_performance(metric_name, runs):
    fig, ax1 = plt.subplots()

    # 绘制 EvalRM 和 TrainRM 的数据
    for key_type, key_mapping in keys_prefix_mapping.items():
        color = color_eval if key_type == "eval_rm" else color_train  # 根据数据类型选择颜色

        for prefix, key in key_mapping.items():
            if metric_name not in key:
                continue

            accuracies = []
            values = []

            for run in runs:
                if run.name.startswith(prefix):
                    accuracy_match = re.search(r'acc_([0-9]+\.[0-9]+)', run.name)

                    if accuracy_match:
                        accuracy = float(accuracy_match.group(1))
                        history = run.history(keys=[key])
                        
                        if not history.empty and key in history:
                            value = history[key].max()
                            accuracies.append(accuracy)
                            values.append(value)

            if accuracies and values:
                sorted_data = sorted(zip(accuracies, values), key=lambda x: x[0])
                sorted_accuracies, sorted_values = zip(*sorted_data)

                label = f"{key.split('/')[-1]} ({key_type})"
                # 多项式拟合
                coefficients = np.polyfit(sorted_accuracies, sorted_values, degree)
                polynomial = np.poly1d(coefficients)
                x_fit = np.linspace(min(sorted_accuracies), max(sorted_accuracies), 100)
                y_fit = polynomial(x_fit)

                # 计算残差
                y_pred = polynomial(sorted_accuracies)
                residuals = np.abs(np.array(sorted_values) - y_pred)
                threshold = np.mean(residuals) + 2 * np.std(residuals)

                # 标记噪音点
                noisy_accuracies = [acc for acc, residual in zip(sorted_accuracies, residuals) if residual > threshold]
                noisy_values = [value for value, residual in zip(sorted_values, residuals) if residual > threshold]

                # 标记正常点
                normal_accuracies = [acc for acc, residual in zip(sorted_accuracies, residuals) if residual <= threshold]
                normal_values = [value for value, residual in zip(sorted_values, residuals) if residual <= threshold]

                # 绘制正常点
                ax1.scatter(normal_accuracies, normal_values, label=label, s=50, color=color)
                # 绘制噪音点
                ax1.scatter(noisy_accuracies, noisy_values, label=f"Noisy {label}", s=50, color=color, alpha=0.3)

                # 绘制拟合曲线
                ax1.plot(x_fit, y_fit, label=f'Polynomial Fit ({label})', linestyle='--', color=color)

    ax1.set_xlabel('RM Accuracy')
    ax1.set_ylabel(f'LM {metric_name.replace("_", " ").title()}')
    ax1.grid(True)

    # 添加次Y轴并绘制 RM Trained Steps 数据
    ax2 = ax1.twinx()
    rm_trained_steps_accuracies = []
    rm_trained_steps_values = []

    for run in runs:
        run_name = run.name.lower()

        task = keys_task_mapping[run_name[0]]

        if task in metric_name:
            step_match = re.search(r'_step_([0-9]+)', run_name)
            accuracy_match = re.search(r'acc_([0-9]+\.[0-9]+)', run_name)

            if step_match and accuracy_match:
                step = int(step_match.group(1))
                accuracy = float(accuracy_match.group(1))
                rm_trained_steps_accuracies.append(accuracy)
                rm_trained_steps_values.append(step)

    if rm_trained_steps_accuracies and rm_trained_steps_values:
        sorted_data = sorted(zip(rm_trained_steps_accuracies, rm_trained_steps_values), key=lambda x: x[0])
        sorted_accuracies, sorted_steps = zip(*sorted_data)

        # 拟合 RM Trained Steps 数据
        coefficients = np.polyfit(sorted_accuracies, sorted_steps, degree)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)

        # 计算残差
        y_pred_steps = polynomial(sorted_accuracies)
        residuals_steps = np.abs(np.array(sorted_steps) - y_pred_steps)
        threshold_steps = np.mean(residuals_steps) + 2 * np.std(residuals_steps)

        # 标记噪音点
        noisy_steps_accuracies = [acc for acc, residual in zip(sorted_accuracies, residuals_steps) if residual > threshold_steps]
        noisy_steps_values = [step for step, residual in zip(sorted_steps, residuals_steps) if residual > threshold_steps]

        # 标记正常点
        normal_steps_accuracies = [acc for acc, residual in zip(sorted_accuracies, residuals_steps) if residual <= threshold_steps]
        normal_steps_values = [step for step, residual in zip(sorted_steps, residuals_steps) if residual <= threshold_steps]

        # 绘制正常点
        ax2.scatter(normal_steps_accuracies, normal_steps_values, color='orange', marker='x', label=f'RM Trained Steps ({metric_name})')
        # 绘制噪音点
        ax2.scatter(noisy_steps_accuracies, noisy_steps_values, color='orange', marker='x', label=f'Noisy RM Trained Steps ({metric_name})', alpha=0.3)

        # 绘制拟合曲线
        ax2.plot(x_fit, y_fit, color='orange', linestyle='--', label=f'Polynomial Fit (RM Trained Steps - {metric_name})')

    ax2.set_ylabel('RM Trained Steps')

    # 改变图例位置到右上角
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper right', bbox_to_anchor=(1.5, 1), fancybox=True, shadow=True)

    plt.title(f'{metric_name.replace("_", " ").title()} Performance in {project.split("_")[0]}')
    plt.savefig(f"/home/llm/FineGrainedRLHF/Pictures/2D-C-v2/{metric_name}-Performance-{project.split('_')[0]}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# 获取所有run
runs = api.runs(f"{entity}/{project}")

# 绘制每个指标的性能图表
for metric_name in ['factuality_ratios', 'relevance_ratios', 'completeness_rewards']:
    plot_performance(metric_name, runs)
