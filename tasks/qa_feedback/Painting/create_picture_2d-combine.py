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

# project = "T5-small_RM_research_StepTest_StepOnly"
# project = "T5-base_RM_research_StepTest_StepOnly"
project = "T5-large_RM_research_StepTest_StepOnly"
entity = "battam"  # 替换为您的实体名称

# 定义要绘制的key映射字典
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

            steps = []
            values = []

            for run in runs:
                if run.name.startswith(prefix):
                    step_match = re.search(r'_step_([0-9]+)', run.name)

                    if step_match:
                        step = int(step_match.group(1))
                        history = run.history(keys=[key])
                        
                        if not history.empty and key in history:
                            value = history[key].max()
                            steps.append(step)
                            values.append(value)

            if steps and values:
                sorted_data = sorted(zip(steps, values), key=lambda x: x[0])
                sorted_steps, sorted_values = zip(*sorted_data)

                label = f"{key.split('/')[-1]} ({key_type})"
                # 多项式拟合
                coefficients = np.polyfit(sorted_steps, sorted_values, degree)
                polynomial = np.poly1d(coefficients)
                x_fit = np.linspace(min(sorted_steps), max(sorted_steps), 100)
                y_fit = polynomial(x_fit)
                
                # 计算残差
                y_pred = polynomial(sorted_steps)
                residuals = np.abs(np.array(sorted_values) - y_pred)
                threshold = np.mean(residuals) + 2 * np.std(residuals)

                # 标记噪音点
                noisy_steps = [step for step, residual in zip(sorted_steps, residuals) if residual > threshold]
                noisy_values = [value for value, residual in zip(sorted_values, residuals) if residual > threshold]

                # 标记正常点
                normal_steps = [step for step, residual in zip(sorted_steps, residuals) if residual <= threshold]
                normal_values = [value for value, residual in zip(sorted_values, residuals) if residual <= threshold]

                # 绘制正常点
                ax1.scatter(normal_steps, normal_values, label=label, s=50, color=color)
                # 绘制噪音点
                ax1.scatter(noisy_steps, noisy_values, label=f"Noisy {label}", s=50, color=color, alpha=0.3)

                # 绘制拟合曲线
                ax1.plot(x_fit, y_fit, label=f'Polynomial Fit ({label})', linestyle='--', color=color)

    ax1.set_xlabel('RM Trained Steps')
    ax1.set_ylabel(f'LM {metric_name.replace("_", " ").title()}')
    ax1.grid(True)

    # 添加次Y轴并绘制 RM Accuracy 数据
    ax2 = ax1.twinx()
    rm_accuracy_steps = []
    rm_accuracy_values = []

    # 仅绘制对应任务的 RM Accuracy 数据
    for run in runs:
        run_name = run.name.lower()

        task = keys_task_mapping[run_name[0]]

        if task in metric_name:
            accuracy_match = re.search(r'acc_([0-9]+\.[0-9]+)', run_name)
            step_match = re.search(r'_step_([0-9]+)', run_name)

            if accuracy_match and step_match:
                step = int(step_match.group(1))
                accuracy = float(accuracy_match.group(1))
                rm_accuracy_steps.append(step)
                rm_accuracy_values.append(accuracy)

    if rm_accuracy_steps and rm_accuracy_values:
        sorted_data = sorted(zip(rm_accuracy_steps, rm_accuracy_values), key=lambda x: x[0])
        sorted_steps, sorted_accuracies = zip(*sorted_data)

        # 拟合 RM Accuracy 数据
        coefficients = np.polyfit(sorted_steps, sorted_accuracies, degree)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x_fit)

        # 计算残差
        y_pred_acc = polynomial(sorted_steps)
        residuals_acc = np.abs(np.array(sorted_accuracies) - y_pred_acc)
        threshold_acc = np.mean(residuals_acc) + 2 * np.std(residuals_acc)

        # 标记噪音点
        noisy_steps_acc = [step for step, residual in zip(sorted_steps, residuals_acc) if residual > threshold_acc]
        noisy_values_acc = [value for value, residual in zip(sorted_accuracies, residuals_acc) if residual > threshold_acc]

        # 标记正常点
        normal_steps_acc = [step for step, residual in zip(sorted_steps, residuals_acc) if residual <= threshold_acc]
        normal_values_acc = [value for value, residual in zip(sorted_accuracies, residuals_acc) if residual <= threshold_acc]

        # 绘制正常点
        ax2.scatter(normal_steps_acc, normal_values_acc, color='orange', marker='x', label=f'RM Accuracy ({metric_name})')
        # 绘制噪音点
        ax2.scatter(noisy_steps_acc, noisy_values_acc, color='orange', marker='x', label=f'Noisy RM Accuracy ({metric_name})', alpha=0.3)

        # 绘制拟合曲线
        ax2.plot(x_fit, y_fit, color='orange', linestyle='--', label=f'Polynomial Fit (RM Accuracy - {metric_name})')

    ax2.set_ylabel('RM Accuracy')

    # 改变图例位置到右上角
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper right', bbox_to_anchor=(1.5, 1), fancybox=True, shadow=True)

    plt.title(f'{metric_name.replace("_", " ").title()} Performance in {project.split("_")[0]}')
    plt.savefig(f"/home/llm/FineGrainedRLHF/Pictures/2D-C/{metric_name}-Performance-{project.split('_')[0]}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# 获取所有run
runs = api.runs(f"{entity}/{project}")

# 绘制每个指标的性能图表
for metric_name in ['factuality_ratios', 'relevance_ratios', 'completeness_rewards']:
    plot_performance(metric_name, runs)
