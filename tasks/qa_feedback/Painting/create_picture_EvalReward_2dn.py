import matplotlib.pyplot as plt
import wandb
import re
import seaborn as sns
import numpy as np

sns.set()

api = wandb.Api()

# project = "T5-small_RM_research_StepTest_StepOnly"
# project = "T5-base_RM_research_StepTest_StepOnly"
project = "T5-large_RM_research_StepTest_StepOnly"
entity = "battam"  # 替换为您的实体名称

# 根据run编号选择的keys
keys_prefix_mapping = {
    "1": "eval/eval_rm/rewards",
    "2": "eval/eval_rm/rewards",
    "3": "eval/eval_rm/rewards"
}

keys_type = {
    "1": "factuality_",
    "2": "relevance_",
    "3": "completeness_"
}

# 遍历每种key，获取数据并绘制图表
for prefix, key in keys_prefix_mapping.items():
    runs = api.runs(f"{entity}/{project}")
    accuracies = []
    steps = []
    max_values = []
    
    for run in runs:
        if run.name.startswith(prefix):
            accuracy_match = re.search(r'acc_([0-9]+\.[0-9]+)', run.name)
            step_match = re.search(r'_step_([0-9]+)', run.name)
            max_step_match = re.search(r'KL-([0-9]+)', run.name)

            Rtype = keys_type[prefix]

            if accuracy_match and step_match:
                accuracy = float(accuracy_match.group(1))
                step = int(step_match.group(1))
                history = run.history(keys=[key])
                
                # 如果匹配到最大步数限制
                if max_step_match:
                    max_step = int(max_step_match.group(1)) + 50
                    # 过滤历史数据，只保留步数小于等于最大步数的数据
                    history = history[history['_step'] <= max_step]

                if not history.empty and key in history:
                    max_value = history[key].max()
                    accuracies.append(accuracy)
                    steps.append(step)
                    max_values.append(max_value)

    # 绘制以accuracy为横轴的图
    if accuracies and max_values:
        # 对数据进行排序
        sorted_acc_data = sorted(zip(accuracies, max_values), key=lambda x: x[0])
        sorted_accuracies, sorted_max_values = zip(*sorted_acc_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_accuracies, sorted_max_values, 'o', label=key.split('/')[-1])  # 只绘制点

        # 多项式拟合
        degree = 3  # 拟合多项式的阶数
        coefficients = np.polyfit(sorted_accuracies, sorted_max_values, degree)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(min(sorted_accuracies), max(sorted_accuracies), 100)
        y_fit = polynomial(x_fit)
        plt.plot(x_fit, y_fit, 'r--', label=f'Polynomial Fit (Degree {degree})')

        plt.title(f"LM Performance in {project.split('_')[0]} (EvalRM)")
        plt.xlabel("RM Accuracy")
        plt.ylabel(f"LM {Rtype} {key.split('/')[-1]}")
        plt.legend()
        plt.grid(True)
        safe_filename = key.split('/')[1] + "-" + Rtype + key.split('/')[-1] + "-accuracy"
        plt.savefig(f"/home/llm/FineGrainedRLHF/Pictures/2D-Eval/LM-{safe_filename}-{project.split('_')[0]}.pdf", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f"No data available for key: {key}")
        plt.close()

    # 绘制以step为横轴的图
    if steps and max_values:
        sorted_step_data = sorted(zip(steps, max_values), key=lambda x: x[0])
        sorted_steps, sorted_max_values = zip(*sorted_step_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_steps, sorted_max_values, 'o', label=key.split('/')[-1])  # 只绘制点

        # 多项式拟合
        coefficients = np.polyfit(sorted_steps, sorted_max_values, degree)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(min(sorted_steps), max(sorted_steps), 100)
        y_fit = polynomial(x_fit)
        plt.plot(x_fit, y_fit, 'r--', label=f'Polynomial Fit (Degree {degree})')

        plt.title(f"LM Performance in {project.split('_')[0]} (EvalRM)")
        plt.xlabel("RM Trained Steps")
        plt.ylabel(f"LM {Rtype} {key.split('/')[-1]}")
        plt.legend()
        plt.grid(True)
        safe_filename = key.split('/')[1] + "-" + Rtype + key.split('/')[-1] + "-step"
        plt.savefig(f"/home/llm/FineGrainedRLHF/Pictures/2D-Eval/LM-{safe_filename}-{project.split('_')[0]}.pdf", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f"No data available for key: {key}")
        plt.close()
