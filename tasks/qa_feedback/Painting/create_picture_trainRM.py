import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 使用Seaborn样式
sns.set()

# 文件夹路径，需要替换为您的实际路径
# folder_path = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/rel_rm/50epoch_onstep'
folder_path = '/code/FineGrainedRLHF/tasks/qa_feedback/model_outputs/fact_rm/50epoch_onstep'

# 初始化步骤和精确度列表
steps = []
accs = []

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    match = re.search(r'model_step_(\d+)_f1_[0-9.]+_acc_([0-9.]+)', filename)
    if match:
        step = int(match.group(1))
        acc = float(match.group(2))
        steps.append(step)
        accs.append(acc)

# 确保按step值排序
sorted_indices = sorted(range(len(steps)), key=lambda k: steps[k])
steps = [steps[i] for i in sorted_indices]
accs = [accs[i] for i in sorted_indices]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(steps, accs, '-o')
plt.title("Model Accuracy by Step", fontsize=16, fontweight='bold')
plt.xlabel("Step", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 保存图表
plt.savefig("/home/llm/FineGrainedRLHF/Pictures/accuracy_by_step_fact.png", dpi=300)
# plt.savefig("/home/llm/FineGrainedRLHF/Pictures/accuracy_by_step_rel.png", dpi=300)
# 显示图表
plt.show()
