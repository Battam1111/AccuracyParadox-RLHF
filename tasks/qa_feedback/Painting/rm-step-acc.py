import wandb
import re
import pandas as pd

# 创建wandb API对象
api = wandb.Api()

# 定义项目和实体名称
project = "T5-small_RM_research_StepTest_StepOnly"
entity = "battam"

# 定义键前缀与实际键的映射关系
keys_prefix_mapping = {
    "1": "eval/eval_rm/factuality_ratios",
    "2": "eval/eval_rm/relevance_ratios",
    "3": "eval/eval_rm/completeness_rewards"
}

# 存储每类奖励模型的数据
data_dict = {
    "factuality": {"steps": [], "accuracies": []},
    "relevance": {"steps": [], "accuracies": []},
    "completeness": {"steps": [], "accuracies": []}
}

# 任务类型映射
task_mapping = {
    "1": "factuality",
    "2": "relevance",
    "3": "completeness"
}

# 遍历映射关系中的前缀和键
for prefix, key in keys_prefix_mapping.items():
    runs = api.runs(f"{entity}/{project}")
    
    for run in runs:
        if run.name.startswith(prefix):
            accuracy_match = re.search(r'acc_([0-9]+\.[0-9]+)', run.name)
            step_match = re.search(r'_step_([0-9]+)', run.name)
            max_step_match = re.search(r'KL-([0-9]+)', run.name)

            if accuracy_match and step_match:
                accuracy = float(accuracy_match.group(1))
                step = int(step_match.group(1))
                history = run.history(keys=[key])

                if max_step_match:
                    max_step = int(max_step_match.group(1)) + 50
                    history = history[history['_step'] <= max_step]

                if not history.empty and key in history:
                    data_dict[task_mapping[prefix]]["steps"].append(step)
                    data_dict[task_mapping[prefix]]["accuracies"].append(accuracy)

# 将数据转换为DataFrame并计算范围
df_list = []
for task in data_dict:
    steps = data_dict[task]["steps"]
    accuracies = data_dict[task]["accuracies"]
    
    if steps and accuracies:
        step_range = f"{min(steps)}~{max(steps)}"
        accuracy_range = f"{min(accuracies):.2f}~{max(accuracies):.2f}"
        df = pd.DataFrame({
            "Task Type": [task] * len(steps),
            "Steps": steps,
            "Accuracies": accuracies
        })
        df["Steps Range"] = step_range
        df["Accuracies Range"] = accuracy_range
        df_list.append(df)

# 合并所有DataFrame
final_df = pd.concat(df_list, ignore_index=True)

# 显示数据
print(final_df)

# 保存数据到文件
final_df.to_csv('/home/llm/FineGrainedRLHF/Datas/reward_model_data.csv', index=False)
print("数据已保存为 reward_model_data.csv")
