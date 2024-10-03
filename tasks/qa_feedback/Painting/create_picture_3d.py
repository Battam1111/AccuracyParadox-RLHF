import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pandas as pd
import wandb
import re

# 设置seaborn样式
sns.set(style="whitegrid")

# 设置matplotlib字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 创建wandb API对象
api = wandb.Api()

# 定义项目和实体名称
project = "T5-small_RM_research_StepTest_StepOnly"
# project = "T5-base_RM_research_StepTest_StepOnly"
# project = "T5-large_RM_research_StepTest_StepOnly"
entity = "battam"

# 定义键前缀与实际键的映射关系
keys_prefix_mapping = {
    "1": "eval/eval_rm/factuality_ratios",
    "2": "eval/eval_rm/relevance_ratios",
    "3": "eval/eval_rm/completeness_rewards"
}

# 绘制3D图表及其在各个平面的投影
def plot_3d_surface_with_separate_projections(x, y, z, title, xlabel, ylabel, zlabel, filename):
    # 绘制三维曲面图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # 绘制3D曲面
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
    fig.colorbar(surface, ax=ax, pad=0.1, label=zlabel)

    # 添加数据点
    ax.scatter(x, y, z, color='r', s=20, label='Data Points', alpha=0.6)

    # 添加等高线
    cset = ax.contour(X, Y, Z, zdir='z', offset=min(z), cmap='coolwarm', linestyles='dashed')
    ax.clabel(cset, fmt='%2.2f', colors='k', fontsize=10)

    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)

    # 调整视角
    ax.view_init(elev=35, azim=120)

    # 反转RM Accuracy轴方向
    ax.invert_yaxis()

    plt.legend(loc='upper left')
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()

    # 创建等高线网格，确保数据覆盖整个平面
    yi_grid = np.linspace(min(y), max(y), 100)
    zi_grid = np.linspace(min(z), max(z), 100)
    xi_grid = np.linspace(min(x), max(x), 100)
    
    # 绘制XY平面投影
    fig_xy = plt.figure(figsize=(10, 8))
    ax_xy = fig_xy.add_subplot(111)
    contour_xy = ax_xy.contourf(X, Y, Z, cmap='viridis', alpha=0.6)
    fig_xy.colorbar(contour_xy, ax=ax_xy, pad=0.1, label=zlabel)

    ax_xy.set_title(f'{title} - Projection on XY Plane', fontsize=14, fontweight='bold')
    ax_xy.set_xlabel(xlabel, fontsize=12)
    ax_xy.set_ylabel(ylabel, fontsize=12)

    plt.savefig(filename.replace('.pdf', '_XY_Projection.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

    # 绘制YZ平面投影，确保数据顺序和维度正确
    YZ = griddata((y, z), x, (yi_grid[None,:], zi_grid[:,None]), method='cubic')

    fig_yz = plt.figure(figsize=(10, 8))
    ax_yz = fig_yz.add_subplot(111)
    contour_yz = ax_yz.contourf(yi_grid, zi_grid, YZ, cmap='viridis', alpha=0.6)
    fig_yz.colorbar(contour_yz, ax=ax_yz, pad=0.1, label=xlabel)

    ax_yz.set_title(f'{title} - Projection on YZ Plane', fontsize=14, fontweight='bold')
    ax_yz.set_xlabel(ylabel, fontsize=12)
    ax_yz.set_ylabel(zlabel, fontsize=12)

    plt.savefig(filename.replace('.pdf', '_YZ_Projection.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

    # 绘制ZX平面投影，确保数据顺序和维度正确
    ZX = griddata((z, x), y, (zi_grid[None,:], xi_grid[:,None]), method='cubic')

    fig_zx = plt.figure(figsize=(10, 8))
    ax_zx = fig_zx.add_subplot(111)
    contour_zx = ax_zx.contourf(zi_grid, xi_grid, ZX, cmap='viridis', alpha=0.6)
    fig_zx.colorbar(contour_zx, ax=ax_zx, pad=0.1, label=ylabel)

    ax_zx.set_title(f'{title} - Projection on ZX Plane', fontsize=14, fontweight='bold')
    ax_zx.set_xlabel(zlabel, fontsize=12)
    ax_zx.set_ylabel(xlabel, fontsize=12)

    plt.savefig(filename.replace('.pdf', '_ZX_Projection.pdf'), dpi=600, bbox_inches='tight')
    plt.show()

# 遍历映射关系中的前缀和键
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

            if accuracy_match and step_match:
                accuracy = float(accuracy_match.group(1))
                step = int(step_match.group(1))
                history = run.history(keys=[key])

                if max_step_match:
                    max_step = int(max_step_match.group(1)) + 50
                    history = history[history['_step'] <= max_step]

                if not history.empty and key in history:
                    max_value = history[key].max()
                    accuracies.append(accuracy)
                    steps.append(step)
                    max_values.append(max_value)

    if steps and accuracies and max_values and (len(steps) == len(accuracies) == len(max_values)):
        sorted_steps, sorted_accuracies, sorted_max_values = zip(*sorted(zip(steps, accuracies, max_values), key=lambda x: x[0]))

        filename = key.split('/')[1] + "-" + key.split('/')[-1]

        safe_filename = f"/home/llm/FineGrainedRLHF/Pictures/3D/3D_Surface-{filename}-{project.split('_')[0]}.pdf"

        plot_3d_surface_with_separate_projections(sorted_steps, sorted_accuracies, sorted_max_values,
                        f'{project.split("_")[0]}-{key.split("/")[-1]}',
                        'RM Training Steps',
                        'RM Accuracy',
                        'LM Performance',
                        safe_filename)
    else:
        print(f"数据列表长度不一致或存在空列表：{key}")
