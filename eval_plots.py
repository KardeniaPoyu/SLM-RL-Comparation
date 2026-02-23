import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics():
    os.makedirs('plots', exist_ok=True)
    
    ppo_file = 'logs/ppo_metrics.csv'
    grpo_file = 'logs/grpo_metrics.csv'
    
    has_ppo = os.path.exists(ppo_file)
    has_grpo = os.path.exists(grpo_file)
    
    if not has_ppo and not has_grpo:
        print("No logs found to plot.")
        return
        
    plt.style.use('seaborn-v0_8-muted') # 使用更学术的绘图风格

    def load_and_process(file):
        df = pd.read_csv(file)
        # 对成功率等波动较大的数据进行滑动平均，使趋势更清晰
        df['success_rate_smooth'] = df['success_rate'].rolling(window=5, min_periods=1).mean()
        return df

    df_ppo = load_and_process(ppo_file) if has_ppo else None
    df_grpo = load_and_process(grpo_file) if has_grpo else None
    
    # 定义绘图函数以减少重复代码
    def create_plot(title, ylabel, ppo_col, grpo_col, filename, use_log=False):
        plt.figure(figsize=(10, 6))
        if has_ppo:
            plt.plot(df_ppo['step'], df_ppo[ppo_col], label='PPO (Critic-based)', alpha=0.8, linewidth=1.5)
        if has_grpo:
            plt.plot(df_grpo['step'], df_grpo[grpo_col], label='GRPO (Group-relative)', alpha=0.8, linewidth=1.5)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Update Steps (Batch size = 8)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        if use_log:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'plots/{filename}.png', dpi=300) # 提高分辨率用于论文
        plt.close()

    # Plot 1: 样本效率（平滑后的成功率）
    create_plot('Sample Efficiency (Arithmetic-24)', 'Success Rate (Moving Avg)', 
                'success_rate_smooth', 'success_rate_smooth', 'success_rate')
    
    # Plot 2: 梯度范数（稳定性对比）
    create_plot('Gradient Norm Stability', 'L2 Norm (Log Scale)', 
                'grad_norm', 'grad_norm', 'grad_norm', use_log=True)

    # Plot 3: 梯度二阶矩（核心论点：方差对比）
    # 这是你论文 5.3 节最重要的图
    create_plot('Gradient Second Moment (Variance Proxy)', 'Second Moment (Log Scale)', 
                'grad_second_moment', 'grad_second_moment', 'grad_variance', use_log=True)

    # Plot 4: KL 散度（策略漂移对比）
    create_plot('Policy Drift (KL Divergence)', 'KL Divergence', 
                'kl_div', 'kl_div', 'kl_div')

    print(f"📊 统计图表已生成至 'plots/' 目录。")
    print(f"提示：请确保 PPO 和 GRPO 的训练步数接近，以获得最佳对比效果。")

if __name__ == "__main__":
    plot_metrics()