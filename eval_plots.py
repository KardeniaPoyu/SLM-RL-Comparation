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
        
    if has_ppo:
        df_ppo = pd.read_csv(ppo_file)
    if has_grpo:
        df_grpo = pd.read_csv(grpo_file)
        
    # Plot 1: Sample Efficiency (Success Rate over Steps)
    plt.figure(figsize=(10, 6))
    if has_ppo:
        plt.plot(df_ppo['step'], df_ppo['success_rate'], label='PPO', alpha=0.8)
    if has_grpo:
        plt.plot(df_grpo['step'], df_grpo['success_rate'], label='GRPO', alpha=0.8)
    plt.title('Sample Efficiency (Success Rate vs Step)')
    plt.xlabel('Step')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/success_rate.png')
    plt.close()
    
    # Plot 2: Gradient Norm (Stability)
    plt.figure(figsize=(10, 6))
    if has_ppo:
        plt.plot(df_ppo['step'], df_ppo['grad_norm'], label='PPO', alpha=0.8)
    if has_grpo:
        plt.plot(df_grpo['step'], df_grpo['grad_norm'], label='GRPO', alpha=0.8)
    plt.title('Gradient Norm over Training')
    plt.xlabel('Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/grad_norm.png')
    plt.close()

    # Plot 3: Gradient Variance / Second Moment
    plt.figure(figsize=(10, 6))
    if has_ppo:
        plt.plot(df_ppo['step'], df_ppo['grad_second_moment'], label='PPO Grad Var', alpha=0.8)
    if has_grpo:
        plt.plot(df_grpo['step'], df_grpo['grad_second_moment'], label='GRPO Grad Var', alpha=0.8)
    plt.title('Gradient Second Moment (Variance Proxy)')
    plt.xlabel('Step')
    plt.ylabel('Second Moment')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/grad_variance.png')
    plt.close()

    # Plot 4: KL Divergence Growth
    plt.figure(figsize=(10, 6))
    if has_ppo:
        plt.plot(df_ppo['step'], df_ppo['kl_div'], label='PPO KL', alpha=0.8)
    if has_grpo:
        plt.plot(df_grpo['step'], df_grpo['kl_div'], label='GRPO KL', alpha=0.8)
    plt.title('KL Divergence from Reference Model')
    plt.xlabel('Step')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/kl_div.png')
    plt.close()

    print("Plots saved to the 'plots/' directory.")

if __name__ == "__main__":
    plot_metrics()
