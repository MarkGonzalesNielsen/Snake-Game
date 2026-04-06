"""
Plot utilities til Snake AI eksperimenter.
Genererer og gemmer visualiseringer automatisk.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def load_metrics(experiment_name: str, results_dir: str = "results") -> dict:
    """Load metrics fra et eksperiment."""
    path = f"{results_dir}/{experiment_name}/metrics.json"
    with open(path, 'r') as f:
        return json.load(f)


def plot_single_experiment(experiment_name: str, results_dir: str = "results", show: bool = True):
    """
    Generer og gem plots for ét eksperiment.
    Gemmer: fitness.png, score.png, deaths.png, summary.png
    """
    metrics = load_metrics(experiment_name, results_dir)
    save_dir = f"{results_dir}/{experiment_name}"
    
    gens = metrics["generations"]
    best_fit = metrics["best_fitness"]
    avg_fit = metrics["avg_fitness"]
    div = metrics["diversity"]
    
    has_detailed = "best_score" in metrics and "avg_score" in metrics
    
    plt.style.use('dark_background')
    
    # === Plot 1: Fitness ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best_fit, label='Best Fitness', color='#10b981', linewidth=2)
    ax.plot(gens, avg_fit, label='Avg Fitness', color='#3b82f6', linewidth=2, alpha=0.7)
    ax.fill_between(gens, avg_fit, alpha=0.2, color='#3b82f6')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(f'{experiment_name} - Fitness Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    peak_idx = np.argmax(best_fit)
    ax.annotate(f'Peak: {best_fit[peak_idx]:,.0f}', 
                xy=(gens[peak_idx], best_fit[peak_idx]),
                xytext=(gens[peak_idx] - 15, best_fit[peak_idx] * 0.8),
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
                fontsize=10, color='#10b981')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fitness.png", dpi=150, facecolor='#111827')
    if show:
        plt.show()
    plt.close()
    
    # === Plot 2: Score (hvis detailed metrics) ===
    if has_detailed:
        best_score = metrics["best_score"]
        avg_score = metrics["avg_score"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(gens, best_score, label='Best Score', color='#f59e0b', linewidth=2)
        axes[0].plot(gens, avg_score, label='Avg Score', color='#fbbf24', linewidth=2, alpha=0.7)
        axes[0].fill_between(gens, avg_score, alpha=0.2, color='#f59e0b')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Score (mad spist)')
        axes[0].set_title('Score Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if "best_moves_per_food" in metrics:
            mpf = metrics["best_moves_per_food"]
            mpf_clean = [m if m > 0 and m < 1000 else None for m in mpf]
            axes[1].plot(gens, mpf_clean, color='#06b6d4', linewidth=2)
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Moves per Food')
            axes[1].set_title('Efficiency (lavere = bedre)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/score.png", dpi=150, facecolor='#111827')
        if show:
            plt.show()
        plt.close()
    
    # === Plot 3: Death Reasons (hvis detailed metrics) ===
    if "deaths_wall" in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        walls = metrics["deaths_wall"]
        tails = metrics["deaths_tail"]
        loops = metrics["deaths_loop"]
        max_moves = metrics["deaths_max_moves"]
        
        axes[0].stackplot(gens, walls, tails, loops, max_moves,
                         labels=['Wall', 'Tail', 'Loop', 'Max Moves'],
                         colors=['#ef4444', '#f59e0b', '#a855f7', '#6b7280'],
                         alpha=0.8)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Deaths')
        axes[0].set_title('Death Reasons Over Time')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        total_deaths = {
            'Wall': sum(walls),
            'Tail': sum(tails),
            'Loop': sum(loops),
            'Max Moves': sum(max_moves)
        }
        colors = ['#ef4444', '#f59e0b', '#a855f7', '#6b7280']
        non_zero = {k: v for k, v in total_deaths.items() if v > 0}
        
        if non_zero:
            axes[1].pie(non_zero.values(), labels=non_zero.keys(), colors=colors[:len(non_zero)],
                       autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Total Death Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/deaths.png", dpi=150, facecolor='#111827')
        if show:
            plt.show()
        plt.close()
    
    # === Plot 4: Diversity ===
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, div, color='#a855f7', linewidth=2)
    ax.fill_between(gens, div, alpha=0.3, color='#a855f7')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Diversity (std)', fontsize=12)
    ax.set_title(f'{experiment_name} - Genetic Diversity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/diversity.png", dpi=150, facecolor='#111827')
    if show:
        plt.show()
    plt.close()
    
    # === Plot 5: Summary ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{experiment_name} - Training Summary', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(gens, best_fit, color='#10b981', linewidth=2, label='Best')
    axes[0, 0].plot(gens, avg_fit, color='#3b82f6', linewidth=2, alpha=0.7, label='Avg')
    axes[0, 0].set_title('Fitness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    if has_detailed:
        axes[0, 1].plot(gens, metrics["best_score"], color='#f59e0b', linewidth=2, label='Best')
        axes[0, 1].plot(gens, metrics["avg_score"], color='#fbbf24', linewidth=2, alpha=0.7, label='Avg')
        axes[0, 1].set_title('Score (mad)')
        axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(gens, div, color='#a855f7', linewidth=2)
    axes[0, 2].set_title('Diversity')
    axes[0, 2].grid(True, alpha=0.3)
    
    if has_detailed and "avg_moves" in metrics:
        axes[1, 0].plot(gens, metrics["best_moves"], color='#06b6d4', linewidth=2, label='Best')
        axes[1, 0].plot(gens, metrics["avg_moves"], color='#22d3ee', linewidth=2, alpha=0.7, label='Avg')
        axes[1, 0].set_title('Moves')
        axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    if "deaths_wall" in metrics:
        total_deaths = {
            'Wall': sum(metrics["deaths_wall"]),
            'Tail': sum(metrics["deaths_tail"]),
            'Loop': sum(metrics["deaths_loop"]),
            'Max': sum(metrics["deaths_max_moves"])
        }
        non_zero = {k: v for k, v in total_deaths.items() if v > 0}
        if non_zero:
            axes[1, 1].pie(non_zero.values(), labels=non_zero.keys(),
                          colors=['#ef4444', '#f59e0b', '#a855f7', '#6b7280'][:len(non_zero)],
                          autopct='%1.0f%%')
            axes[1, 1].set_title('Death Reasons')
    
    axes[1, 2].axis('off')
    
    max_score = max(metrics["best_score"]) if has_detailed else "N/A"
    avg_score_final = f"{metrics['avg_score'][-1]:.1f}" if has_detailed else "N/A"
    
    stats_text = f"""
    📊 STATISTICS
    
    Peak Fitness:     {max(best_fit):>12,.0f}
    Peak Generation:  {gens[np.argmax(best_fit)]:>12}
    
    Max Score:        {max_score:>12}
    Final Avg Score:  {avg_score_final:>12}
    
    Final Avg Fitness:{avg_fit[-1]:>12,.1f}
    
    Start Diversity:  {div[0]:>12.3f}
    Final Diversity:  {div[-1]:>12.3f}
    
    Generations:      {len(gens):>12}
    """
    axes[1, 2].text(0.05, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', color='white')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/summary.png", dpi=150, facecolor='#111827')
    if show:
        plt.show()
    plt.close()
    
    print(f"✅ Plots gemt i {save_dir}/")
    print(f"   - fitness.png")
    if has_detailed:
        print(f"   - score.png")
    if "deaths_wall" in metrics:
        print(f"   - deaths.png")
    print(f"   - diversity.png")
    print(f"   - summary.png")


def plot_comparison(experiment_names: List[str], results_dir: str = "results", 
                    save_path: Optional[str] = None, show: bool = True):
    """Sammenlign flere eksperimenter."""
    plt.style.use('dark_background')
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Eksperiment Sammenligning', fontsize=16, fontweight='bold')
    
    stats = []
    
    for i, name in enumerate(experiment_names):
        try:
            metrics = load_metrics(name, results_dir)
            color = colors[i % len(colors)]
            
            gens = metrics["generations"]
            
            axes[0, 0].plot(gens, metrics["best_fitness"], label=name, color=color, linewidth=2)
            axes[0, 1].plot(gens, metrics["avg_fitness"], label=name, color=color, linewidth=2)
            axes[0, 2].plot(gens, metrics["diversity"], label=name, color=color, linewidth=2)
            
            if "best_score" in metrics:
                axes[1, 0].plot(gens, metrics["best_score"], label=name, color=color, linewidth=2)
            
            if "avg_score" in metrics:
                axes[1, 1].plot(gens, metrics["avg_score"], label=name, color=color, linewidth=2)
            
            # Stats - med avg_score tilføjet
            max_score = max(metrics["best_score"]) if "best_score" in metrics else 0
            final_avg_score = metrics["avg_score"][-1] if "avg_score" in metrics else 0
            stats.append({
                'name': name,
                'peak_fit': max(metrics["best_fitness"]),
                'peak_gen': gens[np.argmax(metrics["best_fitness"])],
                'max_score': max_score,
                'avg_score': final_avg_score,
                'final_div': metrics["diversity"][-1]
            })
            
        except FileNotFoundError:
            print(f"⚠️  Ingen resultater for {name}")
    
    axes[0, 0].set_title('Best Fitness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Average Fitness')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Diversity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Best Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Avg Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Stats table - med avg_score kolonne
    axes[1, 2].axis('off')
    if stats:
        table_text = "  {:15} {:>10} {:>6} {:>8} {:>8} {:>8}\n".format(
            "Experiment", "Peak Fit", "Gen", "Max Scr", "Avg Scr", "Div")
        table_text += "  " + "-" * 58 + "\n"
        for s in stats:
            table_text += "  {:15} {:>10,.0f} {:>6} {:>8} {:>8.1f} {:>8.3f}\n".format(
                s['name'][:15], s['peak_fit'], s['peak_gen'], 
                s['max_score'], s['avg_score'], s['final_div'])
        
        axes[1, 2].text(0.02, 0.5, table_text, fontsize=10, family='monospace',
                        verticalalignment='center', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#111827')
        print(f"✅ Comparison plot gemt: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def generate_all_plots(results_dir: str = "results", show: bool = False):
    """Generer plots for alle eksperimenter."""
    experiments = []
    
    for name in os.listdir(results_dir):
        metrics_path = f"{results_dir}/{name}/metrics.json"
        if os.path.exists(metrics_path):
            experiments.append(name)
            print(f"📊 Genererer plots for {name}...")
            plot_single_experiment(name, results_dir, show=show)
    
    if len(experiments) > 1:
        print(f"\n📊 Genererer comparison plot...")
        plot_comparison(experiments, results_dir, 
                       save_path=f"{results_dir}/comparison.png", show=show)
    
    print(f"\n✅ Færdig! Genererede plots for {len(experiments)} eksperimenter.")


if __name__ == "__main__":
    generate_all_plots()