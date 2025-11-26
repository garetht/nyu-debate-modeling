import sys
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

def graph_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    with open(file_path, 'r') as f:
        data = json.load(f)

    models_to_plot = []
    judge_calls_section = None

    # Find all model statistics sections and the judge calls section
    for section in data:
        if isinstance(section, dict):
            if 'standard' in section and 'binary' in section:
                judge_calls_section = section
                continue
            for key, value in section.items():
                if isinstance(value, dict) and 'binary_win_pct' in value:
                    models_to_plot.append((key, value))

    if not models_to_plot:
        print("Error: No model statistics found in the JSON file.")
        sys.exit(1)

    # Generate a separate plot for each model
    for model_name, main_stats_section in models_to_plot:
        fig, axs = plt.subplots(3, 2, figsize=(18, 22))
        fig.suptitle(f'Detailed Analysis for: {model_name}\n(from {os.path.basename(file_path)})', fontsize=20, y=1.0)

        # Plot 1: Key Performance Indicators
        ax = axs[0, 0]
        labels = ['Binary Win %', 'Avg Reward', 'Avg Correct Reward']
        values = [
            main_stats_section.get('binary_win_pct', 0) * 100,
            main_stats_section.get('average_reward', 0) * 100,
            main_stats_section.get('average_correct_reward', 0) * 100
        ]
        bars = ax.bar(labels, values, color=['#4c72b0', '#55a868', '#c44e52'])
        ax.set_title('Key Performance Indicators', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.bar_label(bars, fmt='%.2f%%')
        ax.tick_params(axis='x', rotation=10)

        # Plot 2: Raw Event Counts
        ax = axs[0, 1]
        labels = ['Matches', 'Wins', 'Correct Matches', 'First Matches']
        values = [
            main_stats_section.get('matches', 0),
            main_stats_section.get('wins', 0),
            main_stats_section.get('correct_matches', 0),
            main_stats_section.get('first_matches', 0)
        ]
        bars = ax.bar(labels, values, color=['#8172b2', '#64b5cd', '#9d755d', '#da8d31'])
        ax.set_title('Raw Event Counts', fontsize=14)
        ax.set_ylabel('Count', fontsize=12)
        ax.bar_label(bars, fmt='%d')
        ax.tick_params(axis='x', rotation=10)

        # Plot 3: Positional Advantage
        ax = axs[1, 0]
        labels = ['Avg First Wins', 'Avg Second Wins']
        values = [
            main_stats_section.get('average_first_wins', 0),
            main_stats_section.get('average_second_wins', 0)
        ]
        bars = ax.bar(labels, values, color=['#4c72b0', '#c44e52'])
        ax.set_title('Positional Win Averages', fontsize=14)
        ax.set_ylabel('Average Win Rate', fontsize=12)
        ax.set_ylim(0, 1)
        ax.bar_label(bars, fmt='%.3f')

        # Plot 4: Reward Breakdown
        ax = axs[1, 1]
        labels = ['Avg Reward', 'Avg Correct Reward', 'Avg Incorrect Reward']
        values = [
            main_stats_section.get('average_reward', 0),
            main_stats_section.get('average_correct_reward', 0),
            main_stats_section.get('average_incorrect_reward', 0)
        ]
        bars = ax.bar(labels, values, color=['#55a868', '#4c72b0', '#c44e52'])
        ax.set_title('Reward Averages', fontsize=14)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.bar_label(bars, fmt='%.3f')
        ax.tick_params(axis='x', rotation=10)

        # Plot 5 & 6: Judge Calls
        if judge_calls_section:
            judge_key = f"{model_name}_v_{model_name}"
            # Standard Judge
            ax = axs[2, 0]
            standard_data = judge_calls_section.get('standard', {}).get(judge_key)
            if standard_data:
                labels = ['Matches', 'Correct Calls', 'First Calls']
                values = [
                    standard_data.get('matches', 0),
                    standard_data.get('correct_calls', 0),
                    standard_data.get('first_calls', 0)
                ]
                bars = ax.bar(labels, values, color=['#8172b2', '#55a868', '#4c72b0'])
                ax.set_title(f'Judge Calls (Standard) for {model_name}', fontsize=14)
                ax.set_ylabel('Count', fontsize=12)
                ax.bar_label(bars, fmt='%.1f')
            else:
                ax.text(0.5, 0.5, f'No Standard Judge Data for\n{judge_key}', ha='center', va='center', fontsize=12)
            ax.tick_params(axis='x', rotation=10)

            # Binary Judge
            ax = axs[2, 1]
            binary_data = judge_calls_section.get('binary', {}).get(judge_key)
            if binary_data:
                labels = ['Matches', 'Correct Calls', 'First Calls']
                values = [
                    binary_data.get('matches', 0),
                    binary_data.get('correct_calls', 0),
                    binary_data.get('first_calls', 0)
                ]
                bars = ax.bar(labels, values, color=['#8172b2', '#55a868', '#4c72b0'])
                ax.set_title(f'Judge Calls (Binary) for {model_name}', fontsize=14)
                ax.set_ylabel('Count', fontsize=12)
                ax.bar_label(bars, fmt='%d')
            else:
                ax.text(0.5, 0.5, f'No Binary Judge Data for\n{judge_key}', ha='center', va='center', fontsize=12)
            ax.tick_params(axis='x', rotation=10)
        else:
            for i in range(2):
                axs[2, i].text(0.5, 0.5, 'No Judge Call Data Found', ha='center', va='center', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        graph_dir = "graphs"
        os.makedirs(graph_dir, exist_ok=True)
        base_filename = os.path.basename(file_path).replace('.json', '')
        model_name_sanitized = re.sub(r'[\s/:]', '_', model_name)
        graph_filename = os.path.join(graph_dir, f"{base_filename}--{model_name_sanitized}.png")
        plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
        plt.close(fig) # Close the figure to free memory
        print(f"Graph for {model_name} saved to {graph_filename}")
