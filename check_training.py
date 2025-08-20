import json
import re
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Any, TypeVar, Type, cast, List, Optional

T = TypeVar("T")

def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x

def from_int(x: Any) -> int:
    assert isinstance(x, int)
    return x

def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()

@dataclass
class TrainingLine:
    prompt: str
    completion: str
    debater_name: str  # "Debater_A" or "Debater_B"
    percentage: int    # The percentage as an integer (e.g., 51 for 51%)

    @staticmethod
    def from_dict(obj: Any) -> 'TrainingLine':
        assert isinstance(obj, dict)
        prompt = from_str(obj.get("prompt"))
        completion = from_str(obj.get("completion"))

        # Parse debater name and percentage from completion
        debater_name, percentage = parse_completion(completion)

        return TrainingLine(prompt, completion, debater_name, percentage)

    def to_dict(self) -> dict:
        result: dict = {}
        result["prompt"] = from_str(self.prompt)
        result["completion"] = from_str(self.completion)
        result["debater_name"] = from_str(self.debater_name)
        result["percentage"] = from_int(self.percentage)
        return result

def parse_completion(completion: str) -> tuple[str, int]:
    """
    Parse the completion string to extract debater name and percentage.

    Expected format: "Debater_A | 51%" or "Debater_B | 50%"

    Args:
        completion: The completion string to parse

    Returns:
        Tuple of (debater_name, percentage)

    Raises:
        ValueError: If the completion string doesn't match expected format
    """
    # Remove any leading/trailing whitespace
    completion = completion.strip()

    # Regular expression to match the pattern
    pattern = r'^(Debater[_ ][AB])\s*\|\s*(\d+)%$'
    match = re.match(pattern, completion)

    if not match:
        raise ValueError(f"Invalid completion format: '{completion}'. Expected format: 'Debater_A | 51%' or 'Debater_B | 50%'")

    debater_name = match.group(1)
    percentage = int(match.group(2))

    # Validate debater name
    if debater_name == "Debater A":
        debater_name = "Debater_A"
    if debater_name == "Debater B":
        debater_name = "Debater_B"

    if debater_name not in ["Debater_A", "Debater_B"]:
        raise ValueError(f"Invalid debater name: '{debater_name}'. Must be 'Debater_A' or 'Debater_B'")

    # Validate percentage range (optional - you might want percentages outside 0-100)
    if not (0 <= percentage <= 100):
        print(f"Warning: Percentage {percentage} is outside typical range 0-100")

    return debater_name, percentage

def training_line_from_dict(s: Any) -> TrainingLine:
    return TrainingLine.from_dict(s)

def training_line_to_dict(x: TrainingLine) -> Any:
    return to_class(TrainingLine, x)

def parse_jsonl_file(file_path: str) -> List[TrainingLine]:
    """
    Parse a JSONL file and return a list of TrainingLine objects.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of TrainingLine objects
    """
    training_lines = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse JSON from the line
                    json_obj = json.loads(line)

                    # Convert to TrainingLine object
                    training_line = training_line_from_dict(json_obj)
                    training_lines.append(training_line)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except (AssertionError, KeyError, ValueError) as e:
                    print(f"Error creating TrainingLine from line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    return training_lines

def analyze_data(training_lines: List[TrainingLine]):
    """
    Analyze the parsed training data and print statistics.
    """
    if not training_lines:
        print("No data to analyze.")
        return

    # Count debaters
    debater_a_count = sum(1 for line in training_lines if line.debater_name == "Debater_A")
    debater_b_count = sum(1 for line in training_lines if line.debater_name == "Debater_B")

    # Calculate percentage statistics
    percentages_a = [line.percentage for line in training_lines if line.debater_name == "Debater_A"]
    percentages_b = [line.percentage for line in training_lines if line.debater_name == "Debater_B"]

    print(f"\n=== Data Analysis ===")
    print(f"Total entries: {len(training_lines)}")
    print(f"Debater_A wins: {debater_a_count} ({debater_a_count/len(training_lines)*100:.1f}%)")
    print(f"Debater_B wins: {debater_b_count} ({debater_b_count/len(training_lines)*100:.1f}%)")

    if percentages_a:
        print(f"\nDebater_A percentages:")
        print(f"  Average: {sum(percentages_a)/len(percentages_a):.1f}%")
        print(f"  Min: {min(percentages_a)}%")
        print(f"  Max: {max(percentages_a)}%")

    if percentages_b:
        print(f"\nDebater_B percentages:")
        print(f"  Average: {sum(percentages_b)/len(percentages_b):.1f}%")
        print(f"  Min: {min(percentages_b)}%")
        print(f"  Max: {max(percentages_b)}%")

def create_plots(training_lines: List[TrainingLine], output_prefix: str = "debate_analysis"):
    """
    Create and save plots analyzing the debate data.

    Args:
        training_lines: List of TrainingLine objects
        output_prefix: Prefix for saved image files
    """
    if not training_lines:
        print("No data to plot.")
        return

    # Extract data
    debater_a_count = sum(1 for line in training_lines if line.debater_name == "Debater_A")
    debater_b_count = sum(1 for line in training_lines if line.debater_name == "Debater_B")

    percentages_a = [line.percentage for line in training_lines if line.debater_name == "Debater_A"]
    percentages_b = [line.percentage for line in training_lines if line.debater_name == "Debater_B"]
    all_percentages = [line.percentage for line in training_lines]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Debate Analysis Dashboard', fontsize=16, fontweight='bold')

    # 1. Win count pie chart
    labels = ['Debater_A', 'Debater_B']
    sizes = [debater_a_count, debater_b_count]
    colors = ['#ff9999', '#66b3ff']

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title(f'Win Distribution\n(Total: {len(training_lines)} debates)', fontweight='bold')

    # 2. Overall confidence histogram
    ax2.hist(all_percentages, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Confidence Percentage')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Confidence Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    mean_conf = np.mean(all_percentages)
    median_conf = np.median(all_percentages)
    ax2.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.1f}%')
    ax2.axvline(median_conf, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_conf:.1f}%')
    ax2.legend()

    # 3. Debater_A confidence histogram
    if percentages_a:
        ax3.hist(percentages_a, bins=15, alpha=0.7, color='#ff9999', edgecolor='black')
        ax3.set_xlabel('Confidence Percentage')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Debater_A Confidence Distribution\n({len(percentages_a)} wins)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        mean_a = np.mean(percentages_a)
        ax3.axvline(mean_a, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_a:.1f}%')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Debater_A wins', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)
        ax3.set_title('Debater_A Confidence Distribution', fontweight='bold')

    # 4. Debater_B confidence histogram
    if percentages_b:
        ax4.hist(percentages_b, bins=15, alpha=0.7, color='#66b3ff', edgecolor='black')
        ax4.set_xlabel('Confidence Percentage')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Debater_B Confidence Distribution\n({len(percentages_b)} wins)', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        mean_b = np.mean(percentages_b)
        ax4.axvline(mean_b, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {mean_b:.1f}%')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Debater_B wins', transform=ax4.transAxes,
                 ha='center', va='center', fontsize=12)
        ax4.set_title('Debater_B Confidence Distribution', fontweight='bold')

    plt.tight_layout()

    # Save the plot
    filename = f"{output_prefix}_dashboard.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Dashboard plot saved as: {filename}")

    # Create a separate detailed comparison plot
    plt.figure(figsize=(12, 8))

    # Subplot 1: Side-by-side histograms
    plt.subplot(2, 2, 1)
    if percentages_a:
        plt.hist(percentages_a, bins=15, alpha=0.6, color='#ff9999', label='Debater_A', edgecolor='black')
    if percentages_b:
        plt.hist(percentages_b, bins=15, alpha=0.6, color='#66b3ff', label='Debater_B', edgecolor='black')
    plt.xlabel('Confidence Percentage')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Box plot comparison
    plt.subplot(2, 2, 2)
    data_to_plot = []
    labels_to_plot = []
    if percentages_a:
        data_to_plot.append(percentages_a)
        labels_to_plot.append('Debater_A')
    if percentages_b:
        data_to_plot.append(percentages_b)
        labels_to_plot.append('Debater_B')

    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels_to_plot)
        plt.ylabel('Confidence Percentage')
        plt.title('Confidence Distribution Box Plot')
        plt.grid(True, alpha=0.3)

    # Subplot 3: Win rate bar chart
    plt.subplot(2, 2, 3)
    debaters = ['Debater_A', 'Debater_B']
    win_counts = [debater_a_count, debater_b_count]
    colors = ['#ff9999', '#66b3ff']
    bars = plt.bar(debaters, win_counts, color=colors, edgecolor='black')
    plt.ylabel('Number of Wins')
    plt.title('Win Count Comparison')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, count in zip(bars, win_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Confidence ranges
    plt.subplot(2, 2, 4)
    confidence_ranges = ['0-50%', '51-60%', '61-70%', '71-80%', '81-90%', '91-100%']

    def count_in_range(percentages, min_val, max_val):
        return sum(1 for p in percentages if min_val <= p <= max_val)

    ranges = [(0, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
    counts_a = [count_in_range(percentages_a, min_val, max_val) for min_val, max_val in ranges] if percentages_a else [0] * 6
    counts_b = [count_in_range(percentages_b, min_val, max_val) for min_val, max_val in ranges] if percentages_b else [0] * 6

    x = np.arange(len(confidence_ranges))
    width = 0.35

    plt.bar(x - width/2, counts_a, width, label='Debater_A', color='#ff9999', edgecolor='black')
    plt.bar(x + width/2, counts_b, width, label='Debater_B', color='#66b3ff', edgecolor='black')

    plt.xlabel('Confidence Range')
    plt.ylabel('Frequency')
    plt.title('Confidence Range Distribution')
    plt.xticks(x, confidence_ranges, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save the detailed comparison plot
    filename2 = f"{output_prefix}_detailed.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"Detailed comparison plot saved as: {filename2}")

    plt.show()

def process_jsonl_file(file_path: str):
    """
    Process a JSONL file and print information about the parsed data.
    """
    print(f"Processing file: {file_path}")
    training_lines = parse_jsonl_file(file_path)

    print(f"Successfully parsed {len(training_lines)} training lines")

    # Print first few examples if any exist
    if training_lines:
        print("\nFirst few examples:")
        for i, line in enumerate(training_lines[:5]):
            print(f"\nExample {i+1}:")
            print(f"  Prompt: {line.prompt[:100]}{'...' if len(line.prompt) > 100 else ''}")
            print(f"  Completion: {line.completion}")
            print(f"  Debater: {line.debater_name}")
            print(f"  Percentage: {line.percentage}%")

    # Analyze the data
    analyze_data(training_lines)

    # Create and save plots
    create_plots(training_lines, "debate_analysis")

    return training_lines

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "training_data.jsonl"
        print("No filename provided. Using default: 'training_data.jsonl'")
        print("To specify a file, run: python check_training.py <your_file.jsonl>")


    # Process the file
    training_data = process_jsonl_file(file_path)

    # You can now work with the training_data list
    # For example, filter by debater:
    # debater_a_lines = [line for line in training_data if line.debater_name == "Debater_A"]
    # high_confidence = [line for line in training_data if line.percentage >= 80]

    # Convert back to dictionaries if needed:
    # dict_data = [training_line_to_dict(line) for line in training_data]
