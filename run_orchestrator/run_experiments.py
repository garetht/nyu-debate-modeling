import sys
import yaml
import subprocess
import json
from jsonschema import validate
import os
import glob
from graph_generator import graph_results
from datetime import datetime

def run_experiments(config_name):
    yaml_path = f"run_orchestrator/runs/{config_name}.yaml"
    schema_path = f"run_orchestrator/experiment_orchestrator.schema.json"

    # Check if files exist
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)
    if not os.path.exists(schema_path):
        print(f"Error: {schema_path} not found.")
        sys.exit(1)

    # Load the schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Load the yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Validate the yaml against the schema
    validate(instance=data, schema=schema)

    for (index, config_group) in enumerate(data['configurations']):
        instance_ip = config_group['instance_ip']
        config = config_group['configurations']
        name = config['name']
        num_iters = config['num_iters']
        count = config['count']

        for i in range(count):
            command = [
                './cli.sh',
                '--ip',
                instance_ip,
                'bg-task',
                'start',
                '-n',
                name.replace(" ", ""),
                '--',
                'python',
                './scripts/run_debate.py',
                f'--configuration={name}',
                f'--num_iters={num_iters}',
                # f'--starting_index={(i + count * index) * 157}',
            ]

            print(f"Running command: {' '.join(command)}")
            
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='')
                for line in process.stderr:
                    print(line, end='')

def download_results(config_name):
    yaml_path = f"run_orchestrator/runs/{config_name}.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Get unique IPs
    ip_home_dirs = set((cg['instance_ip'], cg['home_dir'] )for cg in data['configurations'])

    for (ip, home_dir) in ip_home_dirs:
        remote_path = f"{home_dir}/outputs/"
        local_path = "./outputs/"
        
        command = [
            './cli.sh',
            '--ip',
            ip,
            'rsync-to-host',
            '--remote-path',
            remote_path,
            '--local-path',
            local_path
        ]

        print(f"Running command: {' '.join(command)}")
        
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='')
            for line in process.stderr:
                print(line, end='')

def merge_data():
    stats_dir = "outputs/stats"
    file_paths = glob.glob(os.path.join(stats_dir, "*.json"))
    # Exclude already merged files
    file_paths = [p for p in file_paths if not os.path.basename(p).startswith("merge-")]

    if not file_paths:
        print("No stats files found to merge.")
        return

    merged_data = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not merged_data:
                merged_data = data
            else:
                for i, item in enumerate(data):
                    if i < len(merged_data):
                        merge_recursively(merged_data[i], item)
                    else:
                        merged_data.append(item)

    recalculate_recursively(merged_data)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    merged_filename = os.path.join(stats_dir, f"merge-{timestamp}.json")

    with open(merged_filename, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged stats saved to {merged_filename}")

def merge_recursively(d1, d2):
    for k, v2 in d2.items():
        if k in d1:
            v1 = d1[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                merge_recursively(v1, v2)
            elif isinstance(v1, list) and isinstance(v2, list):
                v1.extend(v2)
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if "average" not in k and "pct" not in k:
                    d1[k] = v1 + v2
        else:
            d1[k] = v2

def recalculate_recursively(data):
    if isinstance(data, dict):
        for k, v in data.items():
            recalculate_recursively(v)
        
        if 'matches' in data and data['matches'] > 0:
            if 'wins' in data:
                data['average_reward'] = data['wins'] / data['matches']
            if 'binary_wins' in data:
                data['binary_win_pct'] = data['binary_wins'] / data['matches']
            if 'correct_wins' in data and 'correct_matches' in data and data['correct_matches'] > 0:
                data['average_correct_reward'] = data['correct_wins'] / data['correct_matches']
            if 'first_wins' in data and 'first_matches' in data and data['first_matches'] > 0:
                data['average_first_wins'] = data['first_wins'] / data['first_matches']
            if 'wins' in data and 'first_wins' in data and 'matches' in data and 'first_matches' in data and (data['matches'] - data['first_matches']) > 0:
                    data['average_second_wins'] = (data['wins'] - data['first_wins']) / (data['matches'] - data['first_matches'])

    elif isinstance(data, list):
        for item in data:
            recalculate_recursively(item)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_orchestrator/run_experiments.py <start|merge-data|graph> [args]")
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == 'start':
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py start <config_name>")
            sys.exit(1)
        config_name = sys.argv[2]
        run_experiments(config_name)
    elif subcommand == 'merge-data':
        merge_data()
    elif subcommand == 'graph':
        if len(sys.argv) < 3:
            print("Usage: python run_orchestrator/run_experiments.py graph <file_path>")
            sys.exit(1)
        file_path = sys.argv[2]
        graph_results(file_path)
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Usage: python run_orchestrator/run_experiments.py <start|merge-data|graph> [args]")
        sys.exit(1)

if __name__ == "__main__":
    main()
