from experimenting_code_refactored import get_top_tfidf_files_for_words, reiterate_background
import os
import tqdm
import json
import csv


def csv_converter_final(csv_file, jsonl_file, columns, tuple_columns=None, tuple_field_name=None, prompt_folder=None, prompt_field_name="prompt_file_content"):
    with open(csv_file, 'r', encoding='utf-8') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            filtered_row = {k: row[k] for k in columns if k in row}
            if tuple_columns and tuple_field_name:
                filtered_row[tuple_field_name] = tuple(row[col] for col in tuple_columns if col in row)
                for col in tuple_columns:
                    filtered_row.pop(col, None)

            if prompt_folder and "original_id" in filtered_row:
                file_path = os.path.join(prompt_folder, f"background_{filtered_row['original_id']}.txt")
                try:
                    with open(file_path, 'r', encoding='utf-8') as pf:
                        filtered_row[prompt_field_name] = pf.read()
                except Exception as e:
                    filtered_row[prompt_field_name] = f"File not found: {file_path}"

            json_line = json.dumps(filtered_row, ensure_ascii=False)
            f_out.write(json_line + '\n')

csv_converter_final(
    "data.csv",
    "lojban_dataset.jsonl",
    columns=["original_id", "prompt", "original_key","choice_a", "choice_b"],
    tuple_columns=["choice_a", "choice_b"],
    tuple_field_name="answers",
    prompt_folder=r"c:\Users\tolly\Desktop\MARS\nyu-debate-modeling\task_relevant\background_files",
    prompt_field_name="prompt_file_content"
)

def generate_lojban_dataset():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "converted_data"), "r", encoding="utf-8") as f:
        for line in f:
            data_json = json.loads(line.strip())
            choice_a = data_json.get("choice_a")
            choice_b = data_json.get("choice_b")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            relevant_background = reiterate_background(choice_a, choice_b)
            choices_tfidf = get_top_tfidf_files_for_words(os.path.join(script_dir, "downloaded_sections"), choice_a, choice_b, top_n=2)


