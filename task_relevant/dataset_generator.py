from experimenting_code_refactored import get_top_tfidf_files_for_words, reiterate_background, create_background
from word_search import csv_converter
import os
import tqdm
import json
import csv
import requests
from bs4 import BeautifulSoup
import os
import re
import pandas as pd



def extract_link(url):
    source_url = requests.get(url)
    soup = BeautifulSoup(source_url.content, "html.parser")
    links = soup.find_all("a", href=True)

    return soup


def download_section_links(base_url, folder_name="downloaded_sections"):
    response = requests.get(base_url)
    response.raise_for_status()  #
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    links = soup.find_all("a", href=True)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for link in links:
        href = link.get("href")
        if href and "section" in href:
            full_url = requests.compat.urljoin(base_url, href)
            response = requests.get(full_url)
            response.raise_for_status() 
            downloaded_html = response.text

            h = html2text.HTML2Text()
            h.ignore_links = True      
            h.ignore_images = True     
            h.body_width = 0           
            h.unicode_snob = True      # Handle Unicode characters better
            h.skip_internal_links = True # Skip anchors within the same document
            h.ignore_tables = False    # You might want to ignore tables too if structure is not needed
            h.single_line_break = True # Treat <br> as single newline

            plain_text = h.handle(downloaded_html)


            clean_href = re.sub(r'[#?].*$', '', href)
            base_filename = os.path.basename(clean_href)            
            file_name_without_ext = os.path.splitext(base_filename)[0]
            output_path = os.path.join(folder_name, f"{file_name_without_ext}.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(plain_text)
            print(f"Saved: {file_name_without_ext}.txt")


def extract_sentences_from_prompt(prompt_text):
    """
    Extract sentences A and B from the prompt text.
    
    Args:
        prompt_text (str): The full prompt text containing the sentences
        
    Returns:
        tuple: (sentence_a, sentence_b) or (None, None) if not found
    """
    import re
    
    # Look for pattern: (A) [sentence] followed by (B) [sentence]
    pattern = r'\(A\)\s+([^\n]+)\s*\n\s*\(B\)\s+([^\n]+)'
    match = re.search(pattern, prompt_text)
    
    if match:
        sentence_a = match.group(1).strip()
        sentence_b = match.group(2).strip()
        return sentence_a, sentence_b
    
    return None, None

def csv_converter_training(csv_file, jsonl_file):
    """
    Convert CSV to JSONL for training data, generating background on the fly from choice_a and choice_b.
    """
    columns = ["original_id", "prompt", "original_key", "choice_a", "choice_b", "original_explanation"]
    tuple_columns = ["choice_a", "choice_b"]
    tuple_field_name = "answers"
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            filtered_row = {k: row[k] for k in columns if k in row}
            
            # Convert choices to tuple
            if tuple_columns and tuple_field_name:
                filtered_row[tuple_field_name] = tuple(row[col] for col in tuple_columns if col in row)
                for col in tuple_columns:
                    filtered_row.pop(col, None)

            # Generate background from choice_a and choice_b
            if "choice_a" in row and "choice_b" in row:
                choice_a = row["choice_a"]
                choice_b = row["choice_b"]
                
                # Generate background for the choices
                script_dir = os.path.dirname(os.path.abspath(__file__))
                relevant_background = reiterate_background(choice_a, choice_b)
                choices_tfidf = get_top_tfidf_files_for_words(os.path.join(script_dir, "downloaded_sections"), choice_a, choice_b, top_n=2)
                background = f"""
The following definitions may be particularly useful:
{relevant_background}

The following are the top TF-IDF files for the words in choice A and choice B:
{choices_tfidf}"""
                filtered_row["prompt_file_content"] = background

            json_line = json.dumps(filtered_row, ensure_ascii=False)
            f_out.write(json_line + '\n')

def csv_converter_test(csv_file, jsonl_file):
    """
    Convert CSV to JSONL for test data, generating background directly from extracted sentences.
    Uses caching to avoid regenerating backgrounds for the same stimulus ID.
    """
    import re

    columns = ["original_id", "prompt", "original_key", "correct_answer", "choice_a", "choice_b", "original_explanation"]
    tuple_columns = ["choice_a", "choice_b"]
    tuple_field_name = "answers"
    
    # Cache to store background by stimulus ID
    stimulus_background_cache = {}
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f_in, open(jsonl_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            filtered_row = {k: row[k] for k in columns if k in row}
            
            # Convert choices to tuple
            if tuple_columns and tuple_field_name:
                filtered_row[tuple_field_name] = tuple(row[col] for col in tuple_columns if col in row)
                for col in tuple_columns:
                    filtered_row.pop(col, None)

            # Generate or retrieve cached background from extracted sentences
            if "prompt" in filtered_row and "original_id" in filtered_row:
                # Extract stimulus ID (e.g., 'stim177' from 'stim177_gpt3.5_A_1')
                original_id = filtered_row["original_id"]
                stimulus_match = re.match(r'(stim\d+)', original_id)
                
                if stimulus_match:
                    stimulus_id = stimulus_match.group(1)
                    
                    # Check if we already have background for this stimulus
                    if stimulus_id in stimulus_background_cache:
                        filtered_row["prompt_file_content"] = stimulus_background_cache[stimulus_id]
                    else:
                        # Generate background for new stimulus
                        sentence_a, sentence_b = extract_sentences_from_prompt(filtered_row["prompt"])
                        if sentence_a and sentence_b:
                            # Generate background for the extracted sentences
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            relevant_background = reiterate_background(sentence_a, sentence_b)
                            choices_tfidf = get_top_tfidf_files_for_words(os.path.join(script_dir, "downloaded_sections"), sentence_a, sentence_b, top_n=2)
                            sentence_background = f"""
The following definitions may be particularly useful:
{relevant_background}

The following are the top TF-IDF files for the words in choice A and choice B:
{choices_tfidf}"""
                            # Cache the background for future use
                            stimulus_background_cache[stimulus_id] = sentence_background
                            filtered_row["prompt_file_content"] = sentence_background

            json_line = json.dumps(filtered_row, ensure_ascii=False)
            f_out.write(json_line + '\n')

def generate_lojban_background(data):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, data), "r", encoding="utf-8") as f:
        for line in f:
            data_json = json.loads(line.strip())
            choice_a = data_json.get("choice_a")
            choice_b = data_json.get("choice_b")
            original_id = data_json.get("original_id")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            relevant_background = reiterate_background(choice_a, choice_b)
            choices_tfidf = get_top_tfidf_files_for_words(os.path.join(script_dir, "downloaded_sections"), choice_a, choice_b, top_n=2)
            background = f"""
The following definitions may be particularly useful:
{relevant_background}

The following are the top TF-IDF files for the words in choice A and choice B:
{choices_tfidf}"""
            _ = create_background(background, original_id)


def main():
    url = "https://lojban.org/publications/cll/cll_v1.1_xhtml-section-chunks/"
    # download_section_links(url) uncomment if you want to download the book sections
    
    # Convert training data (generates background on the fly from choice_a and choice_b)
    csv_converter_training("data.csv", "lojban_dataset.jsonl")
    
    # Convert test data (generates background on the fly from extracted sentences)
    csv_converter_test("lojban_test_set.csv", "lojban_dataset_test.jsonl")

if __name__ == "__main__":
    main()