"""
Word Search Module for Lojban Language Processing

This module provides functionality for processing and analyzing Lojban text data,
including searching for words in various Lojban dictionaries, converting file formats,
and creating comprehensive language resources.
"""

import tiktoken
import pandas as pd
import csv
import json
import re
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


def convert_cmavo_rafsi(input_txt: str, output_txt: str, rafsi_flag: bool) -> None:
    """
    Convert cmavo and rafsi files from dubious formatting to CSV format.
    
    Args:
        input_txt (str): Path to the input text file
        output_txt (str): Path to the output CSV file
        rafsi_flag (bool): If True, use single space pattern for rafsi; 
                          if False, use double space pattern for cmavo
    
    Returns:
        None
    """
    if rafsi_flag:
        split_pattern = r"[ \u00A0]{1,}"  # any space like character
    else:
        split_pattern = r"[ \u00A0]{2,}"
        
    with open(input_txt, "r", newline="", encoding="utf-8") as infile:
        with open(output_txt, "w", newline="", encoding="utf-8") as outfile:
            csv_writer = csv.writer(outfile, delimiter=",")
            for line in infile:
                clean_line = line.strip()
                parts = re.split(split_pattern, clean_line)
                csv_writer.writerow(parts)


def handle_bad_line(bad_line: List[str]) -> List[str]:
    """
    Handle malformed CSV lines by truncating to maximum of 3 fields.
    
    Args:
        bad_line (List[str]): List of fields from a malformed CSV line
        
    Returns:
        List[str]: Truncated list with maximum 3 fields
    """
    if len(bad_line) > 3:
        return bad_line[:3]
    return bad_line


def handle_bad_rafsi(bad_line: List[str]) -> List[str]:
    """
    Handle malformed rafsi CSV lines by joining excess fields.
    
    Args:
        bad_line (List[str]): List of fields from a malformed rafsi CSV line
        
    Returns:
        List[str]: Properly formatted list with 3 fields
    """
    joined_rest = " ".join(bad_line[2:])
    return [bad_line[0], bad_line[1], joined_rest]


def searching_match(word: str, datasets: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Search through datasets for matching Lojban words.
    
    Args:
        word (str): The Lojban word to search for
        datasets (List[pd.DataFrame]): List of dataframes to search through
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing exact matches, or None if no match found
    """
    for d in datasets:
        for w in d["Lojban"]:
            if str(w) == word:
                exact_matches = d[d['Lojban'] == word]
                return exact_matches
    return None


def csv_converter(csv_file: str, jsonl_file: str) -> None:
    """
    Convert CSV file to JSONL format.
    
    Args:
        csv_file (str): Path to input CSV file
        jsonl_file (str): Path to output JSONL file
        
    Returns:
        None
    """
    with open(csv_file, 'r', encoding='utf-8') as csv_input:
        csv_reader = csv.DictReader(csv_input)
        with open(jsonl_file, 'w', encoding='utf-8') as jsonl_output:
            for row in csv_reader:
                json_line = json.dumps(row, ensure_ascii=False)
                jsonl_output.write(json_line + '\n')


def return_sentences() -> List[str]:
    """
    Extract Lojban sentences from questions in converted data file.
    
    Returns:
        List[str]: List of all sentences starting with (A) or (B)
    """
    all_sentences = []
    with open("converted_data", "r", encoding="utf-8") as f:
        for line in f:
            data_json = json.loads(line.strip())
            lines = data_json["prompt"].splitlines()
            for l in lines:
                if l.startswith("(A)") or l.startswith("(B)"):
                    all_sentences.append(l)
    return all_sentences


def grab_unique_words(sentences: List[str]) -> List[str]:
    """
    Extract unique words from a list of sentences.
    
    Args:
        sentences (List[str]): List of sentences to process
        
    Returns:
        List[str]: List of unique words found in all sentences
    """
    unique_words = []
    for s in sentences:
        for w in s.split():
            if w not in unique_words:
                unique_words.append(w)
    return unique_words


def create_dfs(list_unique_words: List[str], data_gismu: pd.DataFrame, 
               data_lujvo: pd.DataFrame, data_cmavo: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create separate DataFrames for different Lojban word types from unique words.
    
    Args:
        list_unique_words (List[str]): List of unique words to categorize
        data_gismu (pd.DataFrame): DataFrame containing gismu data
        data_lujvo (pd.DataFrame): DataFrame containing lujvo data
        data_cmavo (pd.DataFrame): DataFrame containing cmavo data
        
    Returns:
        Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - List of words not found in any dataset
            - DataFrame of found gismu
            - DataFrame of found lujvo
            - DataFrame of found cmavo
    """
    not_found_words = []
    found_gismu = []
    found_lujvo = []
    found_cmavo = []
    
    for w in list_unique_words:
        matched = searching_match(w, [data_gismu, data_lujvo, data_cmavo])
        if matched is not None:
            matched_dict = matched.iloc[0].to_dict()
            
            if matched_dict["Type"] == "gismu":
                found_gismu.append(matched_dict)
            elif matched_dict["Type"] == "lujvo":
                found_lujvo.append(matched_dict)
            elif matched_dict["Type"] == "cmavo":
                found_cmavo.append(matched_dict)
        else:
            not_found_words.append(w)
    
    return not_found_words, pd.DataFrame(found_gismu), pd.DataFrame(found_lujvo), pd.DataFrame(found_cmavo)


def return_rasfi(rasfi_def: str, not_found_final: Set[str], data_rafsi: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str], pd.DataFrame]:
    """
    Find rafsi matches from unmatched words using pattern matching.
    
    Args:
        rasfi_def (str): Path to output rafsi definition file
        not_found_final (Set[str]): Set of words not found in main dictionaries
        data_rafsi (pd.DataFrame): DataFrame containing rafsi data
        
    Returns:
        Tuple[Dict[str, List[str]], List[str], pd.DataFrame]:
            - Dictionary mapping input strings to their rafsi components
            - List of words that couldn't be matched
            - DataFrame of filtered rafsi data
    """
    rafsi_search = data_rafsi["Lojban"].to_list()
    not_matched = []
    compiled_patterns = sorted([(re.escape(str(s))) for s in rafsi_search], key=lambda x: (x))
    matched = {}
    rafsi_matched = []
    rafsi_gismu_matched = []

    for input_string in not_found_final:
        current_search_string = input_string 
        current_matches = []

        while current_search_string:
            found_match_segment = False
            for pattern_str in compiled_patterns:
                if current_search_string.startswith(pattern_str):
                    match_gismu = searching_match(pattern_str, [data_rafsi])
                    if match_gismu is not None:
                        matched_dict = match_gismu.iloc[0].to_dict()
                        current_matches.append(matched_dict["Lojban Gismu"])
                        rafsi_gismu_matched.append(matched_dict["Lojban"])
                        rafsi_matched.append(matched_dict)
                        current_search_string = current_search_string[len(pattern_str):]
                        found_match_segment = True
                        break

            if not found_match_segment:
                if current_search_string:
                    current_search_string = current_search_string[1:] 
                else:
                    break
    
        if current_matches:
            matched[input_string] = current_matches
        else:
            not_matched.append(input_string)

    with open(rasfi_def, 'w', encoding='utf-8', newline="") as rafsi_f:
        for match in rafsi_matched:
            json_line = json.dumps(match, ensure_ascii=False)
            rafsi_f.write(json_line + "\n")

    filtered_rafsi = data_rafsi[data_rafsi["Lojban"].isin(rafsi_gismu_matched)]    
    return matched, not_matched, filtered_rafsi


def filter_not_found(not_found_words: List[str]) -> Tuple[Set[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Filter and categorize words that weren't found in main dictionaries.
    
    Args:
        not_found_words (List[str]): List of words not found in dictionaries
        
    Returns:
        Tuple[Set[str], List[str], List[str], List[str], List[str], List[str]]:
            - Set of words for final processing
            - List of cmevla (proper names)
            - List of incorrectly formatted lujvo
            - List of fu'ivla
            - List of experimental gismu
            - List of standard gismu
    """
    cmevla = []
    wrong_lujvo = ["kurjysi'u", 'frilyrai', 'fesybaktu', 'tarbykansa', 'solnuncanci', 
                   "tarbykansi'u", 'seltcitygau', 'toljinsa', "ruskygu'e", "kurkydu'e", 
                   'vimstizu', "kanryze'a", 'bardyrai', "kajnyta'e", "jdimyjdikygau", 
                   'snimycarvi', 'zilfadni']
    fuivla = ['pelpeli', '.aidji', '.ekcala', 'ckuliki', 'iumle', "kancusu'oi", 
              'bolgaro', 'relxima']
    experimental_gismu = ["darca"]
    gismu = ["serbo"]

    for i in not_found_words:
        if i.startswith(".") and i.endswith("."):
            cmevla.append(i)

    not_found_final = set(not_found_words) - set(wrong_lujvo) - set(fuivla) - set(experimental_gismu) - set(gismu) - set(cmevla)
    return not_found_final, cmevla, wrong_lujvo, fuivla, experimental_gismu, gismu


def get_online_translation(word: str) -> Tuple[bool, str]:
    """
    Fetch online translation for a Lojban word from vlasisku.lojban.org.
    
    Args:
        word (str): The Lojban word to translate
        
    Returns:
        Tuple[bool, str]: Success flag and either the definition or the original word
    """
    base_url = "https://vlasisku.lojban.org/"
    search_url = f"{base_url}{word}"

    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        definition_element = soup.find('p', class_='definition')    

        definition_parts = []

        if definition_element is not None:
            for content in definition_element.contents:
                if content.name == 'sub':
                    sub_text = content.get_text(strip=True)
                    if definition_parts and definition_parts[-1].strip().endswith('x'):
                        definition_parts[-1] = definition_parts[-1].strip() + f"_{{{sub_text}}}"
                else:
                    text_content = content.get_text(strip=True)
                    definition_parts.append(text_content)

            definition = ' '.join(definition_parts).strip()
        else:
            return False, word
        return True, definition
    except Exception:
        return False, word


def create_lujvo(filtered_data_rafsi: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Create lujvo DataFrame from rafsi combinations with online translations.
    
    Args:
        filtered_data_rafsi (Dict[str, List[str]]): Dictionary mapping words to their rafsi components
        
    Returns:
        pd.DataFrame: DataFrame containing lujvo information with translations
    """
    df = pd.DataFrame(columns=["Lojban", "Lojban Composition", "English"])
    for k, v in filtered_data_rafsi.items():
        success, output_val = get_online_translation(k)
        composition = "+".join(v)
        english = output_val if success else "No definition has been found"
        
        new_row = {"Lojban": k, "Lojban Composition": composition, 
                  "English": english, "Type": "lujvo"}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df


def create_jsonl_defs(file_name: str, data_list: Union[List[Dict], List], flag: bool) -> None:
    """
    Create JSONL files for different types of Lojban language data.
    
    Args:
        file_name (str): Path to output JSONL file
        data_list (Union[List[Dict], List]): Data to write to file
        flag (bool): If True, write dictionary data; if False, write document data
        
    Returns:
        None
    """
    with open(file_name, 'w', encoding='utf-8', newline="") as f:
        if flag:
            for item in data_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")
        else:
            for indx, data in enumerate(data_list):
                data_to_write = {"id": indx, "content": data.page_content, 
                               "metadata": data.metadata}
                json_line = json.dumps(data_to_write, ensure_ascii=False)
                f.write(json_line + "\n")


def read_jsonl_file(jsonl_file: str, flag: bool = False) -> Union[List[Dict], str]:
    """
    Read JSONL file and return data in specified format.
    
    Args:
        jsonl_file (str): Path to JSONL file to read
        flag (bool): If True, return list of dictionaries; if False, return JSON string
        
    Returns:
        Union[List[Dict], str]: Either list of records or formatted JSON string
    """
    records = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    
    if flag:
        return records
    return json.dumps(records, indent=2)

def main():
    """
    Main function that processes Lojban data and generates filtered JSONL files.
    This replicates the functionality from word_search.ipynb.
    """
    # Initialize tiktoken encoder
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Set up file paths
    notebook_dir = Path(__file__).parent
    input_tsv_gismu = notebook_dir / "input_data" / "gismu_list_format_2_(less_info_but_tab_delimited).txt"
    input_txt_lujvo = notebook_dir / "input_data" / "lujvo.txt"
    input_txt_cmavo = notebook_dir / "input_data" / "cmavo.txt"
    output_csv_cmavo = notebook_dir / "input_data" / "output_cmavo.csv"
    input_txt_rafsi = notebook_dir / "input_data" / "rafsi.txt"
    output_csv_rafsi = notebook_dir / "input_data" / "output_rafsi.csv"
    
    # Define column titles
    column_titles = ["Lojban", "Lojban definition", "English"]
    column_titles_lujvo = ["Lojban", "Lojban Composition", "English", "Arguments"]
    column_titles_cmavo = ["Lojban", "Formal Language", "English", "Definition", "Confer"]
    column_titles_rafsi = ["Lojban", "Lojban Gismu", "English"]
    
    # Load and process data
    print("Loading Lojban data...")
    
    # Load lujvo data
    data_lujvo = pd.read_csv(input_txt_lujvo, sep=":", encoding='utf-8', on_bad_lines='warn', names=column_titles_lujvo)
    data_lujvo["Type"] = "lujvo"
    
    # Load gismu data
    data_gismu = pd.read_csv(input_tsv_gismu, sep="\t", on_bad_lines='warn', names=column_titles)
    data_gismu["Type"] = "gismu"
    
    # Process and load cmavo data
    convert_cmavo_rafsi(str(input_txt_cmavo), str(output_csv_cmavo), rafsi_flag=False)
    data_cmavo = pd.read_csv(output_csv_cmavo, on_bad_lines=handle_bad_line, names=column_titles_cmavo, engine="python")
    data_cmavo["Type"] = "cmavo"
    
    # Process and load rafsi data
    convert_cmavo_rafsi(str(input_txt_rafsi), str(output_csv_rafsi), rafsi_flag=True)
    data_rafsi = pd.read_csv(output_csv_rafsi, sep=",", encoding='utf-8', on_bad_lines=handle_bad_rafsi, 
                            header=None, names=column_titles_rafsi, engine="python")
    data_rafsi["Type"] = "rafsi"
    
    # Convert CSV to JSONL and extract sentences
    print("Converting data and extracting sentences...")
    csv_converter("data.csv", "converted_data")
    
    # Extract unique words from sentences
    sentences = return_sentences()
    list_unique_words = grab_unique_words(sentences)
    
    # Create dataframes for different word types
    print("Categorizing words...")
    not_found_words, found_gismu, found_lujvo, found_cmavo = create_dfs(list_unique_words, data_gismu, data_lujvo, data_cmavo)
    
    # Filter not found words
    not_found_final, cmevla, wrong_lujvo, fuivla, experimental_gismu, gismu = filter_not_found(not_found_words)
    
    # Process rafsi matches
    print("Processing rafsi matches...")
    rasfi_def = "rasfi_def.jsonl"
    filtered_lujvo_rasfi, not_matched, filtered_data_rasfi = return_rasfi(rasfi_def, not_found_final, data_rafsi)
    
    # Create comprehensive rafsi and gismu filters
    filtered_rasfi_gismu = data_rafsi[data_rafsi["Lojban Gismu"].isin(found_gismu["Lojban"])]
    filtered_rasfi = pd.concat([filtered_rasfi_gismu, filtered_data_rasfi], ignore_index=True).drop_duplicates()
    
    filtered_gismu_not_found = data_gismu[data_gismu["Lojban"].isin(filtered_data_rasfi["Lojban Gismu"])]
    filtered_gismu = pd.concat([filtered_gismu_not_found, found_gismu], ignore_index=True).drop_duplicates()
    
    not_filtered_rasfi_gismu = filtered_data_rasfi[~filtered_data_rasfi["Lojban Gismu"].isin(filtered_gismu_not_found["Lojban"])]
    filtered_rasfi_cmavo = data_cmavo[data_cmavo["Lojban"].isin(not_filtered_rasfi_gismu["Lojban Gismu"])]
    filtered_cmavo = pd.concat([filtered_rasfi_cmavo, found_cmavo]).drop_duplicates()
    
    # Create lujvo from rafsi combinations
    print("Creating lujvo from rafsi combinations...")
    df = create_lujvo(filtered_lujvo_rasfi)
    
    # Define predefined word lists
    wrong_lujvo_df = [
        {"Lojban": "kurjysi'u", "Lojban Composition": "kurji+simxu", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "frilyrai", "Lojban Composition": "frili+traji", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "fesybaktu", "Lojban Composition": "festi+baktu", "English": "b1 is a trash bin/trash can/recycle bin with contents b2=f1, made of material b3", "Type": "lujvo"},
        {"Lojban": "tarbykansa", "Lojban Composition": "tarbi+kansa", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "solnuncanci", "Lojban Composition": "solri+nu+canci", "English": "x1 is a sunset at location x2 as observed by x3", "Type": "lujvo"},
        {"Lojban": "tarbykansi'u", "Lojban Composition": "tarbi+kansa+simxu", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "seltcitygau", "Lojban Composition": "se+tcita+gasnu", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "toljinsa", "Lojban Composition": "to'e+jinsa", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "ruskygu'e", "Lojban Composition": "rusko+gugde", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "kurkydu'e", "Lojban Composition": "kurki+dukse", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "vimstizu", "Lojban Composition": "vikmi+stizu", "English": "s1 is a toilet for v1 to excrete v2 from source v3 via means/route v4", "Type": "lujvo"},
        {"Lojban": "kanryze'a", "Lojban Composition": "kanro+zenba", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "bardyrai", "Lojban Composition": "barda+traji", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "kajnyta'e", "Lojban Composition": "kajna+tanxe", "English": "t1=k2 is a cupboard for storing t2, made of t3, with shelves k1", "Type": "lujvo"},
        {"Lojban": "jdimyjdikygau", "Lojban Composition": "jdima+jdika+gasnu", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "snimycarvi", "Lojban Composition": "snime+carvi", "English": "No definition has been found", "Type": "lujvo"},
        {"Lojban": "zilfadni", "Lojban Composition": "zi'o+fadni", "English": "f2 (ka) is an ordinary / common / general / typical / usual property among f3.", "Type": "lujvo"}
    ]
    
    fuivla_list = [
        {"Lojban": "pelpeli", "English": "x1 is black pepper of species/variety/cultivar x2", "Type": "fu'ivla"},
        {"Lojban": "aidji", "English": "x1 intends to do/be x2 (ka)", "Type": "fu'ivla"},
        {"Lojban": "ekcala", "English": "x1 is a clown", "Type": "fu'ivla"},
        {"Lojban": "ckuliki", "English": "x1 is a mosquito of species/breed x2", "Type": "fu'ivla"},
        {"Lojban": "iumle", "English": "x1 is lovely/kawaii to x2 in aspect x3", "Type": "fu'ivla"},
        {"Lojban": "kancusu'oi", "English": "Something is such that both x1 and x2 are related to it by predicate x3", "Type": "fu'ivla"},
        {"Lojban": "bolgaro", "English": "x1 is Bulgarian in aspect x2", "Type": "fu'ivla"},
        {"Lojban": "relxima", "English": "x1 is a bicycle/two-wheeled vehicle", "Type": "fu'ivla"},
        {"Lojban": "ulmu", "English": "x1 is an elm of species/variety x2.", "Type": "fu'ivla"},
        {"Lojban": "akti", "English": "x1 is running service/in operation/performs functions x2", "Type": "fu'ivla"},
        {"Lojban": "elsa", "English": "x1 is a song", "Type": "fu'ivla"}
    ]
    
    experimental_gismu_list = [
        {"Lojban": "darca", "Lojban definition": "x1 arrives at x2 via route x3", "English": "arrives", "Type": "gismu"}
    ]
    
    cmevla_list = [
        {"Lojban": ".robert.", "Type": "cmevla"},
        {"Lojban": ".ianik.", "Type": "cmevla"},
        {"Lojban": ".zoros.", "Type": "cmevla"},
        {"Lojban": ".makax.", "Type": "cmevla"},
        {"Lojban": ".alis.", "Type": "cmevla"},
        {"Lojban": ".patrik.", "Type": "cmevla"},
        {"Lojban": ".xelen.", "Type": "cmevla"},
        {"Lojban": ".mark.", "Type": "cmevla"},
        {"Lojban": ".djein.", "Type": "cmevla"},
        {"Lojban": ".ualter.", "Type": "cmevla"},
        {"Lojban": ".kseni'as.", "Type": "cmevla"}
    ]
    
    # Combine all lujvo data
    final_df_lujvo = pd.concat([df, pd.DataFrame(wrong_lujvo_df), found_lujvo], ignore_index=True)
    
    # Process lujvo composition for additional gismu
    filtered_lujvo_gismu_list = []
    for i in final_df_lujvo["Lojban Composition"].to_list():
        for j in i.split("+"):
            filtered_lujvo_gismu_list.append(j)
    
    filtered_rasfi_lujvo = data_rafsi[data_rafsi["Lojban Gismu"].isin(filtered_lujvo_gismu_list)]
    filtered_lujvo_gismu = data_gismu[data_gismu["Lojban"].isin(filtered_lujvo_gismu_list)]
    final_filtered_gismu = pd.concat([filtered_gismu, filtered_lujvo_gismu]).drop_duplicates()
    
    final_rasfi = pd.concat([filtered_rasfi_lujvo, filtered_rasfi]).drop_duplicates()
    
    # Define output file names
    gismu_jsonl = "gismu_def.jsonl"
    lujvo_jsonl = "lujvo_def.jsonl"
    cmavo_jsonl = "cmavo_def.jsonl"
    fuivla_jsonl = "fuivla_def.jsonl"
    cmevla_jsonl = "cmevla_def.jsonl"
    rafsi_jsonl = "rafsi_def.jsonl"
    experimental_gismu_jsonl = "experimental_gismu_def.jsonl"
    
    # Generate JSONL files
    print("Generating JSONL files...")
    create_jsonl_defs(gismu_jsonl, final_filtered_gismu.to_dict(orient="records"), True)
    create_jsonl_defs(lujvo_jsonl, final_df_lujvo.to_dict(orient="records"), True)
    create_jsonl_defs(cmavo_jsonl, filtered_cmavo.to_dict(orient="records"), True)
    create_jsonl_defs(fuivla_jsonl, fuivla_list, True)
    create_jsonl_defs(cmevla_jsonl, cmevla_list, True)
    create_jsonl_defs(experimental_gismu_jsonl, experimental_gismu_list, True)
    create_jsonl_defs(rafsi_jsonl, final_rasfi.to_dict(orient="records"), True)
    
    # Calculate token count    
    print("Filtered JSONL files have been generated successfully:")
    print(f"- {gismu_jsonl}")
    print(f"- {lujvo_jsonl}")
    print(f"- {cmavo_jsonl}")
    print(f"- {fuivla_jsonl}")
    print(f"- {cmevla_jsonl}")
    print(f"- {experimental_gismu_jsonl}")
    print(f"- {rafsi_jsonl}")
    

if __name__ == "__main__":
    main()