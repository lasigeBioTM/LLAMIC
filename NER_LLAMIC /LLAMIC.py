import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import re
import textwrap
from llms import LLAMIC

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def split_text_preserving_words(query_row: str, max_length: int = 2636) -> list:
    """
    Splits the text into chunks with a maximum specified length while preserving whole words.
    """
    text = re.sub(r'\s+', ' ', query_row['hospital_course'])
    return textwrap.wrap(text, width=max_length)


def main(args):
    """
    Main function to process hospital course data and extract diseases and ICD-10 codes.
    """
    llamas = LLaMAS(args)
    data = pd.read_csv(args.data_path)

    results = []
    errors_total = []
    chunk_size = args.chunk_size 
    output_file = args.output_file
    errors_file = args.errors_file
    processed_ids = set()

    # Check if the output and errors files already exist.
    if not os.path.exists(errors_file):
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump({"errors": []}, f, indent=4, cls=NpEncoder)
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"results": []}, f, indent=4, cls=NpEncoder)

    # Load previously processed IDs to avoid reprocessing.
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            processed_ids.update([result["id"] for result in existing_data.get("results", [])])
    except (json.JSONDecodeError, KeyError):
        pass
    try:
        with open(errors_file, "r", encoding="utf-8") as f:
            existing_errors = json.load(f)
            processed_ids.update([error["id"] for error in existing_errors.get("errors", [])])
    except (json.JSONDecodeError, KeyError):
        pass

    all_ids = set(data['hadm_id'])

    # Load specific IDs from a list file if provided.
    if args.list_ids:
        with open(args.list_ids, "r", encoding="utf-8") as f:
            data_ids = json.load(f)
            target_id = [key for key, value in data_ids.items() if value == "test"]
            print("Data IDs", len(target_id))

    if target_id:
        target_set = set(map(str, target_id))
        mask = data['hadm_id'].astype(str).isin(target_set)
        data = data.loc[mask]
        all_ids = set(data['hadm_id'])

    pending_ids = list(all_ids - processed_ids)
    random.shuffle(pending_ids)
    print('Number of all IDs:', len(all_ids))

    # Limit the number of notes to process
    if args.num_notes > 0:
        pending_ids = pending_ids[:args.num_notes]

    for hadm_id in tqdm(pending_ids, desc="Processing IDs"):
        query_row = data[data['hadm_id'] == hadm_id].iloc[0]
        query_chunks = split_text_preserving_words(query_row)

        diseases_true = query_row['diseases_ICDs']
        predicted_diseases_chunks = []
        error = {hadm_id: []}

        for chunk in query_chunks:
            if not chunk.endswith('.'):
                chunk += '.'
            result, errors = llamas.call(chunk)
            predicted_diseases_chunks.append(json.loads(result)["results"])
            error[hadm_id].extend(errors)
            
            print(result)

        final_predicted_diseases = [item for sublist in predicted_diseases_chunks for item in sublist]

        results.append({
            "id": int(hadm_id),
            "diseases_true": diseases_true,
            "diseases_predicted": final_predicted_diseases
        })
        errors_total.append({
            "id": int(hadm_id),
            "errors": error[hadm_id]
        })

        # Save results and errors periodically to prevent data loss
        if len(results) % chunk_size == 0:
            with open(output_file, "r+", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {"results": []}

                existing_data["results"].extend(results)
                f.seek(0)
                json.dump(existing_data, f, indent=4, cls=NpEncoder)
                f.truncate()
            results = []

        if len(errors_total) % chunk_size == 0:
            with open(errors_file, "r+", encoding="utf-8") as f:
                try:
                    existing_errors = json.load(f)
                except json.JSONDecodeError:
                    existing_errors = {"errors": []}
                existing_errors["errors"].extend(errors_total)
                f.seek(0)
                json.dump(existing_errors, f, indent=4)
                f.truncate()
            errors_total = []

    # Save any remaining data
    if results:
        with open(output_file, "r+", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {"results": []}
            existing_data["results"].extend(results)
            f.seek(0)
            json.dump(existing_data, f, indent=4, cls=NpEncoder)
            f.truncate()

    if errors_total:
        with open(errors_file, "r+", encoding="utf-8") as f:
            try:
                existing_errors = json.load(f)
            except json.JSONDecodeError:
                existing_errors = {"errors": []}
            existing_errors["errors"].extend(errors_total)
            f.seek(0)
            json.dump(existing_errors, f, indent=4)
            f.truncate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process hospital course data to extract diseases and ICD-10 codes."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the JSON config file containing all parameters."
    )

    #Extract the arguments
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    llm_name = config["llm_name"]
    data_path = config["data_path"]
    output_file = config["output_file"]
    errors_file = config["errors_file"]
    chunk_size = config["chunk_size"]
    num_notes = config["num_notes"]
    list_ids = config["list_ids"]

    args = argparse.Namespace(
        llm_name=llm_name,
        data_path=data_path,
        output_file=output_file,
        errors_file=errors_file,
        chunk_size=chunk_size,
        num_notes=num_notes,
        list_ids=list_ids
    )

    main(args)
