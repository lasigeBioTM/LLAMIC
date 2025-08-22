import os
import re
import ast
import json
import random
import argparse
import textwrap

import numpy as np
import pandas as pd
from tqdm import tqdm

from llms import LLAMIC
from eval import EvaluatorNER_NEL

class Lexicon:
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Lexicon file not found: {filepath}")
        try:
            self.lexicon = self._load_lexicon(filepath)
        except ValueError:
            raise ValueError("Lexicon file must contain 'terminology' and 'description' columns.")

    def _load_lexicon(self, filepath: str) -> dict:
        df = pd.read_csv(filepath, usecols=['terminology', 'description'])
        df['terminology'] = df['terminology'].astype(str).str.strip()
        df['description'] = df['description'].astype(str).str.strip()
        return dict(zip(df['terminology'], df['description']))

    def get_description(self, code: str) -> str:
        return self.lexicon.get(code.strip(), "Unknown")

   
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def split_text_preserving_words(query_row: str, max_length: int = 2636) -> list:
    return textwrap.wrap(text, width=max_length)

def main(args):
    lexicon = Lexicon(args.lexicon_path)
    llamas = LLAMIC(args, lexicon)
    if args.debug:
        print(f"Running in {args.run_mode} mode with entity type: {args.entity_type}", flush=True)
    try:         # atualizar no github!!
        data = pd.read_csv(args.input_file, compression='gzip')
    except Exception as e:
        data = pd.read_csv(args.input_file)

    if args.debug:
        print(f"Loaded {len(data)} records from {args.input_file}", flush=True)
        
    results = []
    errors_total = []

    chunk_size = args.save_chunk_size
    output_file = os.path.join(args.output_dir, "results.json")
    errors_file = os.path.join(args.output_dir, "errors.json")
    processed_ids = set()
    if args.debug:
        print(f"Output file: {output_file}", flush=True)
        print(f"Errors file: {errors_file}", flush=True)

    for file_path, key in [(errors_file, "errors"), (output_file, "results")]:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({key: []}, f, indent=4, cls=NpEncoder)

    for file_path, key in [(output_file, "results"), (errors_file, "errors")]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                count = len(existing.get(key, []))
                processed_ids.update([item["id"] for item in existing.get(key, [])])
        except Exception as e:
            print(f"[WARNING] Failed to load {file_path}: {e}", flush=True)

    all_ids = set(data['id'])
    pending_ids = list(all_ids - processed_ids)
    random.shuffle(pending_ids)

    def split_text_preserving_words(text):
        text = re.sub(r'\s+', ' ', text)
        return textwrap.wrap(text, width=args.window_size)


    for i in tqdm(range(0, len(pending_ids), chunk_size), desc="Processing IDs in batch"):
        print(f"Processing batch {i // chunk_size + 1}", flush=True)
        batch_ids = pending_ids[i:i + chunk_size]
        batch_texts = []
        id_to_chunks = {}
        for hadm_id in batch_ids:
            row = data[data['id'] == hadm_id].iloc[0]
            chunks = split_text_preserving_words(row['documents'])
            chunks = [chunk if chunk.endswith('.') else chunk + '.' for chunk in chunks]

            if args.debug:
                print(f"\n[DEBUG] ID: {hadm_id} → {len(chunks)} chunks:", flush=True)
                for idx, chunk in enumerate(chunks):
                    print(f"  Chunk {idx + 1}: {chunk[:100]}...", flush=True)

            id_to_chunks[hadm_id] = chunks
            batch_texts.extend(chunks)

        offsets = {}
        cumulative_length = 0
        for idx, chunk in enumerate(batch_texts):
            offsets[idx] = cumulative_length
            cumulative_length += len(chunk)

        if args.debug:
            print("\n[DEBUG] Offsets:", flush=True)
            for idx, offset in offsets.items():
                print(f"  Chunk {idx}: Offset {offset}", flush=True)
        batch_results, batch_errors = llamas.call(batch_texts)
        if args.debug:
            print("\n[DEBUG] Batch results:", flush=True)
            print(batch_results, flush=True)
            print("\n[DEBUG] Batch errors:", flush=True)
            print(batch_errors, flush=True)

        flat_results = []
        for i, res in enumerate(batch_results):
            try:
                if isinstance(res, str):
                    parsed = json.loads(res)
                else:
                    parsed = res
                flat_results.append(parsed)
            except Exception as e:
                flat_results.append([])

        flat_results = [item for sublist in flat_results for item in sublist]
        flat_errors = batch_errors

        adjusted_results = []

        for idx, chunk_entities in enumerate(flat_results):
            chunk_offset = offsets[idx]
            adjusted_chunk_entities = []
            for ent in chunk_entities:
                if not isinstance(ent, dict):
                    continue
                ent['start'] = ent['start'] + chunk_offset
                ent['end'] = ent['end'] + chunk_offset
                adjusted_chunk_entities.append(ent)
            adjusted_results.append(adjusted_chunk_entities)


        result_idx = 0
        for hadm_id in batch_ids:
            chunk_count = len(id_to_chunks[hadm_id])
            chunk_results = adjusted_results[result_idx:result_idx + chunk_count]
            chunk_errors = flat_errors[result_idx:result_idx + chunk_count]
            result_idx += chunk_count

            predicted_entities = []
            for res in chunk_results:
                if isinstance(res, list):
                    predicted_entities.extend(res)
                else:
                    predicted_entities.append(res)

            if args.entity_type == "drug":
                results.append({
                    "id": int(hadm_id),
                    "drugs_predicted": predicted_entities
                })
            else:
                results.append({
                    "id": int(hadm_id),
                    "diseases_predicted": predicted_entities
                })

            errors_total.append({
                "id": int(hadm_id),
                "errors": chunk_errors
            })

        # Save intermediate results
        if len(results) >= chunk_size:
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

        if len(errors_total) >= chunk_size:
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

    if args.save_annotations.lower() == "yes":
        with open(output_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)['results']

        true_entities_map = {
            int(row['id']): ast.literal_eval(row['entities'])
            for _, row in data.iterrows()
        }

        list_predicted = []
        list_true = []

        for result in results_data:
            hadm_id = int(result['id'])
            if args.entity_type == "drug":
                predicted_entities = result['drugs_predicted']
            else:
                predicted_entities = result['diseases_predicted']
            
            true_entities = true_entities_map.get(hadm_id, [])

            list_predicted.append(predicted_entities)
            list_true.append(true_entities)

        if args.debug:
            print("\n[DEBUG] Final results:", flush=True)
            print("Predicted entities:", list_predicted, flush=True)
            print("True entities:", list_true, flush=True)
        evaluator = EvaluatorNER_NEL(list_predicted, list_true, args.entity_type)
        evaluator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="LLAMIC Named Entity Recognition and Linking Pipeline for Clinical Texts"
    )

    # Model paths
    parser.add_argument("--model_name_or_path_ner", type=str, required=True, help="Path to the NER model.")
    parser.add_argument("--model_name_or_path_nel", type=str, required=True, help="Path to the NEL model.")
    parser.add_argument("--model_name_or_path_review", type=str, required=True, help="Path to the review/classifier model.")

    # Data and processing
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--lexicon_path", type=str, required=True, help="Path to the domain-specific lexicon.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where output will be saved.")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging.')
    parser.add_argument("--n_iterations", type=int, default=1, help="Number of iterations for lexicon enhancement.")
    parser.add_argument("--window_size", type=int, default=2636, help="Sliding window size for long document splitting.")
    parser.add_argument("--max_input_tokens", type=int, default=1024, help="Maximum input tokens for the model.")
    parser.add_argument("--entity_type", type=str, choices=["disease", "drug"], default="disease", help="Entity type to process: disease or drug.")

    # Execution control
    parser.add_argument("--run_mode", type=str, choices=["PREDICT", "EVAL"], default="PREDICT", help="Execution mode: prediction or evaluation.")
    parser.add_argument("--save_annotations", type=str, choices=["yes", "no"], default="no", help="Whether to save annotated outputs.")
    parser.add_argument("--save_chunk_size", type=int, default=200, help="Chunk size for intermediate saves.")


    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
