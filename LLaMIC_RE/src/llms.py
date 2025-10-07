import argparse
import json
import logging
import os
import random
import re
from typing import Tuple

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from llamic import TripleGenerator
from eval import RelationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMBase:
    MODEL_MAP = {
        "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "txgemma": "google/txgemma-9b-chat"
    }

    def __init__(self, llm_name: str) -> None:
        self.llm_name = llm_name.lower()
        self.base_model_name = self.MODEL_MAP.get(self.llm_name, llm_name)

        logger.info(f"Initializing model '{self.base_model_name}' for LLM '{self.llm_name}'")
        self.tokenizer, self.model = self._initialize_llm()

    def _initialize_llm(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        tokenizer = self._load_tokenizer()
        model = self._load_model(tokenizer)
        return tokenizer, model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
            logger.debug(f"Tokenizer loaded with pad_token_id={tokenizer.pad_token_id}")
            return tokenizer
        except Exception as e:
            logger.exception(f"Failed to load tokenizer for '{self.base_model_name}': {e}")
            raise RuntimeError(f"Could not initialize tokenizer for '{self.base_model_name}'") from e

    def _load_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                llm_int8_enable_fp32_cpu_offload=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                torch_dtype="float16",
                quantization_config=bnb_config,
            )
            logger.debug("Loaded causal language model with 4-bit quantization.")

            model.config.rope_scaling = {"factor": 2.0, "type": "linear"}
            logger.debug("Set rope_scaling configuration.")

            model.config.pad_token_id = tokenizer.pad_token_id
            logger.debug(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")

            return model
        except Exception as e:
            logger.exception(f"Failed to load model for '{self.base_model_name}': {e}")
            raise RuntimeError(f"Could not initialize model for '{self.base_model_name}'") from e

    def format_prompt(self, query: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and "txgemma" in self.base_model_name:
            messages = [{"role": "user", "content": query}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return query

    def generate(self, query: str, max_new_tokens: int, relation_list: list = None) -> str:
        formatted_query = self.format_prompt(query)

        try:
            inputs = self.tokenizer(
                formatted_query,
                return_tensors="pt",
                truncation=True,
                max_length=512 if "sequenceclassification" in self.model.__class__.__name__.lower() else 1024,
            )

            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

        except Exception as e:
            logger.exception("Error tokenizing the query: %s", e)
            raise

        try:
            if "sequenceclassification" in self.model.__class__.__name__.lower():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                return relation_list[pred_idx]
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response_tokens = outputs[0][input_ids.shape[-1]:]
                generated_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                return generated_text

        except Exception as e:
            logger.exception("Error generating response: %s", e)
            raise

def save_json(file_path: str, key: str, items: list) -> None:
    if not os.path.exists(file_path):
        data = {key: []}
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {key: []}
    data.setdefault(key, []).extend(items)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_processed_ids(file_path: str, key: str) -> set:
    if not os.path.exists(file_path):
        return set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(item.get("id") for item in data.get(key, []))
    except (json.JSONDecodeError, KeyError):
        return set()


def main(args: argparse.Namespace) -> None:
    llm_pairs = args.model_name_or_path_pg
    llm_relations = args.model_name_or_path_rc
    run_mode = args.run_mode

    if not llm_relations:
        logger.error("Chave 'llm_name_relations' é obrigatória no config.")
        return

    llamas_p = LLMBase(llm_pairs) if llm_pairs else None
    llamas_t = llamas_p if llm_pairs == llm_relations else LLMBase(llm_relations)

    triple_extractor = TripleGenerator(llamas_p, llamas_t, device='auto')

    data = pd.read_csv(args.input_file)

    processed_ids = load_processed_ids(args.output_file, "results")

    pending_ids = list(set(data["id"]) - processed_ids)
    random.shuffle(pending_ids)
    results, errors_total = [], []

    for id in tqdm(pending_ids, desc="Processing IDs"):
        row = data[data["id"] == id].iloc[0]
        text = row["documents"]
        if len(re.findall(r"\w+", text)) > 600:
            logger.warning(f"Skipping {id} due to length > 600 words.")
            results.append({"id": id, "documents": text, "relations": None})
            errors_total.append({"id": id, "errors": "Note too long"})
            continue

        if llamas_p:
            pairs, err_p = triple_extractor.generate_pairs(text)
        else:
            pairs, err_p = triple_extractor.generate_pairs_perm(text)

        relations, err_t = triple_extractor.generate_labels(text, pairs)

        results.append({"id": id, "relations": relations})
        errors_total.append({"id": id, "errors": [err_p, err_t]})

        if len(results) >= args.save_chunk_size:
            save_json(args.output_file, "results", results)
            results.clear()

        if len(errors_total) >= args.save_chunk_size:
            save_json(args.errors_file, "errors", errors_total)
            errors_total.clear()

    if results:
        save_json(args.output_file, "results", results)
    if errors_total:
        save_json(args.errors_file, "errors", errors_total)

    if run_mode == "EVAL":
        gold_data = data
        with open(args.output_file, "r", encoding="utf-8") as f:
            pred_data = json.load(f)['results']
        evaluator = RelationEvaluator(min_support=10)
        evaluator.evaluate(gold_data, pred_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process hospital course data to extract disease-drug relations."
    )
    parser.add_argument("--model_name_or_path_pg", type=str, required=False, help="Path to the model for pair generation. If not provided, will use the same model as for relation classification.")
    parser.add_argument("--model_name_or_path_rc", type=str, required=True, help="Path to the model for relation classification.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file with hospital course data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file with extracted relations.")
    parser.add_argument("--errors_file", type=str, default="errors.json", help="Path to save the JSON file with errors.")
    parser.add_argument("--save_chunk_size", type=int, default=100, help="Number of results to save in each chunk.")
    parser.add_argument("--run_mode", type=str, choices=["EVAL", "PREDICT"], default="PREDICT", help="Mode to run the script: EVAL or PREDICT.")
    args = parser.parse_args()

    main(args)

