import json
import logging
from typing import List
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from generate import Generate
from review import Review
import torch

class LLMBase:
    def __init__(self, llm_name: str) -> None:
        self.llm_name = llm_name.lower()
        self.base_model_name = (
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
            if self.llm_name in ['llama3.1', 'llama3']
            else llm_name
        )
        print(f"Loading model: {self.base_model_name}")

        self.tokenizer, self.model = self._initialize_llm()

    def _initialize_llm(self):
        try:
            torch.cuda.empty_cache()  # ====
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                torch_dtype="float16",
                quantization_config=bnb_config,
            )

            model.config.rope_scaling = {"factor": 2.0, "type": "linear"}
            model.config.use_cache = False

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"

            return tokenizer, model

        except Exception as e:
            logger.error(f"Failed to load model '{self.base_model_name}': {e}")
            raise

    def generate(self, document: str, max_input_tokens: int, max_new_tokens: int) -> str:
        try:
            input_ids = self.tokenizer(
                document,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,  # 1024
            ).input_ids.to(self.model.device)
        except Exception as e:
            logger.exception("Error tokenizing the document: %s", e)
            raise

        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            logger.exception("Error generating response: %s", e)
            raise

        response_tokens = outputs[0][input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return generated_text
    
    def generate_batch(self, documents: List[str], max_input_tokens: int, max_new_tokens: int) -> List[str]:
        try:
            inputs = self.tokenizer(
                documents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(self.model.device)
        except Exception as e:
            logger.exception("Error tokenizing the batch of documents: %s", e)
            raise

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            logger.exception("Error generating responses: %s", e)
            raise

        responses = []
        for i, output in enumerate(outputs):
            response_tokens = output[inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(generated_text)

        return responses


class LLAMIC:
    def __init__(self, args, lexicon):
        self.args = args
        self.lexicon = lexicon
        self.n_iterations = args.n_iterations
        self.max_input_tokens = args.max_input_tokens
        self.window_size = args.window_size
        model_cache = {}

        # NER
        ner_path = args.model_name_or_path_ner
        if ner_path not in model_cache:
            model_cache[ner_path] = LLMBase(ner_path)
        self.llm_ner = model_cache[ner_path]

        # NEL
        nel_path = args.model_name_or_path_nel
        if nel_path not in model_cache:
            model_cache[nel_path] = LLMBase(nel_path)
        self.llm_nel = model_cache[nel_path]

        # REVIEW
        review_path = args.model_name_or_path_review
        if review_path not in model_cache:
            model_cache[review_path] = LLMBase(review_path)
        self.llm_review = model_cache[review_path]

        self.pairs_generator = Generate(self.llm_ner, self.llm_nel, args.max_input_tokens)
        self.review = Review(self.llm_review, lexicon, args.max_input_tokens, "cuda")

    def call(self, documents):
        generated_pairs, icd_errors, disease_errors = self.pairs_generator.call(documents, self.n_iterations)
        review_results_list = [[] for _ in range(len(documents))]
        errors_list = [[] for _ in range(len(documents))]

        for idx, (doc, pairs) in enumerate(zip(documents, generated_pairs)):
            if pairs.strip():
                review_result, review_error = self.review.call(doc, pairs)
            else:
                review_result = json.dumps([], indent=4)
                review_error = ["No pair found for review"]

            review_results_list[idx].append(review_result)
            errors_list[idx].append(review_error)

        return review_results_list, [disease_errors, icd_errors, errors_list]

