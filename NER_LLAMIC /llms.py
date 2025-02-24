import json
import logging
from typing import Any, Dict, List, Tuple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from generate import Generate
from review import Review

class LLMBase:
    def __init__(self, llm_name: str) -> None:
        """
        Initializes the base LLM with the appropriate model.
        """
        self.llm_name = llm_name
        if llm_name.lower() in ['llama3.1', 'llama3']:
            self.base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            print("Model name:", self.base_model_name)
        else:
            self.base_model_name = llm_name
            print("Model name:", self.base_model_name)

        self.tokenizer, self.model = self.initialize_llm()

    def initialize_llm(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Initializes the model and tokenizer based on the model name.
        """
        try:
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
        except Exception as e:
            logger.exception("Error loading the model: %s", e)
            raise

        # Configure the model
        model.config.rope_scaling = {"factor": 2.0, "type": "linear"}
        model.config.use_cache = False
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        except Exception as e:
            logger.exception("Error loading the tokenizer: %s", e)
            raise
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

    def generate(self, query: str, max_new_tokens: int) -> str:
        """
        Generates a response from the LLM based on a query.
        """
        try:
            input_ids = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).input_ids.to(self.model.device)
        except Exception as e:
            logger.exception("Error tokenizing the query: %s", e)
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


class LLAMIC:
    def __init__(self, args: Any) -> None:
        """
        Initializes the LLAMIC for extracting diseases and ICD-10 codes.
        """
        self.agent_name = "LLAMIC"
        self.role = "Extract diseases and their respective ICD-10 codes from a given note."
        self.args = args

        self.llm = {
            "LLaMA_gen": LLMBase(args.llm_name['LLaMA_gen']),
            "LLaMA_genID": LLMBase(args.llm_name['LLaMA_genID']),
            "LLaMA_revi": LLMBase(args.llm_name['LLaMA_revi'])
        }

        self.pairs_generator = Generate(self.llm["LLaMA_gen"],self.llm["LLaMA_genID"], args)
        self.review = Review(self.llm["LLaMA_revi"], args)

    def call(self, query: str) -> Tuple[str, List[str]]:
        """
        Processes the query to generate and review pairs.
        """
        try:
            generated_pairs, er_p, er_d = self.pairs_generator.call(query)
        except Exception as e:
            logger.exception("Error generating pairs: %s", e)
            raise

        if generated_pairs.strip():
            try:
                review_results, er_r = self.review.call(query, generated_pairs)
            except Exception as e:
                logger.exception("Error reviewing generated pairs: %s", e)
                raise
        else:
            review_results = json.dumps({"results": []}, indent=4)
            er_r = "No pair found"

        return review_results, [er_d, er_p, er_r]
