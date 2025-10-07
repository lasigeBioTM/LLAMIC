import argparse
import json
import pandas as pd
import torch
import ast
import re
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import wandb
from promptTemplate import generation_prompt_template, generate_icd_per_disease_prompt, generate_id_per_entity_drug_prompt, generation_drug_prompt_template



class LLMBase:
    def __init__(self, params):
        self.llm_name = params["llm_name"]
        self.params = params
        self.base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" if 'llama3' in self.llm_name.lower() else self.llm_name
        self.tokenizer, self.model = self.initialize_llm()

    def initialize_llm(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.use_cache = False
        model.enable_input_require_grads()
        return tokenizer, model


    def tokenize_function(self, examples):

        texts = [
            f"{prompt}\n```\n{json.dumps(json.loads(label), indent=4, ensure_ascii=False).replace('\"[(', '(').replace(')]\"', ')').replace('\'', '\"')}\n```"
            for prompt, label in zip(examples['documents'], examples['entities'])
        ]

        print("TEXT:\n\n", texts[0])
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )


        tokenized["entities"] = tokenized["input_ids"].clone().detach()

        return tokenized


    def fine_tune(self, train_dataset, eval_dataset, params, option):

        wandb.login(key=params["wandb_key"])
        config = LoraConfig(
            r=self.params["lora_r"],
            lora_alpha=self.params["lora_alpha"],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.params["lora_dropout"],
            task_type="CAUSAL_LM",
            bias="none"
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['documents', 'entities']
        )
        tokenized_eval = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['documents', 'entities']
        )

        training_args = TrainingArguments(
            output_dir=self.params["output_dir"] + option,
            per_device_train_batch_size=self.params["per_device_train_batch_size"],
            per_device_eval_batch_size=self.params["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.params["gradient_accumulation_steps"],
            num_train_epochs=self.params["num_epochs"],
            learning_rate=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
            fp16=self.params["fp16"],
            optim=self.params["optim"],
            logging_steps=self.params["logging_steps"],
            evaluation_strategy=self.params["evaluation_strategy"],
            eval_steps=self.params["eval_steps"],
            save_strategy=self.params["save_strategy"],
            save_steps=self.params["save_steps"],
            report_to="wandb" if self.params["use_wandb"] else None,
            run_name=self.params["wandb_run_name"] + option,
            gradient_checkpointing=self.params["gradient_checkpointing"],
            lr_scheduler_type=self.params["lr_scheduler_type"],
            warmup_ratio=self.params["warmup_ratio"],
            max_grad_norm=self.params["max_grad_norm"],
            push_to_hub=self.params["push_to_hub"]
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval, 
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        torch.cuda.empty_cache()
        trainer.train()
        self.tokenizer.save_pretrained(self.params["output_dir"])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on clinical NER task")
    
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--entity_type", type=str, choices=["disease", "drug"], required=True, help="Type of entity to fine-tune on")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_run_name", type=str, default="llama3_finetuning")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--wandb_key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_dataset_path)
    eval_df = pd.read_csv(args.eval_dataset_path)

    import ast

    import ast
    import json

    def convert_to_json(data, type_entity, phase):
        json_list = []

        if hasattr(data, "iterrows"):
            rows = data.to_dict(orient="records")
        else:
            rows = data

        for row in rows:
            entities = row.get("entities", [])
            if isinstance(entities, str):
                try:
                    entities = ast.literal_eval(entities)
                except:
                    entities = []
            if not isinstance(entities, list):
                entities = [entities]

            if phase == "ner":
                if type_entity == "disease":
                    hospital_course = generation_prompt_template.replace('{text}', row["documents"])
                    entities_dict = {
                        "cvd_terminologies": [
                            ent['entity'] if isinstance(ent, dict) else ent
                            for ent in entities
                        ]
                    }
                else:
                    hospital_course = generation_drug_prompt_template.replace('{text}', row["documents"])
                    entities_dict = {
                        "drugs_terminologies": [
                            ent['entity'] if isinstance(ent, dict) else ent
                            for ent in entities
                        ]
                    }

            elif phase == "nel":
                if type_entity == "disease":
                    hospital_course = generate_icd_per_disease_prompt.replace('{text}', row["documents"])
                    entities_dict = {
                        "Pairs": [
                            (ent['entity'], ent.get('icd')) if isinstance(ent, dict) else (ent, None)
                            for ent in entities
                        ]
                    }
                else:
                    hospital_course = generate_id_per_entity_drug_prompt.replace('{text}', row["documents"])
                    entities_dict = {
                        "Pairs": [
                            (ent['entity'], ent.get('mesh')) if isinstance(ent, dict) else (ent, None)
                            for ent in entities
                        ]
                    }

            entities_json = json.dumps(entities_dict, ensure_ascii=False)

            json_list.append({
                "documents": hospital_course,
                "entities": entities_json
            })

        return json_list

    def entity_in_text(text, entity):
        entity_regex = re.escape(entity)
        pattern = rf'(?<!\w){entity_regex}(?!\w)'
        return re.search(pattern, text) is not None


    def chunk_text_and_entities(df, chunk_size=512):

        chunked_data = []

        for idx, row in df.iterrows():
            text = row['documents']
            entities = row.get('entities', [])
            entities = ast.literal_eval(entities) if isinstance(entities, str) else entities
            n = len(text)
            found_entities = set()
            
            # Cria chunks
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = text[start:end]
                
                # Encontra entidades neste chunk
                chunk_entities = []
                if not entities:
                    chunked_data.append({
                        'documents': chunk,
                        'entities': []
                    })
                    continue

                for ent in entities:
                    if isinstance(ent, dict):
                        ent_text = ent['entity']
                    else:
                        ent_text = ent

                    if entity_in_text(chunk, ent_text):
                        if isinstance(ent, dict):
                            chunk_entities.append(ent)
                        else:
                            chunk_entities.append({"entity": ent_text})
                        found_entities.add(ent_text)

                chunked_data.append({
                    'documents': chunk,
                    'entities': chunk_entities
                })

            if entities:
                missing_entities = []
                for ent in entities:
                    ent_text = ent['entity'] if isinstance(ent, dict) else ent
                    if ent_text not in found_entities:
                        missing_entities.append(ent_text)

                if missing_entities:
                    print(f"[WARNING] Document {idx} - entities not found in any chunk:", missing_entities)

        return chunked_data


    train_df_chunked = chunk_text_and_entities(train_df, chunk_size=2636)
    eval_df_chunked = chunk_text_and_entities(eval_df, chunk_size=2636)

    train_json = convert_to_json(train_df_chunked, args.entity_type, phase="ner")
    eval_json = convert_to_json(eval_df_chunked, args.entity_type, phase="ner")

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_json))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_json))
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))
    print("Example train data:", train_dataset[0])
    for i in range(2):
        print(f"Train example {i}: {train_dataset[i]['documents']}")
        print(f"Entities: {train_dataset[i]['entities']}")

    llm = LLMBase(vars(args))
    llm.fine_tune(train_dataset, eval_dataset, vars(args), option ="ner")

    train_json = convert_to_json(train_df_chunked, args.entity_type, phase="nel")
    eval_json = convert_to_json(eval_df_chunked, args.entity_type, phase="nel")

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_json))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_json))
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))
    print("Example train data:", train_dataset[0])

    for i in range(2):
        print(f"Train example {i}: {train_dataset[i]['documents']}")
        print(f"Entities: {train_dataset[i]['entities']}")

    Inicializa LLM e fine-tune
    llm = LLMBase(vars(args))
    llm.fine_tune(train_dataset, eval_dataset, vars(args), option ="nel")


if __name__ == "__main__":
    main()





