import argparse
import json
import pandas as pd
import ast
import re
import os
import torch
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
from promptTemplate import generation_di_di_prompt, generation_dr_dr_prompt, generation_di_dr_prompt, prompt_label_first, prompt_label_second, prompt_label_relaxed
from label_description import description_di_di_each_label, description_di_dr_each_label, description_dr_dr_each_label



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
            for prompt, label in zip(examples['documents'], examples['relations'])
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

        tokenized["relations"] = tokenized["input_ids"].clone().detach()
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
            remove_columns=['documents', 'relations']
        )
        tokenized_eval = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['documents', 'relations']
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
            run_name=f"{self.params['wandb_run_name']}_{option}", ##
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
        self.tokenizer.save_pretrained(f"{self.params['output_dir']}_{option}") ##


def main():
    parser = argparse.ArgumentParser(description="Test prompts generation in JSON format")
    
    # Dataset args
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    
    # Model args
    parser.add_argument("--llm_name", type=str, required=True)

    # Training args
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
    
    # LoRA args
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Wandb
    parser.add_argument("--wandb_key", type=str, required=True)
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Debug
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    train_df = pd.read_csv(args.train_dataset_path)
    eval_df = pd.read_csv(args.eval_dataset_path)

    # Normalize disease tags
    def normalize_disease_tag(text):
        match = re.search(r'(\d+)>.*?</disease\1>', text)
        if match:
            return f'<disease{match.group(1)}>'
        return text

    # Normalize drug tags
    def normalize_drug_tag(text):
        match = re.search(r'(\d+)>.*?</drug\1>', text)
        if match:
            return f'<drug{match.group(1)}>'
        return text
    
    # Convert dataset to JSON format
    def convert_to_json(data, phase):
        json_list = []
        if hasattr(data, "iterrows"):
            rows = data.to_dict(orient="records")
        else:
            rows = data

        for row in rows:
            relations = row.get("relations", [])
            if isinstance(relations, str):
                try:
                    relations = ast.literal_eval(relations)
                except:
                    relations = []
            if not isinstance(relations, list):
                relations = [relations]

            if phase == "pg":
                sentence = row["documents"]
                list_diseases = sorted(set(re.findall(r"(<disease\d+>)", sentence)))
                list_drugs = sorted(set(re.findall(r"(<drug\d+>)", sentence)))

                triples_disease_disease = []
                triples_disease_drug = []
                triples_drug_drug = []

                # Create relation triples based on entity types
                for triplet in relations:
                    if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
                        continue
                    h, _, t = triplet
                    if "<disease" in h and "<disease" in t:
                        h = normalize_disease_tag(h)
                        t = normalize_disease_tag(t)
                        triples_disease_disease.append([h, t])
                    elif "<disease" in h and "<drug" in t:
                        h = normalize_disease_tag(h)
                        t = normalize_drug_tag(t)
                        triples_disease_drug.append([h, t])
                    elif "<drug" in h and "<disease" in t:
                        h = normalize_drug_tag(h)
                        t = normalize_disease_tag(t)
                        triples_disease_drug.append([h, t])
                    elif "<drug" in h and "<drug" in t:
                        h = normalize_drug_tag(h)
                        t = normalize_drug_tag(t)
                        triples_drug_drug.append([h, t])

                # Generate prompts for disease-disease pairs
                if len(list_diseases) > 1:
                    hospital_course = generation_di_di_prompt.format(
                        result_s=sentence, list_diseases=', '.join(list_diseases)
                    )
                    relations_json = json.dumps({"Pairs": triples_disease_disease}, ensure_ascii=False)
                    json_list.append({"documents": hospital_course, "relations": relations_json})

                # Generate prompts for disease-drug pairs
                if len(list_diseases) > 0 and len(list_drugs) > 0:
                    hospital_course = generation_di_dr_prompt.format(
                        result_s=sentence, list_diseases=', '.join(list_diseases), list_drugs=', '.join(list_drugs)
                    )
                    relations_json = json.dumps({"Pairs": triples_disease_drug}, ensure_ascii=False)
                    json_list.append({"documents": hospital_course, "relations": relations_json})

                # Generate prompts for drug-drug pairs
                if len(list_drugs) > 1:
                    hospital_course = generation_dr_dr_prompt.format(
                        result_s=sentence, list_drugs=', '.join(list_drugs)
                    )
                    relations_json = json.dumps({"Pairs": triples_drug_drug}, ensure_ascii=False)
                    json_list.append({"documents": hospital_course, "relations": relations_json})
            else:
                # Function to find entity text by tag
                def find_entity_name_by_tag_in_sentence(tag, sentence):
                    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>")
                    match = pattern.search(sentence)
                    if match:
                        return f"<{tag}>{match.group(1)}</{tag}>"
                    else:
                        raise ValueError(f"Entity with tag {tag} not found in sentence.")
                
                # Get the key of the entity from all entities
                def get_entity_key(entity, all_entities):
                    if entity in all_entities:
                        return entity
                    for k, v in all_entities.items():
                        if v == entity:
                            return k
                    return ""

                sentence = row["documents"]
                if isinstance(row["relations"], str):
                    triplets = ast.literal_eval(row["relations"])
                elif isinstance(row["relations"], list):
                    triplets = row["relations"]
                else:
                    triplets = []

                # Map tags to entity names
                all_entities = {
                    tag: f"<{tag}>{match.group(1)}</{tag}>"
                    for tag in set(re.findall(r"(disease\d+|drug\d+)", sentence))
                    if (match := re.search(rf"<{tag}>(.*?)</{tag}>", sentence))
                }
                print(f"All entities: {all_entities}")

                # Iterate over triplets to create prompts
                for triplet in triplets:
                    print("Triplet", triplet)
                    new_query = sentence

                    key1 = get_entity_key(triplet[0], all_entities)
                    key2 = get_entity_key(triplet[2], all_entities)

                    entity1_name = all_entities.get(key1, "")
                    entity2_name = all_entities.get(key2, "")

                    print(f"Entity1 (key={key1}): {entity1_name}")
                    print(f"Entity2 (key={key2}): {entity2_name}")

                    # Replace other entities with blanks
                    for tag, entity in all_entities.items():
                        if tag not in (key1, key2):
                            new_query = new_query.replace(entity, "___")
                    print(f"Original sentence: {sentence}")
                    print(f"New query: {new_query}")

                    # Choose relation type and description based on entities
                    if "disease" in triplet[0] and "disease" in triplet[2]:
                        relation_list = ['related to', 'comorbidity', 'no relation', 'other']
                        description_each_label = description_di_di_each_label

                    elif ("disease" in triplet[0] and "drug" in triplet[2]) or \
                         ("drug" in triplet[0] and "disease" in triplet[2]):
                        relation_list = ['indicated', 'no_indicated', 'discontinued', 'no relation', 'other']
                        description_each_label = description_di_dr_each_label

                    else:
                        relation_list = ['related to', 'combined with', 'replaced by', 'no relation', 'other']
                        description_each_label = description_dr_dr_each_label

                    # Create prompt for LLM
                    prompt = prompt_label_first.replace("{relation_list}", ", ".join(relation_list))
                    for label, description in description_each_label.items():
                        prompt += f"- '{label}': {description}\n"

                    prompt += prompt_label_second \
                        .replace("{new_query}", new_query) \
                        .replace("{entity1_name}", f'<{key1}>') \
                        .replace("{entity2_name}", f'<{key2}>')

                    label_json = json.dumps({
                        "Label": [entity1_name, triplet[1], entity2_name]
                    }, indent=None, ensure_ascii=False)

                    json_list.append({"documents": prompt, "relations": label_json})

        return json_list



    # Filter sentences shorter than 600 characters
    train_df = train_df[train_df['documents'].str.len() < 600]
    eval_df = eval_df[eval_df['documents'].str.len() < 600]

    # Convert to JSON for prompt generation
    train_json = convert_to_json(train_df, phase="pg")
    eval_json = convert_to_json(eval_df, phase="pg")

    print("Train examples (first 3):")
    for ex in train_json[:8]:
        print("==================")
        print(json.dumps(ex, indent=2, ensure_ascii=False))

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_json))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_json))
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))
    print("Example train data:", train_dataset[0])
    for i in range(100):
        print(f"Train example {i}: {train_dataset[i]['documents']}")
        print(f"relations: {train_dataset[i]['relations']}")

    # Initialize LLM and fine-tune
    llm = LLMBase(vars(args))
    llm.fine_tune(train_dataset, eval_dataset, vars(args), option="pg")


    # =================== RC =================    
    train_json = convert_to_json(train_df, phase="rc")
    eval_json = convert_to_json(eval_df, phase="rc")

    print("Train examples (first 3):")
    for ex in train_json[:8]:
        print("========RC==========")
        print(json.dumps(ex, indent=2, ensure_ascii=False))


    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_json))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_json))
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))
    print("Example train data:", train_dataset[0])
    for i in range(100):
        print(f"Train example {i}: {train_dataset[i]['documents']}")
        print(f"relations: {train_dataset[i]['relations']}")
    llm = LLMBase(vars(args))   ##
    llm.fine_tune(train_dataset, eval_dataset, vars(args), option="rc")


if __name__ == "__main__":
    main()

