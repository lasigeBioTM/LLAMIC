import re
import json
from typing import List, Tuple, Optional
from promptTemplate import generation_di_di_prompt, generation_dr_dr_prompt, generation_di_dr_prompt, prompt_label_first, prompt_label_second, prompt_label_relaxed
from label_description import description_di_di_each_label, description_di_dr_each_label, description_dr_dr_each_label

class TripleGenerator:
    def __init__(self, llm_p,llm_t, device):
        self.llm_p = llm_p
        self.llm_t = llm_t
        self.device = device
        
    def check_output(self, resp_text, listofdiseases, listofdrugs):
        try:
            resp_text = resp_text.replace("\n", "").replace("\r", "").replace(" ", "")
            resp_text = resp_text.replace("'", '"')

            pattern = r"\{.*?\}"
            matches = re.search(pattern, resp_text, re.DOTALL)
            
            if not matches:
                return None, "JSON pattern not found"
            
            json_str = matches.group(0)
            
            try:
                result_dict = json.loads(json_str)
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON: {str(e)}"
            
            if "Pairs" not in result_dict or not isinstance(result_dict["Pairs"], list):
                return None, "Invalid format: 'Pairs' key missing or not a list"
            
            valid_pairs = []
            error_count = 0
            
            for pair in result_dict["Pairs"]:
                if len(pair) != 2:
                    error_count += 1
                    continue
                
                entity1, entity2 = pair
                
                if (entity1 in listofdiseases and entity2 in listofdiseases) or \
                (entity1 in listofdrugs and entity2 in listofdrugs) or \
                (entity1 in listofdiseases and entity2 in listofdrugs) or \
                (entity1 in listofdrugs and entity2 in listofdiseases):
                    valid_pairs.append(pair)
                else:
                    error_count += 1
            
            return valid_pairs, error_count
        
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    def check_label_output(self, resp_text, entity1_name, entity2_name):
        try:
            resp_text = " ".join(resp_text.split())  
            resp_text = resp_text.replace("'", '"') 

            pattern = r"\{.*?\}"
            matches = re.search(pattern, resp_text, re.DOTALL)
            
            if not matches:
                return None, "JSON pattern not found"
            
            json_str = matches.group(0)
            
            try:
                result_dict = json.loads(json_str)
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON: {str(e)}"
            
            if "Label" not in result_dict or not isinstance(result_dict["Label"], (str, list)):
                return None, "Invalid format: 'Label' key missing or not a string/list"
            
            if isinstance(result_dict["Label"], list):
                if len(result_dict["Label"]) != 3:
                    return None, "Invalid format: 'Label' list should have 3 elements"
                if not all(entity in result_dict["Label"] for entity in [entity1_name, entity2_name]):
                    return None, "Entity names not found in the label"

            return result_dict["Label"], ""

        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    def check_label_output(self, resp_text, entity1_name, entity2_name):
        try:
            resp_text = " ".join(resp_text.split())
            resp_text = resp_text.replace("'", '"')

            pattern = r"\{.*?\}"
            matches = re.search(pattern, resp_text, re.DOTALL)
            
            if not matches:
                return None, "JSON pattern not found"
            
            json_str = matches.group(0)
            try:
                result_dict = json.loads(json_str)
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON: {str(e)}"
            
            if "Label" not in result_dict or not isinstance(result_dict["Label"], (str, list)):
                return None, "Invalid format: 'Label' key missing or not a string/list"
            
            if isinstance(result_dict["Label"], list):
                if len(result_dict["Label"]) != 3:
                    return None, "Invalid format: 'Label' list should have 3 elements"
                
                found_1 = any(entity1_name in part for part in result_dict["Label"])
                found_2 = any(entity2_name in part for part in result_dict["Label"])
                
                if not (found_1 and found_2):
                    if not any(re.match(r"<.*?>", part) for part in result_dict["Label"]):
                        return None, "Entity names not found in the label"
            
            return result_dict["Label"], ""

        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
        
    def label_each_pair(self, query, pairs):
        """
        For each pair, remove all other entities and replace them with '_____'.
        """
        results = []
        
        all_entities = re.findall(r"(<(disease|drug)\d+>.*?</(disease|drug)\d+>)", query)
        
        for entity1_tag, entity2_tag in pairs:
            new_query = query

            def extract_entity(query, tag):
                tag_name = tag.strip("<>")
                match = re.search(fr"<{tag_name}>(.*?)</{tag_name}>", query)
                return f"<{tag_name}>{match.group(1)}</{tag_name}>" if match else ""


            entity1_match = re.search(fr"{entity1_tag}(.*?)</{entity1_tag[1:]}>", query)
            entity2_match = re.search(fr"{entity2_tag}(.*?)</{entity2_tag[1:]}>", query)

            entity1_name, entity2_name = "", ""
            
            if entity1_match:
                entity1_name = entity1_match.group(1)
                new_query = new_query.replace(entity1_match.group(0), entity1_name)
            if entity2_match:
                entity2_name = entity2_match.group(1)
                new_query = new_query.replace(entity2_match.group(0), entity2_name)

            for entity in all_entities:
                if entity1_tag not in entity and entity2_tag not in entity:
                    new_query = new_query.replace(entity[0], "_____")

            entity1_name, entity2_name = extract_entity(query, entity1_tag), extract_entity(query, entity2_tag)
            new_query = re.sub(r"\s+", " ", new_query).strip()

            prompt = prompt_label_relaxed.replace("{new_query}", new_query).replace("{entity1_name}", entity1_name).replace("{entity2_name}", entity2_name)

            done = False
            count = 0
            result, error = None, None
            while not done and count < 3:
                count += 1
                print(prompt)
                resp_text = self.llm.generate(prompt, max_new_tokens=50)
                print(resp_text)
                result, error = self.check_label_output(resp_text, entity1_name, entity2_name)
                print(f"============= Result LABELING: {result}, Error: {error} ==============")
                if result:
                    done = True

            if result:
                results.append(result)

        return results, error

    def label_each_pair_strict(self, query, pairs):

        results = []
        all_entities = {
            f"<{tag}>": f"<{tag}>{match.group(1)}</{tag}>"
            for tag in set(re.findall(r"(disease\d+|drug\d+)", query))
            if (match := re.search(rf"<{tag}>(.*?)</{tag}>", query))
        }
        print(f"All entities: {all_entities}")

        for entity1_tag, entity2_tag in pairs:
            new_query = query

            def extract_entity(tag):
                return all_entities.get(tag, "")
            
            entity1_name = extract_entity(entity1_tag)
            print(f"Entity 1: {entity1_name}")
            entity2_name = extract_entity(entity2_tag)

            for tag, entity in all_entities.items():
                if tag not in (entity1_tag, entity2_tag):
                    new_query = new_query.replace(entity, "_____")

            print(f"Entity 1: {entity1_name}, Entity 2: {entity2_name}")
            print(f"New Query: {new_query}\n")

            if "<disease" in entity1_name and "<disease" in entity2_name:
                relation_list = ['related to', 'comorbidity', 'no relation', 'other']
                description_each_label = description_di_di_each_label

            elif "<disease" in entity1_name and "<drug" in entity2_name or "<drug" in entity1_name and "<disease" in entity2_name:
                relation_list = ['indicated', 'no_indicated', 'discontinued', 'no relation', 'other']
                description_each_label = description_di_dr_each_label

            else:
                relation_list = ['related to', 'combined with', 'replaced by', 'no relation', 'other']
                description_each_label = description_dr_dr_each_label

            prompt = prompt_label_first.replace("{relation_list}", ", ".join(relation_list))

            for label, description in description_each_label.items():
                prompt += f"- '{label}': {description}\n"
            prompt += prompt_label_second.replace("{new_query}", new_query).replace("{entity1_name}", entity1_name).replace("{entity2_name}", entity2_name)

            done = False
            count = 0
            result, error = None, None
            while not done and count < 3:
                count += 1
                resp_text = self.llm_t.generate(prompt, max_new_tokens=50)
                result, error = self.check_label_output(resp_text, entity1_name, entity2_name)
                if result:
                    done = True

            if result:
                results.append(result)
        
        return results, error

    def generate_pairs_perm(self, query):
        try:
            list_diseases = sorted(set(re.findall(r"(<disease\d+>)", query)))
            list_drugs = sorted(set(re.findall(r"(<drug\d+>)", query)))

            result = []
            def unique_pairs(entities):
                pairs = []
                n = len(entities)
                for i in range(n):
                    for j in range(i+1, n):
                        pairs.append([entities[i], entities[j]])
                return pairs

            if len(list_diseases) > 1:
                disease_disease_pairs = unique_pairs(list_diseases)
                result.extend(disease_disease_pairs)

            if len(list_drugs) > 1:
                drug_drug_pairs = unique_pairs(list_drugs)
                result.extend(drug_drug_pairs)

            disease_drug_pairs = [[d, dr] for d in list_diseases for dr in list_drugs]
            if disease_drug_pairs:
                result.extend(disease_drug_pairs)

            if not result:
                return [], "No valid pairs found."

            return result, None

        except Exception as e:
            return [], f"Unexpected error: {str(e)}"

    def generate_pairs(self, query):
        try:
            disease_count = len(set(re.findall(r"<disease(\d+)>", query)))
            drug_count = len(set(re.findall(r"<drug(\d+)>", query)))
            list_diseases = sorted(set(re.findall(r"(<disease\d+>)", query)))
            list_drugs = sorted(set(re.findall(r"(<drug\d+>)", query)))

            result_s = query
            if not result_s.endswith('.'):
                result_s += '.'

            generation_di_di = generation_di_di_prompt.replace('{result_s}', result_s).replace('{list_diseases}', ', '.join(list_diseases))
            generation_dr_dr = generation_dr_dr_prompt.replace('{result_s}', result_s).replace('{list_drugs}', ', '.join(list_drugs))
            generation_di_dr = generation_di_dr_prompt.replace('{result_s}', result_s).replace('{list_diseases}', ', '.join(list_diseases)).replace('{list_drugs}', ', '.join(list_drugs))

            #print('===============================================================')
            max_new_tokens = 100 + (disease_count + drug_count) * 17
            count = 0
            result = []
            error = None
            if disease_count > 1:
                result_disease_disease = []
                while not result_disease_disease and count < 3:
                    count += 1
                    resp_text = self.llm_p.generate(generation_di_di, max_new_tokens=max_new_tokens)
                    disease_result, error = self.check_output(resp_text, list_diseases, list_drugs)
                    if disease_result:
                        result_disease_disease.extend([list(pair) for pair in disease_result])
                if result_disease_disease:
                    result.extend(result_disease_disease)

            if drug_count > 1:
                result_drug_drug = []
                count = 0
                while not result_drug_drug and count < 3:
                    count += 1
                    resp_text = self.llm_p.generate(generation_dr_dr, max_new_tokens=max_new_tokens)
                    drug_result, error = self.check_output(resp_text, list_diseases, list_drugs)
                    if drug_result:
                        result_drug_drug.extend([list(pair) for pair in drug_result])

                if result_drug_drug:
                    result.extend(result_drug_drug)

            result_disease_drug = []
            count = 0
            while not result_disease_drug and count < 3:
                count += 1
                resp_text = self.llm_p.generate(generation_di_dr, max_new_tokens=max_new_tokens)
                disease_drug_result, error = self.check_output(resp_text, list_diseases, list_drugs)
                if disease_drug_result:
                    result_disease_drug.extend([list(pair) for pair in disease_drug_result])

            if result_disease_drug:
                result.extend(result_disease_drug)

            if not result:
                return [], "No valid pairs found."
            return result, error
        except Exception as e:
            return [], f"Unexpected error: {str(e)}"
       
    def generate_labels(self, query, result):
            errors_labeling = None
            results_labeling = []
            for pair in result:

                labeled_pairs, errors_labeling = self.label_each_pair_strict(query, [pair])
                
                results_labeling.extend(labeled_pairs)
            
            return results_labeling, errors_labeling