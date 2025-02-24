import re
import json
from promptTemplate import generation_prompt_template, generate_icd_per_disease_prompt

class PairGenerator(object):
    """
    Class responsible for extracting diseases from clinical notes and generating ICD-10 code pairs.
    It uses two LLMs: one for disease extraction and another for ICD code assignment.
    """
    def __init__(self, llm_g1, llm_g2, args, model_name, device):
        self.llm = llm_g1 # LLM for disease extraction
        self.llm_icd = llm_g2 # LLM for ICD code generation
        self.args = args
        self.model_name = model_name
        self.device = device

    def check_diseases(self, note, resp_text):
        """
        Extracts diseases from model response and verifies if they are present in the clinical note.
        """
        try:
            pattern = r"\{(.*?)\}"
            matches = re.findall(pattern, resp_text.replace("\n", ""))
            if not matches:
                return "", "JSON pattern not found"

            # Tentar carregar o JSON
            json_str = "{" + matches[0] + "}"
            try:
                keys_dict = json.loads(json_str)
            except json.JSONDecodeError:
                return "", f"Invalid JSON"

            # Verificar chave esperada
            if "cvd_terminologies" not in keys_dict:
                return "", "JSON unreadable"

            # Verificar conteúdo da lista
            if not keys_dict["cvd_terminologies"]:
                return "", "No diseases found"

            diseases = list(set(keys_dict['cvd_terminologies']))
            mt = [disease for disease in diseases if re.search(rf'\b{re.escape(disease)}\b', note, re.IGNORECASE)]
            if not mt:
                errors = len([disease for disease in diseases if disease.lower() not in note.lower()])
                return "", errors

            return ', '.join(mt), len([disease for disease in diseases if disease.lower() not in note.lower()])

        except Exception as e:
            return "", f"Unexpected error: {str(e)}"
        
    def check_pairs(self, disease_list, resp_text):
        """
        Extracts disease-ICD pairs from model response and validates their format.
        """
        try:
            pattern = r"\{(.*?)\}"
            matches = re.findall(pattern, resp_text.replace("\n", "").replace("(", "[").replace(")", "]"))
            if not matches:
                return "", "JSON not found"

            keys_dict = json.loads("{" + matches[0] + "}")

            if "Pairs" not in keys_dict or not keys_dict["Pairs"]:
                return "", "No pairs found in JSON response"

        except json.JSONDecodeError:
            return "", "JSON unreadable"
        except Exception as e:
            return "", f"Error during model processing: {e}"

        # Process and filter pairs
        filtered_pairs = []
        error_pairs = 0
        for pair in keys_dict['Pairs']:
            try:
                if isinstance(pair, list) and len(pair) == 2:
                    disease, icd = pair
                    if disease.lower() in disease_list.lower() and len(icd) == 3 and icd[0:1].isalpha() and icd[1:].isdigit():
                        filtered_pairs.append((disease, icd))
                    else:
                        error_pairs += 1
                else:
                    error_pairs += 1
            except (TypeError, IndexError):
                return "", "Error processing pair structure"


        if not filtered_pairs:
            return "", error_pairs

        mt = ', '.join(f"('{disease}', '{icd}')" for disease, icd in filtered_pairs)
        return mt, error_pairs

    def generate_pairs(self, note):
        """
        Main function to extract diseases from a clinical note and match them with ICD-10 codes.
        """
        note_or = note
        mt = ""
        errors = {}
        count = 0
        n_tokens = 100 + len(note.split()) / 3 
        while count < 3:
            generated_text = self.llm.generate(generation_prompt_template.replace('{text}', note), n_tokens)
            mt_iter, er_dises = self.check_diseases(note_or, generated_text)
            errors[count] = er_dises
            count += 1
    
            if mt_iter:
                if mt != "":
                    mt += ", " + mt_iter
                else:
                    mt = mt_iter
    
                mt_l = mt_iter.split(", ")
                for term in mt_l:
                    note = re.sub(rf'\b{re.escape(term)}\b', '', note, flags=re.IGNORECASE)
    
                note = re.sub(r'\s+', ' ', note).strip()
    
        pairs_final = ""
        errors_p = {}
        mt = ", ".join(list(set(mt.split(", "))))
        if mt != "":
            mt_c = mt
            count = 0
            def split_into_chunks(lst, chunk_size):
                return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
            while mt_c != "" and count < 5:
                mt_list = mt_c.split(", ")
                disease_chunks = split_into_chunks(mt_list, 12)
                
                for chunk in disease_chunks:
                    mt_chunk = ", ".join(chunk)
                    generated_text = self.llm_icd.generate(generate_icd_per_disease_prompt.replace('{text}', note_or).replace('{diseases_list}', mt_chunk), 200)
                    pairs, er_pairs = self.check_pairs(mt_chunk, generated_text)
                    
                    if pairs.strip():
                        if pairs_final != "":
                            pairs_final += ", " + pairs
                        else:
                            pairs_final = pairs
                    else:
                        print("ICD ERROR:", mt_chunk, generated_text)
    
                    pairs_list = re.findall(r"\('([^']*)', '([^']*)'\)", pairs)
                    mt_list = [item for item in mt_list if not any(re.fullmatch(rf"\b{re.escape(disease)}\b", item, flags=re.IGNORECASE) for disease, _ in pairs_list)]
                
                mt_c = ", ".join(mt_list)
                errors_p[count] = er_pairs
                count += 1

        return pairs_final, errors_p, errors


class Generate(object):
    """
    Wrapper class to initialize and call disease extraction and ICD matching.
    """
    def __init__(self, llm_g1, llm_g2, args) -> None:
        super().__init__()
        self.class_name = 'Extract_Diseases_ICDs'
        self.class_desc = 'Extracts diseases and their respective ICD-10 codes from clinical notes.'
        self.llm = llm_g1
        self.llm_icds = llm_g2

        self.diseaseExtraction = PairGenerator(llm_g1, llm_g2, args=args, model_name=args.llm_name, device='auto')
        self.args = args

    def call(self,query):
        return self.diseaseExtraction.generate_pairs(query)
