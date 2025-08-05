import re
import json
from promptTemplate import generation_prompt_template, generate_icd_per_disease_prompt
from typing import List

class PairGenerator:
    def __init__(self, llm_disease_extraction, llm_icd_mapping, max_input_tokens: int):
        self.llm = llm_disease_extraction
        self.llm_icd = llm_icd_mapping
        self.max_input_tokens = max_input_tokens

    def _extract_json_from_response(self, text, key_expected):
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, text.replace("\n", "").replace("(", "[").replace(")", "]"))

        if not matches:
            return None, f"JSON pattern not found for key '{key_expected}'"

        json_str = "{" + matches[0] + "}"
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None, f"JSON decode error: {json_str}"

        if key_expected not in data:
            return None, f"Expected key '{key_expected}' not found"

        return data[key_expected], None

    def _check_diseases(self, note, response):
        diseases, error = self._extract_json_from_response(response, "cvd_terminologies")
        if not diseases:
            return "", error or "Empty disease list"

        diseases = list(set(diseases))
        matched = [d for d in diseases if re.search(rf'\b{re.escape(d)}\b', note, re.IGNORECASE)]
        error_count = len([d for d in diseases if d.lower() not in note.lower()])
        error_msg = f"Diseases not found in note: {error_count}"

        if not matched:
            return "", error_msg

        return ', '.join(matched), error_msg

    def _check_pairs(self, disease_list_str, response):
        pairs_raw, error = self._extract_json_from_response(response, "Pairs")
        if error or not pairs_raw:
            return "", error or "Empty pairs list"

        disease_list = [d.strip().lower() for d in disease_list_str.split(",")]
        valid_pairs = []
        invalid_count = 0

        for item in pairs_raw:
            if isinstance(item, list) and len(item) == 2:
                disease, icd = item
                if (disease.lower() in disease_list and len(icd) == 3 
                        and icd[0].isalpha() and icd[1:].isdigit()):
                    valid_pairs.append((disease, icd))
                else:
                    invalid_count += 1
            else:
                invalid_count += 1

        if not valid_pairs:
            return "", f"No valid pairs found, errors: {invalid_count}"

        formatted = ', '.join(f"('{d}', '{c}')" for d, c in valid_pairs)
        return formatted, f"Invalid pairs count: {invalid_count}"

    def _remove_extracted_terms(self, note, terms):
        for term in terms:
            note = re.sub(rf'\b{re.escape(term)}\b', '', note, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', note).strip()

    def _split_chunks(self, items, size):
        return [items[i:i + size] for i in range(0, len(items), size)]

    def generate_pairs(self, notes: List[str], n_iterations: int):
        batch_size = len(notes)
        all_diseases_list = [[] for _ in range(batch_size)]
        disease_errors_list = [{} for _ in range(batch_size)]
        icd_errors_list = [{} for _ in range(batch_size)]
        updated_notes = notes.copy()

        # ==== PHASE 1: Disease generation ====

        for i in range(n_iterations):
            prompts = []
            max_new_tokens_list = []

            for note in updated_notes:
                prompt = generation_prompt_template.replace('{text}', note)
                prompts.append(prompt)
                max_new_tokens = 100 + len(note.split()) // 3
                max_new_tokens_list.append(max_new_tokens)

            max_new_tokens = max(max_new_tokens_list)
            responses = self.llm.generate_batch(prompts, self.max_input_tokens, max_new_tokens)

            for idx, response in enumerate(responses):
                diseases_str, error = self._check_diseases(notes[idx], response)
                disease_errors_list[idx][i] = error
                print(f"[Phase 1] Note {idx} | Diseases: {diseases_str} | Error: {error}", flush=True)

                if diseases_str:
                    extracted = [d.strip() for d in diseases_str.split(",")]
                    all_diseases_list[idx].extend(extracted)
                    updated_notes[idx] = self._remove_extracted_terms(updated_notes[idx], extracted)

        all_diseases_list = [list(set(ds)) for ds in all_diseases_list]

        # ==== PHASE 2: Pair generation (disease, ICD) ====
        final_pairs_list = []

        for idx, (note, diseases) in enumerate(zip(notes, all_diseases_list)):
            if not diseases:
                final_pairs_list.append("")
                continue

            all_pairs = []
            chunk_size = 12
            max_pair_iterations = 5
            remaining_diseases = diseases.copy()

            for i in range(max_pair_iterations):
                if not remaining_diseases:
                    break

                chunk_prompts = []
                chunk_disease_chunks = []

                for chunk in self._split_chunks(remaining_diseases, chunk_size):
                    disease_str = ', '.join(chunk)
                    prompt = generate_icd_per_disease_prompt \
                        .replace('{text}', note) \
                        .replace('{diseases_list}', disease_str)

                    chunk_prompts.append(prompt)
                    chunk_disease_chunks.append(chunk)

                responses = self.llm_icd.generate_batch(chunk_prompts, self.max_input_tokens, max_new_tokens)

                for resp, chunk, chunk_prompt_idx in zip(responses, chunk_disease_chunks, range(len(responses))):
                    pairs_str, error = self._check_pairs(', '.join(chunk), resp)
                    icd_errors_list[idx][i * chunk_size + chunk_prompt_idx] = error
                    if pairs_str:
                        all_pairs.append(pairs_str)
                        matched = re.findall(r"\('([^']*)', '([^']*)'\)", pairs_str)
                        matched_diseases = [d for d, _ in matched]
                        remaining_diseases = [
                            d for d in remaining_diseases
                            if not any(re.fullmatch(rf'\b{re.escape(md)}\b', d, re.IGNORECASE) for md in matched_diseases)
                        ]

            final_pairs_list.append(', '.join(all_pairs))

        return final_pairs_list, icd_errors_list, disease_errors_list


class Generate:
    def __init__(self, llm_disease, llm_icd, max_input_tokens):
        self.generator = PairGenerator(llm_disease, llm_icd, max_input_tokens)

    def call(self, notes, n_iterations):
        result = self.generator.generate_pairs(notes, n_iterations)
        return result
