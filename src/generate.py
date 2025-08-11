import re
import json
from promptTemplate import generation_prompt_template, generate_icd_per_disease_prompt, generate_id_per_entity_drug_prompt, generation_drug_prompt_template
from typing import List
import tqdm

class PairGenerator:
    def __init__(self, llm_entity_extraction, llm_id_mapping, max_input_tokens: int, entity_type: str):
        self.llm = llm_entity_extraction
        self.llm_id = llm_id_mapping
        self.max_input_tokens = max_input_tokens
        self.entity_type = entity_type

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

    def _check_entities(self, note, response, entity_type="disease"):
        if entity_type == "disease":
            entities, error = self._extract_json_from_response(response, "cvd_terminologies")
        else:
            entities, error = self._extract_json_from_response(response, "drugs_terminologies")
        if not entities:
            return "", error or "Empty disease list"

        entities = list(set(entities))
        matched = [d for d in entities if re.search(rf'\b{re.escape(d)}\b', note, re.IGNORECASE)]
        error_count = len([d for d in entities if d.lower() not in note.lower()])
        error_msg = f"Entities not found in note: {error_count}"

        if not matched:
            return "", error_msg

        return ', '.join(matched), error_msg

    def _check_pairs(self, entity_list_str, response):
        pairs_raw, error = self._extract_json_from_response(response, "Pairs")
        if error or not pairs_raw:
            return "", error or "Empty pairs list"

        entity_list = [d.strip().lower() for d in entity_list_str.split(",")]
        valid_pairs = []
        invalid_count = 0

        if self.entity_type == "disease":
            for item in pairs_raw:
                if isinstance(item, list) and len(item) == 2:
                    disease, icd = item
                    if (disease.lower() in entity_list and len(icd) == 3
                            and icd[0].isalpha() and icd[1:].isdigit()):
                        valid_pairs.append((disease, icd))
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
        else:
            for item in pairs_raw:
                if isinstance(item, list) and len(item) == 2:
                    drug, mesh_id = item
                    if (drug.lower() in entity_list and mesh_id.startswith("D")):
                        valid_pairs.append((drug, mesh_id))
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
        all_entities_list = [[] for _ in range(batch_size)]
        entity_errors_list = [{} for _ in range(batch_size)]
        id_errors_list = [{} for _ in range(batch_size)]
        updated_notes = notes.copy()

        # ==== PHASE 1: Entity generation ====

        for i in range(n_iterations):
            prompts = []
            max_new_tokens_list = []

            for note in updated_notes:
                if self.entity_type == "disease":
                    prompt = generation_prompt_template.replace('{text}', note)
                else:
                    prompt = generation_drug_prompt_template.replace('{text}', note)

                prompts.append(prompt)
                max_new_tokens = 100 + len(note.split()) // 3
                max_new_tokens_list.append(max_new_tokens)

            max_new_tokens = max(max_new_tokens_list)
            responses = self.llm.generate_batch(prompts, self.max_input_tokens, max_new_tokens)

            for idx, response in enumerate(responses):
                entities_str, error = self._check_entities(notes[idx], response, self.entity_type)
                entity_errors_list[idx][i] = error

                if entities_str:
                    extracted = [d.strip() for d in entities_str.split(",")]
                    all_entities_list[idx].extend(extracted)
                    updated_notes[idx] = self._remove_extracted_terms(updated_notes[idx], extracted)

        all_entities_list = [list(set(ds)) for ds in all_entities_list]

        # ==== PHASE 2: Pair generation (entity, id) ====
        final_pairs_list = []

        for idx, (note, entities) in enumerate(zip(notes, all_entities_list)):
            if not entities:
                final_pairs_list.append("")
                continue

            all_pairs = []
            chunk_size = 12
            max_pair_iterations = 5
            remaining_entities = entities.copy()

            for i in range(max_pair_iterations):
                if not remaining_entities:
                    break

                chunk_prompts = []
                chunk_entity_chunks = []

                for chunk in self._split_chunks(remaining_entities, chunk_size):
                    entity_str = ', '.join(chunk)
                    if self.entity_type == "disease":
                        prompt = generate_icd_per_disease_prompt \
                            .replace('{text}', note) \
                            .replace('{diseases_list}', entity_str)
                    else:
                        prompt = generate_id_per_entity_drug_prompt \
                            .replace('{text}', note) \
                            .replace('{drug_list}', entity_str)

                    chunk_prompts.append(prompt)
                    chunk_entity_chunks.append(chunk)

                responses = self.llm_id.generate_batch(chunk_prompts, self.max_input_tokens, max_new_tokens)                

                for resp, chunk, chunk_prompt_idx in zip(responses, chunk_entity_chunks, range(len(responses))):
                    pairs_str, error = self._check_pairs(', '.join(chunk), resp)
                    id_errors_list[idx][i * chunk_size + chunk_prompt_idx] = error
                    if pairs_str:
                        all_pairs.append(pairs_str)
                        matched = re.findall(r"\('([^']*)', '([^']*)'\)", pairs_str)
                        matched_entities = [d for d, _ in matched]
                        remaining_entities = [
                            d for d in remaining_entities
                            if not any(re.fullmatch(rf'\b{re.escape(md)}\b', d, re.IGNORECASE) for md in matched_entities)
                        ]

            final_pairs_list.append(', '.join(all_pairs))

        return final_pairs_list, id_errors_list, entity_errors_list


class Generate:
    def __init__(self, llm_entity, llm_id, max_input_tokens, entity_type):
        self.generator = PairGenerator(llm_entity, llm_id, max_input_tokens, entity_type)

    def call(self, notes, n_iterations):
        result = self.generator.generate_pairs(notes, n_iterations)
        return result
