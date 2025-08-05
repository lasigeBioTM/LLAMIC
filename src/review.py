
import re
import json
from promptTemplate import review_disease_icd_pair_prompt


class ReviewGenerator:
    def __init__(self, llm, max_input_tokens, device):
        self.llm = llm
        self.device = device
        self.max_new_tokens = 100
        self.max_input_tokens = max_input_tokens

    def review_disease(self, note, entity, description):
        try:
            prompt = review_disease_icd_pair_prompt \
                .replace("{text}", note) \
                .replace("{entity}", entity) \
                .replace("{description}", description)

            response = self.llm.generate(
                prompt,
                max_input_tokens=self.max_input_tokens,
                max_new_tokens=self.max_new_tokens
            )

            pattern = r"\{(.*?)\}"
            matches = re.findall(pattern, response.replace("\n", ""))

            if not matches:
                return "", "JSON pattern not found"

            json_str = "{" + matches[0] + "}"
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return "", "Invalid JSON structure"

            if "Result" not in data or not isinstance(data["Result"], int):
                return "", "Missing or invalid 'Result' field"

            return data["Result"], ""

        except Exception as e:
            return "", f"Unexpected error: {str(e)}"

class Review:
    def __init__(self, llm, lexicon, max_input_tokens, device):
        self.llm = llm
        self.lexicon = lexicon
        self.max_input_tokens = max_input_tokens
        self.device = device
        self.disease_review = ReviewGenerator(llm=llm, device=device, max_input_tokens=max_input_tokens)

    def call(self, document: str, pairs: str):
        extracted_pairs = re.findall(r"\('([^']+)', '([^']+)'\)", pairs)
        if not extracted_pairs:
            return [], ["No valid (entity, ICD) pairs found"]

        doc_results = []
        doc_errors = []

        for entity, icd in extracted_pairs:
            pattern = r'\b' + re.escape(entity) + r'\b'
            matches = list(re.finditer(pattern, document, flags=re.IGNORECASE))

            if not matches:
                print(f"⚠️ Entity '{entity}' not found in document", flush=True)
                doc_results.append({
                    "entity": entity,
                    "icd": icd,
                    "result": "Entity not found in document"
                })
                doc_errors.append("Entity not found in document")
                continue

            description = self.lexicon.get_description(str(icd))

            for idx, match in enumerate(matches):
                start, end = match.span()
                masked_text = document
                for j, other in enumerate(matches):
                    if j != idx:
                        o_start, o_end = other.span()
                        masked_text = masked_text[:o_start] + "_" * (o_end - o_start) + masked_text[o_end:]

                if description != "Unknown":
                    result, error = self.disease_review.review_disease(masked_text, entity, description)
                    if result != 3:
                        continue
                    result_entry = {
                        "entity": entity,
                        "icd": icd,
                        "start": start,
                        "end": end
                    }
                    doc_results.append(result_entry)
                    doc_errors.append(error)
                else:
                    continue
        return doc_results, doc_errors
