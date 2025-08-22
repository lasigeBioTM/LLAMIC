
generation_di_di_prompt = f"""
Given the following sentence, extract all relevant relations between diseases ONLY.
### IMPORTANT RULES:
- Consider ONLY entities explicitly marked with <diseaseidx>.
- DO NOT create new entities.
- Extract ONLY relations between diseases.
- Return a structured JSON response with the key "Pairs" and a list of list of pairs of entities in the form:
- JUST A EXAMPLE: ['<disease4>', '<disease5>'], ['<disease10>', '<disease11>']
- ONLY MENTION THE DISEASE BY THE TAG <diseaseidx>, where the idx MUST BE THE SAME AS IN THE INPUT.

Sentence to process:
sentence: "{{result_s}}"
List of diseases ONLY to consider: {{list_diseases}}
### JSON Response:
"""

generation_dr_dr_prompt = f"""
Given the following sentence, extract all relevant relations between drugs ONLY.
### IMPORTANT RULES:
- Consider ONLY entities explicitly marked with <drugidx>.
- DO NOT create new entities.
- Extract ONLY relations between drugs.
- Return a structured JSON response with the key "Pairs" and a list of list of pairs of entities in the form:
- JUST A EXAMPLE: ['<drug4>', '<drug5>'], ['<drug10>', '<drug11>']
- ONLY MENTION THE DRUG BY THE TAG <drugidx>, where the idx MUST BE THE SAME AS IN THE INPUT.
Sentence to process:
sentence: "{{result_s}}"
List of drugs ONLY to consider: {{list_drugs}}
### JSON Response:
"""

generation_di_dr_prompt = f"""
Given the following sentence, extract all relevant relations between diseases and drugs ONLY.
### IMPORTANT RULES:
- Consider ONLY entities explicitly marked with <diseaseidx> and <drugidx>.
- DO NOT create new entities.
- Extract ONLY relations between diseases and drugs.
- Return a structured JSON response with the key "Pairs" and a list of list of pairs of entities in the form:
- JUST A EXAMPLE: ['<disease4>', '<drug5>'], ['<drug10>', '<drug11>']
- ONLY MENTION THE DRUG BY THE TAG <drugidx> AND THE DISEASE BY THE TAG <diseaseidx>, where the idx MUST BE THE SAME AS IN THE INPUT.

Sentence to process:
sentence: "{{result_s}}"
List of diseases ONLY to consider: {{list_diseases}}
List of drugs ONLY to consider: {{list_drugs}}
### JSON Response:
"""

prompt_label_first = f"""
Given the following context and a pair of medical entities, determine the most accurate relationship between them.

### Instructions:
- The relation label must be as precise as possible based on the context.
- The relation label should be one of the following: {{relation_list}}.
- The entities in the triplet MUST be exactly as provided in the pair, including their respective tags with the correct numbering, as presented in the context and entity pair.

### Relation Descriptions:
"""

prompt_label_second = f"""
### IMPORTANT RULE:
- The response MUST be in a structured JSON format, with the key "Label" and the value as a triplet: ["<tag>Entity1</tag>", "relation", "<tag>Entity2</tag>"].
- When mention the disease or drug, use the tag <diseaseidx> or <drugidx> respectively, where the idx MUST BE THE SAME NUMBER AS IN THE INPUT.
            

### Input Data:
- **Context:** "{{new_query}}"
- **Entity Pair:** {{entity1_name}}, {{entity2_name}}

### JSON Response with the key "Label" and the value as a triplet:
"""

prompt_label_relaxed = f"""
Given the following context and a pair of medical entities, determine the most accurate relationship between them.

### Instructions:
- The relation label must be as precise as possible, using a maximum of five words.
- The relation label should best describe the relationship between the two entities.
- Only consider interactions relevant to the provided context.
- Ensure that the extracted relationship is meaningful in a medical or biomedical setting.
- The entities in the triplet MUST be exactly as provided in the pair, including their respective tags with the correct numbering, as presented in the context and entity pair.
- The response MUST be in a structured JSON format, with the key "Label" and the value as a triplet: ["<tagidx>Entity1</tagidx>", "relation", "<tagidx>Entity2</tagidx>"].
### IMPORTANT RULE:
- When mention the disease or drug, use the tag <diseaseidx> or <drugidx> respectively, where the idx MUST BE THE SAME NUMBER AS IN THE INPUT.
            

### Input Data:
- **Context:** "{{new_query}}"
- **Entity Pair:** {{entity1_name}}, {{entity2_name}}

### JSON Response:
"""