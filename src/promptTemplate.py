generation_prompt_template = f"""
Given the following text, identify and extract all medically relevant ischemic heart diseases and strokes explicitly or implicitly contained.  

Explicit mentions refer to diseases named directly within the text, preserving exact wording, formatting, and order.  
Implicit mentions must only be included when there is clear evidence linking the described condition to an ischemic heart disease or stroke.

Do not include mentions of medication, symptoms, medical devices, treatments, or medical procedures, even if they are associated with ischemic heart diseases or strokes. Focus exclusively on identifying disease names.

Exclude conditions that are not explicitly or implicitly ischemic heart diseases or strokes, such as hypertension, even if they suggest cardiovascular risk.

Return the output in JSON format with the key "cvd_terminologies", and the value as a list of extracted ischemic heart diseases and strokes, maintaining the exact original wording and order.  
If no ischemic heart diseases or strokes are found in a particular part, return an empty list in the JSON.  

Text: "{{text}}"  

The final JSON is:
"""

generate_icd_per_disease_prompt = f"""
Given the following text and a list of diseases present in the text, generate a set of related pairs. 
Each pair should consist of a disease name (identical to the one provided in the list) and its respective  3 digits ICD-10 code. 
The ICD code must best represent the disease based on the context provided in the text and your knowledge of ICD.

The ICD code must have 3 digits only, like the following : ["I20", "N93"]

Ensure that the disease name in the output pairs matches exactly as given in the input List_Diseases.

Each pair should be formatted as: (Disease Name, 3 digit ICD-10).

Return the generated pairs in a structured JSON format where the key is "Pairs" and the value is a list of pairs (with the format: [(Disease Name, ICD), (Disease Name, ICD)]).

### Input:
Text: {{text}}
List_Diseases: {{diseases_list}}

The final JSON is:
"""


review_disease_icd_pair_prompt = f"""
Given the following sentence: "{{text}}", identify whether the entity "{{entity}}" represents a disease and if it correctly matches the category "{{description}}".  

Respond based on the following guidelines:  
1. If the entity "{{entity}}" is not a disease, select option "1".  
2. If the entity "{{entity}}" is a disease, but does not fit the category "{{description}}", select option "2".  
3. If the entity "{{entity}}" is a disease and matches the category "{{description}}", select option "3".  

Return the response in a structured JSON format where the key is "Result" and the value is the selected option as an integer.  

### Input:  
Entity: "{{entity}}"  
Category: "{{description}}"  

### Response:
"""
