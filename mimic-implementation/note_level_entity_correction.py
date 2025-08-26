#!/usr/bin/env python3
import json
import pandas as pd
import re
from tqdm import tqdm
import os

path_manual = "pre_annotated_corpus.json"
entity_type = "disease"  # "disease" or "drug"
path_notes = "clinical_notes_text.csv"
output_file = "supervised_corpus.csv"

with open(path_manual, "r", encoding="utf-8") as f:
    manual_annotations = json.load(f)['results']

notes_df = pd.read_csv(path_notes, compression='gzip', encoding='utf-8')

manual_dict = {str(x['id']): x for x in manual_annotations}
print(f"Total notes with manual annotations: {len(manual_dict)}")

if os.path.exists(output_file):
    corrections_done = pd.read_csv(output_file)
    corrections_done = corrections_done.to_dict(orient="records")
else:
    corrections_done = []

def highlight_occurrence(text, start, end, color='\033[91m'):
    RESET = '\033[0m'
    return text[:start] + color + text[start:end] + RESET + text[end:]

def review_entity(entity):
    """Allow user to keep, edit, or delete an entity."""
    code_type = "ICD" if entity_type == "disease" else "MeSH"
    code_value = entity.get('icd') if entity_type == "disease" else entity.get('mesh', '')

    print(f"\nEntity: '{entity['entity']}' ({code_type}: {code_value}) span=({entity['start']},{entity['end']})")
    action = input("    (Enter = keep, e = edit, d = delete): ").strip().lower()
    
    if action == "d":
        return None
    elif action == "e":
        new_entity = input(f"      New entity text [{entity['entity']}]: ").strip()
        if new_entity:
            entity['entity'] = new_entity
        new_code = input(f"      New {code_type} [{code_value}]: ").strip()
        if new_code:
            if entity_type == "disease":
                entity['icd'] = new_code
            else:
                entity['mesh'] = new_code
        return entity
    else:
        return entity

for note_id, entry in tqdm(manual_dict.items(), desc="Notes"):
    if any(str(d['id']) == str(note_id) for d in corrections_done):
        print(f"Note {note_id} already corrected. Skipping...")
        continue

    note_row = notes_df[notes_df['hadm_id'].astype(str) == note_id]
    if note_row.empty:
        print(f" Note {note_id} not found in CSV.")
        continue

    document = note_row.iloc[0]['hospital_course']

    entities_field = 'diseases_predicted' if entity_type == "disease" else 'drugs_predicted'
    code_field = 'icd' if entity_type == "disease" else 'mesh'
    entities = entry.get(entities_field, [])

    print("\n" + "="*80)
    print(f"Note ID: {note_id}\n")
    print(document)
    print("="*80)

    corrected_entities = []
    for ent in entities:
        print(highlight_occurrence(document, ent['start'], ent['end']))
        edited = review_entity(ent)
        if edited:
            corrected_entities.append(edited)

    # Add new entities
    while True:
        new = input("\n Add new entity? (y/n): ").strip().lower()
        if new != "y":
            break
        new_entity = input("      Entity text: ").strip()
        new_code = input(f"      {code_field.upper()} code: ").strip()
        try:
            start = int(input("      Start index: ").strip())
            end = int(input("      End index: ").strip())
        except ValueError:
            print("❌ Invalid indices.")
            continue
        new_entry = {"entity": new_entity, "start": start, "end": end}
        if entity_type == "disease":
            new_entry['icd'] = new_code
        else:
            new_entry['mesh'] = new_code
        corrected_entities.append(new_entry)


    corrections_done.append({
        "id": note_id,
        "document": document,
        "entities": json.dumps(corrected_entities, ensure_ascii=False)
    })

    pd.DataFrame(corrections_done).to_csv(output_file, index=False, encoding="utf-8")

    next_one = input("\n Next note? (Enter to continue, q to quit): ")
    if next_one.lower() == "q":
        break

print(f"\n All corrections saved in '{output_file}'")
