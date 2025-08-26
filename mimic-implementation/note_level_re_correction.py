import json
import pandas as pd
from colorama import init, Fore, Style
import re
import sys

# Initialize colorama
init(autoreset=True)

# ------------------------------
# 🔧 Load configuration
# ------------------------------
with open('parameters_re.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

input_json = config['input_json']
corrected_json = config['corrected_output_json']
corrected_output_json = config['corrected_output_json']
documentss_csv = config['documentss_csv']

# ------------------------------
# 🎨 Function to colorize tags
# ------------------------------
def colorize_tags(text):
    text = re.sub(r'(<disease\d+>.*?</disease\d+>)', lambda m: Fore.RED + m.group(1) + Style.RESET_ALL, text)
    text = re.sub(r'(<drug\d+>.*?</drug\d+>)', lambda m: Fore.BLUE + m.group(1) + Style.RESET_ALL, text)
    return text

def print_progress_bar(current, total, bar_length=40):
    """
    Prints a progress bar in the terminal.

    :param current: current index (1-based for display)
    :param total: total steps
    :param bar_length: progress bar length
    """
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = fraction * 100
    sys.stdout.write(f'\rProgress: |{bar}| {current}/{total} ({percent:.1f}%)')
    sys.stdout.flush()
    if current == total:
        print()  # New line when the bar ends

# ------------------------------
# 📦 Load data
# ------------------------------
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)['results']
try:
    with open(corrected_json, 'r', encoding='utf-8') as f:
        data_corrected = json.load(f)['results']
except FileNotFoundError:
    data_corrected = []

df = pd.read_csv(documentss_csv)
df['relations'] = None
df['id'] = df['hadm_id'].astype(str) + "_" + df['idx'].astype(str)
json_ids = {entry['id'] for entry in data}
df = df[df['id'].isin(json_ids)].reset_index(drop=True)
print(f"Total notes in DataFrame: {len(df)}")

json_ids_corrected = {entry['id'] for entry in data_corrected}
df = df[~df['id'].isin(json_ids_corrected)].reset_index(drop=True)
print(f"Total notes after removal: {len(df)}")

# ------------------------------
# 📋 Fill relations
# ------------------------------
for i, row in df.iterrows():
    id_note = row['id']
    match = next((note for note in data if note['id'] == id_note), None)
    if match:
        df.at[i, 'relations'] = match['relations']

# ------------------------------
# 👩‍⚕️ Interactive review
# ------------------------------
def interactive_review(df, existing_data=None):
    updated_data = existing_data if existing_data else []
    total = len(df)
    for i, row in df.iterrows():
        note_id = row['id']
        if any(entry['id'] == note_id for entry in updated_data):
            print(f"Note {note_id} has already been corrected. Skipping...\n")
            continue

        print("\n" + "="*80)
        print_progress_bar(i, total)
        print()
        print(f"documents #{i} (ID: {note_id}):\n")
        print(colorize_tags(row['documents']))

        relations = row['relations'] if isinstance(row['relations'], list) else []

        if not relations:
            print("\n⚠️  No relations detected.")
            choice = input("➕ Do you want to add a new relation? (y/n): ").strip().lower()
            if choice != "y":
                updated_data.append({
                    'id': note_id,
                    'documents': row['documents'],
                    'relations': []
                })
                with open(corrected_json, 'w') as f:
                    json.dump({'results': updated_data}, f, indent=4)
                continue

        updated_relations = []
        for j, triplet in enumerate(relations):
            print(f"\n[{j}] {triplet[0]}  —  {triplet[1]}  —  {triplet[2]}")
            action = input("    (Enter = keep, d = delete, e = edit): ").strip().lower()
            if action == "d":
                continue
            elif action == "e":
                subj, rel, obj = triplet
                if "<disease" in subj and "<disease" in obj:
                    relation_list = ['related to', 'comorbidity', 'no relation', 'other']
                elif "<disease" in subj and "<drug" in obj or "<drug" in subj and "<disease" in obj:
                    relation_list = ['indicated', 'contraindicated', 'discontinued', 'no relation', 'other']
                else:
                    relation_list = ['related to', 'combined with', 'no relation', 'contraindicated with', 'other']
                print("\nAvailable relations:")
                for idx, r in enumerate(relation_list):
                    print(f"{idx}. {r}")
                rel_idx = int(input(f"Select the relation (0-{len(relation_list)-1}): ").strip())
                rel = relation_list[rel_idx]
                updated_relations.append([subj, rel, obj])
            else:
                updated_relations.append(triplet)

        def is_valid_entity_tag(text):
            return re.match(r'^<((disease|drug)\d+)>.*?</\1>$', text.strip())

        while True:
            new = input("\n➕ Add new relation? (y/n): ").strip().lower()
            if new == "y":
                subj = input("      Subject entity: ").strip()
                obj = input("      Object entity: ").strip()
                if not is_valid_entity_tag(subj) or not is_valid_entity_tag(obj):
                    print("⚠️  Malformed entities.")
                    continue
                if "<disease" in subj and "<disease" in obj:
                    relation_list = ['related to', 'comorbidity', 'no relation', 'other']
                elif "<drug" in subj and "<drug" in obj:
                    relation_list = ['related to', 'combined with', 'no relation', 'other']
                else:
                    relation_list = ['indicated', 'contraindicated', 'discontinued', 'no relation', 'other']
                for idx, r in enumerate(relation_list):
                    print(f"{idx}. {r}")
                try:
                    rel_idx = int(input(f"Select the relation (0-{len(relation_list)-1}): ").strip())
                    rel = relation_list[rel_idx]
                    updated_relations.append([subj, rel, obj])
                except (ValueError, IndexError):
                    print("❌ Invalid index.")
                    continue
            else:
                break

        print("\n✅ Final relations:")
        for r in updated_relations:
            print(f"   - {r[0]} — {r[1]} — {r[2]}")

        df.at[i, 'relations'] = updated_relations
        updated_data.append({
            'id': note_id,
            'documents': row['documents'],
            'relations': updated_relations
        })
        with open(corrected_output_json, 'w', encoding='utf-8') as f:
            json.dump({'results': updated_data}, f, indent=4)

        next_one = input("\n➡️  Next? (Enter to continue, q to quit): ")
        if next_one.lower() == "q":
            break

    return updated_data

# ------------------------------
# 🧠 Load previous corrections
# ------------------------------
try:
    with open(corrected_output_json, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)['results']
except FileNotFoundError:
    existing_data = []

# ------------------------------
# ▶️ Run review
# ------------------------------
updated_data = interactive_review(df, existing_data)
print(f"\n✅ Corrections saved in '{corrected_output_json}'")
