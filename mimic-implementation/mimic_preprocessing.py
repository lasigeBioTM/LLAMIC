import pandas as pd
import json
import argparse
import re
import logging
import ast
from collections import Counter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessing:
    def __init__(self, params_path):
        self.params_path = params_path

    def plot_label_distribution(self, df_labels, summary_path, name='disease'):
        """
        Plot the distribution of labels in the dataset.
        """
        
        df_labels['label_ids'] = df_labels['label'].apply(lambda x: x.split(' - ')[0])
        label_counts = df_labels['label_ids'].value_counts()
        # Add to summary txt file the complete list of labels and their counts, without deleting the previous content
        with open(summary_path, 'a') as f:
            f.write("\n\n" + name + " distribution:\n")
            f.write(label_counts.to_string())

    def plot_distribution(self, notes_df, summary_path):
        """
        Plot the distribution of diseases and drugs in the dataset.
        """
        notes_df = notes_df.copy()
        notes_df["label_disease_list"] = notes_df["label_disease"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        notes_df["label_drug_list"] = notes_df["label_drug"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

        n_both = notes_df[(notes_df["label_disease_list"].str.len() > 0) & (notes_df["label_drug_list"].str.len() > 0)].shape[0]
        logging.info(f"\nNumber of notes with both diseases and drugs: {n_both}")

        drugs_set = set(drug for drugs in notes_df["label_drug_list"] for drug in drugs)

        drug_counter = Counter()
        drug_disease_counter = Counter()

        for _, row in notes_df.iterrows():
            has_disease = len(row["label_disease_list"]) > 0
            for drug in row["label_drug_list"]:
                drug_counter[drug] += 1
                if has_disease:
                    drug_disease_counter[drug] += 1

        drugs = {drug: (drug_counter[drug], drug_disease_counter[drug]) for drug in drugs_set}

        drugs["Total"] = (notes_df.shape[0], notes_df[notes_df["label_disease_list"].str.len() > 0].shape[0])

        max_drug_length = max(len(drug) for drug in drugs.keys()) if drugs else 30

        with open(summary_path, 'a') as f:
            f.write("\n\nLabel distribution:\n")
            f.write(f"{'Drug Name':<{max_drug_length+2}}{'Total Count':<15}{'With Disease':<15}\n")
            f.write("-" * (max_drug_length + 34) + "\n")
            for drug, counts in sorted(drugs.items()):
                f.write(f"{drug:<{max_drug_length+2}}{counts[0]:<15}{counts[1]:<15}\n")


    def extract_diseases(self, data_icds_lexicon, data_path, icd_targets_path, summary_path=None):
        """
        Extracts diseases of interest from the dataset based on ICD codes.
        """

        names_icds = {}

        with open(icd_targets_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' - ')[0]

                if '|' in parts:
                    parts = parts.split('|')
                    icd10 = None
                    for part in parts:
                        if part[0].isalpha():
                            icd10 = part
                            break
                    for part in parts:
                        names_icds[part] = icd10
                else:
                    names_icds[parts] = parts

        icd_codes = list(names_icds.keys())
        # Load disease lexicon and simplify ICD codes
        df_icd = pd.read_csv(data_icds_lexicon, compression='gzip')
        df_icd['icd_code_simpl'] = df_icd['icd_code'].astype(str).str[:3]

        # Filter relevant diseases
        df_diseases = df_icd[df_icd['icd_code_simpl'].isin(icd_codes)]
        logging.info(f"\nNumber of diseases found in {data_icds_lexicon}: {len(df_diseases)}")

        # Get unique ICD codes from the filtered data
        diseases_icd_codes = df_diseases['icd_code'].unique()

        # Load and filter data in chunks
        chunk_size = 500000
        filtered_chunks = []

        for chunk in pd.read_csv(data_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'icd_code'], chunksize=chunk_size):
            chunk_filtered = chunk[chunk['icd_code'].isin(diseases_icd_codes)]
            # in df_diseases we have the long_title, so we can add this information to the filtered chunk
            chunk_filtered = chunk_filtered.merge(df_diseases[['icd_code', 'long_title']], on='icd_code', how='left')
            chunk_filtered['label'] = chunk_filtered['icd_code'].str[:3].apply(lambda x: names_icds.get(x, 'NO_ICD_FOUND')) + ' - ' + chunk_filtered['long_title']
            filtered_chunks.append(chunk_filtered)

        # Concatenate filtered chunks
        df_diseases_filtered = pd.concat(filtered_chunks, ignore_index=True)
        
        # Only keep the columns of interest
        df_diseases_filtered = df_diseases_filtered[['subject_id', 'hadm_id', 'label']]

        logging.info(f"\nNumber of diseases of interest found in {data_path}: {len(df_diseases_filtered)}, {df_diseases_filtered['label'].nunique()} unique diseases")
        # Plot the distribution of diseases in the dataset
        self.plot_label_distribution(df_diseases_filtered, summary_path, name='disease')

        return df_diseases_filtered

    def extract_drugs(self, data_path, drugs_targets_path, summary_path=None):
        """
        Given a path to a txt file containing drug names (one per line),
        this function extracts the drugs of interest from the dataset and returns a filtered DataFrame.
        """

        # Load drugs of interest
        with open(drugs_targets_path, 'r') as f:
            lines = f.readlines()

        drugs = [line.split(' - ')[1].strip().lower() for line in lines]

        # Processa os IDs removendo "MESH:" e lidando com múltiplos IDs por medicamento
        ids_list = []
        for line in lines:
            ids_raw = line.split(' - ')[0]  # Obtém a parte dos IDs
            ids_cleaned = [id_part.split(':')[1] for id_part in ids_raw.split('|')]  # Remove "MESH:"
            ids_list.append(ids_cleaned)

        # conver the list in to str. if more than one id is present add |
        ids_list = ['|'.join(ids) for ids in ids_list]

        # Cria um dicionário associando os medicamentos aos seus respectivos IDs
        id_mesh = {drug: ids for drug, ids in zip(drugs, ids_list)}

        # Load data in chunks
        chunk_size = 500000
        filtered_chunks = []

        for chunk in pd.read_csv(data_path, compression='gzip', usecols=['subject_id','hadm_id','drug'], chunksize=chunk_size):
            chunk['drug'] = chunk['drug'].astype(str).str.lower().str.strip()  # Normalize column values
            filtered_chunk = chunk[chunk['drug'].isin(drugs)]
            filtered_chunks.append(filtered_chunk)

        # Concatenate all filtered chunks
        df_drugs = pd.concat(filtered_chunks, ignore_index=True)

        logging.info(f"\nThe number of drugs of interest found in {data_path}: {len(df_drugs)}, {df_drugs['drug'].nunique()} unique drugs")
        #rename the column 'drug' to 'label'
        df_drugs.rename(columns={'drug': 'label'}, inplace=True)

        # add id MESH to the label
        # Caso o nome não esteja no dicionário, retorna um valor padrão ou mantém apenas o nome
        df_drugs['label'] = df_drugs['label'].apply(
            lambda x: id_mesh.get(x, 'ID_NAO_ENCONTRADO') + ' - ' + x
        )

        # Plot the distribution of drugs in the dataset
        if summary_path:
            self.plot_label_distribution(df_drugs, summary_path, name='drug')
        return df_drugs
    
    def clean_text(self, df_notes_filtered):
        
        # Of note these patterns would not caputre all hospital courses, and is indeed a convservative approach to ensure quality of data
        pattern1  = re.compile("Brief Hospital Course.*\n*((?:\n.*)+?)(Medications on Admission|___  on Admission|___ on Admission)")
        pattern2  = re.compile("Brief Hospital Course.*\n*((?:\n.*)+?)Discharge Medications")
        pattern3  = re.compile("(Brief Hospital Course|rief Hospital Course|HOSPITAL COURSE)\
                                .*\n*((?:\n.*)+?)\
                                (Medications on Admission|Discharge Medications|DISCHARGE MEDICATIONS|DISCHARGE DIAGNOSIS|Discharge Disposition|___ Disposition|CONDITION ON DISCHARGE|DISCHARGE INSTRUCTIONS)")
        pattern4  = re.compile("(Mg-[12].|LACTATE-[12].|Epi-|Gap-|COUNT-|TRF-)___(.*\n*((?:\n.*)+?))(Medications on Admission)")


        # Idea here is to try more convservaite pattern first, if not work, try less conservative pattern
        def split_note(note):
            if re.search(pattern1, note):
                return re.search(pattern1, note).group(1)
            else:
                if re.search(pattern2, note):
                    return re.search(pattern2, note).group(1)
                else:
                    if re.search(pattern3, note):
                        return re.search(pattern3, note).group(2)
                    else:
                        if re.search(pattern4, note):
                            return re.search(pattern4, note).group(2)
                        else:
                            return None

        # Apply the function to the 'text' column
        df_notes_filtered['hospital_course'] = df_notes_filtered['text'].apply(split_note)

        # Drop those records that do not have hospital course captured with above regular expression patterns
        dc_summary = df_notes_filtered.dropna(subset=["hospital_course"]).copy()

        dc_summary["num_words"] = dc_summary["hospital_course"].apply(lambda x: len(x.split()))

        # Remove the notes with less than 40 words
        dc_summary = dc_summary[dc_summary["num_words"] > 40]

        # Remove duplicate hospital courses (but keep the first one), as most of these notes represent low quality data
        dc_summary = dc_summary.drop_duplicates(subset=["hospital_course"], keep="first")

        # Mean number of words in the hospital course is 378
        dc_summary["num_words"].mean()

        # only keep hadm_id and hospital_course
        dc_summary = dc_summary[["subject_id", "hadm_id", "hospital_course", "label_disease", "label_drug"]]

        dc_summary['hospital_course'] = dc_summary['hospital_course'].str.replace('\n', ' ')
        dc_summary['hospital_course'] = dc_summary['hospital_course'].str.replace('"', ' ')
        dc_summary['hospital_course'] = dc_summary['hospital_course'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        logging.info("Hospital_course done: %d, Original: %d", len(dc_summary), len(df_notes_filtered))
        return dc_summary

    def filter_notes_by_labels(self, notes_path, df_diseases, df_drugs):
        hadm_ids = set(df_diseases['hadm_id']).union(set(df_drugs['hadm_id']))

        df_notes_chunks = pd.read_csv(
            notes_path, compression='gzip', chunksize=10000, usecols=['subject_id', 'hadm_id', 'text']
        )
        df_notes = pd.concat(df_notes_chunks, ignore_index=True)

        disease_map = df_diseases.groupby('hadm_id')['label'].apply(list).to_dict()
        drug_map = df_drugs.groupby('hadm_id')['label'].apply(list).to_dict()
        
        df_notes['label_disease'] = df_notes['hadm_id'].map(disease_map)
        df_notes['label_disease'] = df_notes['label_disease'].apply(lambda x: x if isinstance(x, list) else [])
        
        df_notes['label_drug'] = df_notes['hadm_id'].map(drug_map)
        df_notes['label_drug'] = df_notes['label_drug'].apply(lambda x: x if isinstance(x, list) else [])
        df_notes['label_drug'] = df_notes['label_drug'].apply(lambda x: list(set(x)))
        return df_notes

    def create_balanced_corpora(self, df, output_disease_path, output_drug_path):

        with_disease = df[df["label_disease"].apply(lambda x: len(x) > 0)]
        without_disease = df[df["label_disease"].apply(lambda x: len(x) == 0)]
        n = min(len(with_disease), len(without_disease))

        corpus_disease = pd.concat([
            with_disease.sample(n, random_state=42),
            without_disease.sample(n, random_state=42)
        ]).sample(frac=1, random_state=42)  # shuffle
        corpus_disease = corpus_disease.rename(columns={"text": "documents", "note_id": "id"})

        corpus_disease.to_csv(output_disease_path, index=False, compression="gzip")
        
        with_drug = df[df["label_drug"].apply(lambda x: len(x) > 0)]
        without_drug = df[df["label_drug"].apply(lambda x: len(x) == 0)]
        n = min(len(with_drug), len(without_drug))

        corpus_drug = pd.concat([
            with_drug.sample(n, random_state=42),
            without_drug.sample(n, random_state=42)
        ]).sample(frac=1, random_state=42)  # shuffle

        corpus_drug = corpus_drug.rename(columns={"text": "documents", "note_id": "id"})

        corpus_drug.to_csv(output_drug_path, index=False, compression="gzip")
        return corpus_disease, corpus_drug


def main():
    #acepts the argument --config_file (the parameters json file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        params_path = json.load(f)

    preprocessing = Preprocessing(params_path)

    df_diseases = preprocessing.extract_diseases(
        params_path['dimension_table_diseases_path'],
        params_path['fact_table_diseases_path'],
        params_path['diseases_targets_path'],
        params_path['summary_path']
    )

    df_drugs = preprocessing.extract_drugs(
        params_path['fact_table_drugs_path'],
        params_path['drugs_targets_path'],
        params_path['summary_path']
    )

    df_notes = preprocessing.filter_notes_by_labels(params_path['notes_path'], df_diseases, df_drugs)

    df_notes_filtered = preprocessing.clean_text(df_notes)
    logging.info(f"\nHead of notes filtered: {df_notes_filtered.head()}")

    # save the data
    df_notes_filtered.to_csv(params_path['output_path'], index=False, compression='gzip')

    preprocessing.plot_distribution(df_notes_filtered, params_path['summary_path'])
    preprocessing.create_balanced_corpora(
    df_notes_filtered,
    params_path['output_disease_corpus'],
    params_path['output_drug_corpus']
    )


if __name__ == '__main__':
    main()
