import pandas as pd
import json
import os
import bent.annotate as bt
import re
from owlready2 import get_ontology

class Ontology:
    def __init__(self):
        self.doid = None

    def load_ontology(self):
        """
        Loads Disease Ontology (DOID) in OWL format.
        """
        self.doid = get_ontology("lexicon/doid_22_01_2021.owl").load()
        if self.doid is None:
            print("Erro ao carregar a ontologia DOID.")

    @staticmethod
    def expand_icd_range(code):    
        if "-" not in code:
            return [code.split(".")[0].strip()]

        match = re.match(r"([A-Za-z]*)(\d+)(?:\.\d+)?-([A-Za-z]*)(\d+)(?:\.\d+)?", code)
        if match:
            prefix1, start, prefix2, end = match.groups()

            if not prefix2:
                prefix2 = prefix1

            if prefix1 != prefix2:
                raise ValueError("Intervalo inválido: prefixos diferentes não são suportados.")

            return [f"{prefix1}{i}" for i in range(int(start), int(end) + 1)]

        return []

    def get_icd_from_doid(self, doid_url):
        if self.doid is None:
            print("Erro: A ontologia DOID ainda não foi carregada. Execute `load_ontology()` primeiro.")
            return None

        results = self.doid.search(iri=doid_url)
        if not results:
            print(f"DOID {doid_url} não encontrado na ontologia.")
            return None

        term = results[0]
        try:
            ancestors = term.ancestors()
        except Exception as e:
            print(f"Erro ao obter ancestrais para {doid_url}: {e}")
            ancestors = []

        allowed_prefixes = ("ICD10", "ICD10CM", "*ICD10", "*ICD10CM")
        icd_codes = []

        for ancestor in ancestors:
            flag = 0
            if ancestor.iri == doid_url:
                flag = 1
            
            xrefs = getattr(ancestor, 'hasDbXref', [])
            for xref in xrefs:
                if isinstance(xref, str) and xref.startswith(allowed_prefixes):
                    icd_version = xref.split(":", 1)[0]
                    code = xref.split(":", 1)[1]
                    code = self.expand_icd_range(code)
                    if not code:
                        print(f"Código ICD inválido: {xref}")
                    else:
                        for c in code:
                            icd_codes.append([flag, c])

        icd_codes_flag_zero = [code[1] for code in icd_codes if code[0] == 0]
        icd_codes_flag_one = [code[1] for code in icd_codes if code[0] == 1]

        icd_codes_zero = list(set(icd_codes_flag_zero))
        icd_codes_one = list(set(icd_codes_flag_one))

        if len(icd_codes_one) == 1:
            return icd_codes_one[0]
        else:
            return icd_codes_one

class DrugNERProcessor:
    def __init__(self, config_file):
        """Initializes the processor with a configuration file."""
        self.load_config(config_file)
        self.df_test = None
        self.data_desc = None
        self.list_total = []
        self.load_data()

    def load_config(self, config_file):
        """Loads the configuration from a JSON file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.notes_csv = config['notes_csv']
        self.splits_json = config['splits_json']
        self.output_dir = config['output_dir']
        self.output_file = config.get('output_file', 'bent_do_ner.json')
        self.type = config['type']

    def load_data(self):
        """Loads and processes the necessary data."""
        df = pd.read_csv(self.notes_csv)
        df['hadm_id'] = df['hadm_id'].astype(str)
        
        if self.splits_json == "":
            test_hadm_ids = set(df['hadm_id'].unique())
        else:
            with open(self.splits_json, 'r') as f:
                data_splits = json.load(f)
            if 'test' not in data_splits:
                raise KeyError("The key 'test' was not found in data_splits.json")
            test_hadm_ids = set(map(str, data_splits['test']))
        
        if self.type == 'drug':
            required_columns = {'hadm_id', 'hospital_course', 'label_drug'}
        elif self.type == 'disease':
            required_columns = {'hadm_id', 'hospital_course', 'label_disease'}
        else:
            required_columns = {'hadm_id', 'hospital_course'}
        
        if not required_columns.issubset(df.columns):
            raise ValueError(f"The CSV must contain the columns: {required_columns}")
        
        self.df_test = df[df['hadm_id'].isin(test_hadm_ids)]


    def run_ner(self):
        """Runs NER and saves the results (simulation, replace with actual model)."""
        if self.type == 'drug':
            bt.annotate(
                recognize=True,
                link=True,
                types={'chemical': 'ctd_chem'},
                input_text=self.df_test['hospital_course'].tolist(),
                out_dir=self.output_dir
            )
        elif self.type == 'disease':
            bt.annotate(
                recognize=True,
                link=True,
                types={'disease': 'do'},
                input_text=self.df_test['hospital_course'].tolist(),
                out_dir=self.output_dir
            )

    def extract_annotations(self):
        """Extracts disease and drug information from annotated files."""
        ids = self.df_test['hadm_id'].tolist()
        if self.type == 'drug':
            drugs_list = self.df_test['label_drug'].tolist()
        elif self.type == 'disease':
            drugs_list = self.df_test['label_disease'].tolist()

        if self.type == 'disease':
            ontology = Ontology()
            ontology.load_ontology()

        for i, hadm_id in enumerate(ids):
            try:
                entities_predicted = []
                file_path = os.path.join(self.output_dir, f'doc_{i+1}.ann')
                entities_info = {}

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split('\t')
                    if parts[0].startswith('T'):
                        tid = parts[0]
                        type_info = parts[1].split(' ')
                        start = int(type_info[1])
                        end = int(type_info[2])
                        text = parts[2]
                        entities_info[tid] = {"entity": text, "start": start, "end": end}
                        
                mesh_targets = ['D001241', 'D006493', 'D017984', 'D005996', 'D009543']
                icd_targets = ['I20','I21','I22','I23','I24','I25','I60','I61','I62','I63','I64']
                for line in lines:
                    parts = line.strip().split('\t')
                    if parts[0].startswith('N'):
                        ref_parts = parts[1].split(' ')
                        t_ref = ref_parts[1]  # Ex: T1
                        id_t = ref_parts[2]
                        if t_ref not in entities_info:
                            continue

                        ent = entities_info[t_ref]
                        name = ent["entity"]
                        start = ent["start"]
                        end = ent["end"]

                        if id_t.startswith('MESH:'):
                            id_t = id_t[5:]
                            if id_t in mesh_targets:
                                entities_predicted.append({
                                    "entity": name,
                                    "id": id_t,
                                    "index_start": start,
                                    "index_end": end
                                })
                        elif id_t.startswith('DOID:'):
                            doid = id_t[5:]
                            icd = ontology.get_icd_from_doid(f'http://purl.obolibrary.org/obo/DOID_{doid}')
                            if isinstance(icd, list):
                                for c in icd:
                                    if c in icd_targets:
                                        entities_predicted.append({
                                            "entity": name,
                                            "id": c,
                                            "doid": doid,
                                            "index_start": start,
                                            "index_end": end
                                        })
                            elif icd in icd_targets:
                                entities_predicted.append({
                                    "entity": name,
                                    "id": icd,
                                    "doid": doid,
                                    "index_start": start,
                                    "index_end": end
                                })
          
                if self.type == 'drug':
                    self.list_total.append({
                        'id': int(hadm_id),
                        'drugs_true': drugs_list[i],
                        'entities': entities_predicted
                    })
                elif self.type == 'disease':
                    self.list_total.append({
                        'id': int(hadm_id),
                        'diseases_true': drugs_list[i],
                        'entities': entities_predicted
                    })

            except Exception as e:
                print(f"Error in hadm_id {hadm_id}: {e}")

    def save_results(self):
        """Saves the results in a JSON file."""
        with open(self.output_file, 'w') as f:
            output_data = {"results": self.list_total} 
            json.dump(output_data, f, indent=4)

    def run(self):
        """Runs the entire pipeline."""
        self.run_ner()
        self.extract_annotations()
        self.save_results()


def main(config_file):
    """Main function to initialize and run the processor."""
    processor = DrugNERProcessor(config_file)
    processor.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])


# python3 bent.py config.json
