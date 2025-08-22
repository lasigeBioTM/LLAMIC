import json
import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tabulate import tabulate
import ast

class RelationEvaluator:
    def __init__(self, min_support: int = 10):
        self.min_support = min_support

    def _safe_parse_relations(self, rels):
        """Ensure relations is a Python list of triples."""
        if isinstance(rels, str):
            try:
                rels = ast.literal_eval(rels)
            except Exception:
                return []
        return rels if isinstance(rels, list) else []

    def extract_tagged_chunks(self, text):
        pattern = re.compile(r'(<(disease|drug)\d+>.*?</\2\d+>)', re.IGNORECASE)
        return [match[0] for match in pattern.findall(text)]

    def create_all_possible_pairs(self, entities):
        """Create all unique pairs of entities."""
        pairs = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pairs.append((entities[i], entities[j]))
        return pairs
    def order_pairs(self, pairs):
        """ Order pairs disease-disease, drug-drug, disease-drug consistently."""
        """And also disease1, disase2, drug1, drug2"""
        ordered_pairs = []
        for ent1, ent2 in pairs:
            if "<disease" in ent1 and "<disease" in ent2:
                ordered_pairs.append((ent1, ent2))
            elif "<drug" in ent1 and "<drug" in ent2:
                ordered_pairs.append((ent1, ent2))
            elif ("<disease" in ent1 and "<drug" in ent2) or ("<drug" in ent1 and "<disease" in ent2):
                ordered_pairs.append((ent1, ent2))
        return ordered_pairs
    def is_the_same_triplet(self, triplet1, triplet2):
        """Check if two triplets are the same regardless of order."""
        return (triplet1[0] == triplet2[0] and triplet1[2] == triplet2[1]) or \
               (triplet1[0] == triplet2[1] and triplet1[2] == triplet2[0])

    def is_the_same_triplet(self, triplet1, triplet2):
        ent1_1, rel1, ent1_2 = triplet1
        ent2_1, ent2_2 = triplet2
        return (ent1_1 == ent2_1 and ent1_2 == ent2_2) or (ent1_1 == ent2_2 and ent1_2 == ent2_1)
        
    def _extract_labels(self, df_true: pd.DataFrame, pred_data: list):
        pred_map = {p["id"]: self._safe_parse_relations(p["relations"])
                    for p in pred_data if p["id"] in df_true["id"].values}

        y_true_labels = []
        y_pred_labels = []

        for idx, row in df_true.iterrows():
            true_relations = self._safe_parse_relations(row["relations"])
            text = row.get("documents", "")
            all_entities = self.extract_tagged_chunks(text)
            all_pairs = self.create_all_possible_pairs(all_entities)
            all_pairs = self.order_pairs(all_pairs)


            pred_relations = pred_map.get(row["id"], [])

            true_labels = []
            pred_labels = []
            for pair in all_pairs:
                true_triplet = next((t for t in true_relations if self.is_the_same_triplet(t, pair)), None)
                pred_triplet = next((t for t in pred_relations if self.is_the_same_triplet(t, pair)), None)

                if true_triplet and true_triplet[1] != "no_relation":
                    true_labels.append(true_triplet[1])
                else:
                    true_labels.append("NA")

                if pred_triplet and pred_triplet[1] != "no_relation":
                    pred_labels.append(pred_triplet[1])
                else:
                    pred_labels.append("NA")
                    
            # Pad to equal length for macro per-relation evaluation
            max_len = max(len(true_labels), len(pred_labels))
            if len(true_labels) < max_len:
                true_labels += ["NA"] * (max_len - len(true_labels))
                # print(f"  Padded Gold labels: {true_labels}")
            if len(pred_labels) < max_len:
                pred_labels += ["NA"] * (max_len - len(pred_labels))
                # print(f"  Padded Pred labels: {pred_labels}")

            y_true_labels.extend(true_labels)
            y_pred_labels.extend(pred_labels)

        return y_true_labels, y_pred_labels
    def _extract_labels_binary(self, df_true: pd.DataFrame, pred_data: list):
        pred_map = {p["id"]: self._safe_parse_relations(p["relations"])
                    for p in pred_data if p["id"] in df_true["id"].values}

        y_true_labels = []
        y_pred_labels = []

        for idx, row in df_true.iterrows():
            true_relations = self._safe_parse_relations(row["relations"])
            text = row.get("documents", "")
            all_entities = self.extract_tagged_chunks(text)
            all_pairs = self.create_all_possible_pairs(all_entities)
            all_pairs = self.order_pairs(all_pairs)

            pred_relations = pred_map.get(row["id"], [])

            true_labels = []
            pred_labels = []
            for pair in all_pairs:
                true_triplet = next((t for t in true_relations if self.is_the_same_triplet(t, pair)), None)
                pred_triplet = next((t for t in pred_relations if self.is_the_same_triplet(t, pair)), None)

                # True labels
                if true_triplet and true_triplet[1] not in ["no_relation", "NA"]:
                    true_labels.append("related to")
                else:
                    true_labels.append("NA")

                # Predicted labels
                if pred_triplet and pred_triplet[1] not in ["no_relation", "NA"]:
                    pred_labels.append("related to")
                else:
                    pred_labels.append("NA")

            # Pad to equal length
            max_len = max(len(true_labels), len(pred_labels))
            if len(true_labels) < max_len:
                true_labels += ["NA"] * (max_len - len(true_labels))
            if len(pred_labels) < max_len:
                pred_labels += ["NA"] * (max_len - len(pred_labels))

            y_true_labels.extend(true_labels)
            y_pred_labels.extend(pred_labels)

        return y_true_labels, y_pred_labels

    def _calc_metrics_per_class(self, y_true, y_pred):
        classes = sorted(set(y_true) | set(y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=classes
        )
        return pd.DataFrame({
            'class': classes,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        })

    def _calc_macro_metrics_filtered(self, y_true, y_pred):
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        filtered_classes = [
            label for label, stats in report.items()
            if isinstance(stats, dict) and stats.get('support', 0) >= self.min_support
        ]
        precisions = [report[label]['precision'] for label in filtered_classes]
        recalls = [report[label]['recall'] for label in filtered_classes]
        f1s = [report[label]['f1-score'] for label in filtered_classes]
        return {
            'precision_macro': sum(precisions) / len(precisions) if precisions else 0,
            'recall_macro': sum(recalls) / len(recalls) if recalls else 0,
            'f1_macro': sum(f1s) / len(f1s) if f1s else 0
        }

    def evaluate(self, df_true: pd.DataFrame, pred_data: list):

        y_true_labels_b, y_pred_labels_b = self._extract_labels_binary(df_true, pred_data)          
        macro_default = self._calc_macro_metrics_filtered(y_true_labels_b, y_pred_labels_b)

        y_true_labels, y_pred_labels = self._extract_labels(df_true, pred_data)
        metrics_default = self._calc_metrics_per_class(y_true_labels, y_pred_labels)

        print("\n=== Binary Metrics ===")
        print(tabulate(pd.DataFrame([macro_default]), headers="keys", tablefmt="grid", floatfmt=".3f"))
        print("\n=== Per-Class Metrics ===")
        print(tabulate(metrics_default, headers="keys", tablefmt="grid", floatfmt=".3f"))


