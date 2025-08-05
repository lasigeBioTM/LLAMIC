from typing import List, Dict, Tuple, Optional
from tabulate import tabulate


class EvaluatorNER_NEL:
    def __init__(self, predicted: List[List[Dict]], gold: List[List[Dict]]):
        self.predicted = predicted
        self.gold = gold

    def normalize_text(self, text: str) -> str:
        return text.lower().strip()

    def normalize_span(self, item: Dict) -> Optional[Tuple[int, int, str]]:
        if not isinstance(item, dict):
            return None

        try:
            start = int(item.get('start'))
            end = int(item.get('end'))
            ent = self.normalize_text(item.get('entity', ''))
        except Exception as e:
            return None

        return (start, end, ent)

    def _get_ner_tp(self, mode: str) -> Tuple[set, List[Tuple], List[Tuple]]:
        pred_list = [self.normalize_span(p) for p in self.predicted if self.normalize_span(p) is not None]
        gold_list = [self.normalize_span(g) for g in self.gold if self.normalize_span(g) is not None]

        tp = set()

        if mode == 'strict':
            pred_set = set(pred_list)
            gold_set = set(gold_list)
            tp = pred_set & gold_set

        elif mode == 'lenient':
            for p_start, p_end, p_ent in pred_list:
                for g_start, g_end, g_ent in gold_list:
                    overlap = max(0, min(p_end, g_end) - max(p_start, g_start))
                    ent_similar = (p_ent in g_ent) or (g_ent in p_ent)
                    if overlap > 0 and ent_similar:
                        tp.add((p_start, p_end, p_ent))
                        break
        else:
            raise ValueError("Modo inválido: use 'strict' ou 'lenient'")

        return tp, pred_list, gold_list

    def _evaluate_ner(self, mode: str) -> Dict:
        tp, pred_list, gold_list = self._get_ner_tp(mode)
        if not pred_list and not gold_list:
            precision = recall = f1 = 1.0
        else:
            precision = len(tp) / len(pred_list) if pred_list else 0.0
            recall = len(tp) / len(gold_list) if gold_list else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        return {
            "Mode": mode,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "Support (Correct)": len(tp),
            "Support (Predicted)": len(pred_list),
            "Support (Gold)": len(gold_list)
        }

    def _evaluate_nel(self, ner_tp: set) -> Dict:
        # Criar dicionários (span_normalizado → icd) apenas com spans válidos
        gold_dict = {
            self.normalize_span(g): g.get('icd')
            for g in self.gold if self.normalize_span(g) is not None
        }
        pred_dict = {
            self.normalize_span(p): p.get('icd')
            for p in self.predicted if self.normalize_span(p) is not None
        }

        nel_tp = sum(
            1 for span in ner_tp
            if span in gold_dict and span in pred_dict and gold_dict[span] == pred_dict[span]
        )

        total_pred = len(pred_dict)
        total_gold = len(gold_dict)

        if not total_pred and not total_gold:
            precision = recall = f1 = 1.0
        else:
            precision = nel_tp / total_pred if total_pred else 0.0
            recall = nel_tp / total_gold if total_gold else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "Support (Correct)": nel_tp,
            "Support (Predicted)": total_pred,
            "Support (Gold)": total_gold
        }

    def __str__(self):
        return f"EvaluatorNER_NEL with {len(self.predicted)} predicted and {len(self.gold)} gold entities."

    def run(self):
        ner_results_all = []
        nel_results_all = []

        n = len(self.predicted)
        for i in range(n):
            pred_i = self.predicted[i]
            gold_i = self.gold[i]

            evaluator_i = EvaluatorNER_NEL(pred_i, gold_i)

            ner_results = []
            for mode in ['strict', 'lenient']:
                ner_metrics = evaluator_i._evaluate_ner(mode)
                ner_results.append(ner_metrics)

            lenient_tp, _, _ = evaluator_i._get_ner_tp('lenient')
            nel_metrics = evaluator_i._evaluate_nel(lenient_tp)
            nel_results_all.append(nel_metrics)

            ner_results_all.append(ner_results)

        # Macro NER (para ambos modos)
        ner_mean = []
        for mode_i in [0, 1]:  # 0 = strict, 1 = lenient
            mode_label = ["strict", "lenient"][mode_i]
            mean_mode = {"Mode": mode_label}
            for key in ner_results_all[0][mode_i]:
                if key != "Mode":
                    mean_mode[key] = round(
                        sum(nr[mode_i][key] for nr in ner_results_all) / n, 4
                    )
            ner_mean.append(mean_mode)

        # Macro NEL
        keys = nel_results_all[0].keys()
        nel_mean = {
            key: round(sum(m[key] for m in nel_results_all) / n, 4)
            for key in keys
        }

        print("\n Macro NER:")
        print(tabulate(ner_mean, headers="keys", tablefmt="fancy_grid"))

        print("\n Macro NEL:")
        print(tabulate([nel_mean], headers="keys", tablefmt="fancy_grid"))
