import re
import warnings
from collections import Counter
from flashrag.evaluator.utils import normalize_answer


class BaseMetric:
    metric_name = "base"
    def __init__(self, config):
        self.config = config

    def calculate_metric(self, data):
        return {}, []


class ExactMatch(BaseMetric):
    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            golden_answer = normalize_answer(golden_answer)
            if golden_answer == normalized_prediction:
                score = 1.0
                break
        return score

    def calculate_metric(self, pred_list, answer_list):

        metric_score_list = [
            self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, answer_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return em_score, metric_score_list


class Sub_ExactMatch(BaseMetric):
    metric_name = "acc"

    def __init__(self, config):
        super().__init__(config)

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            golden_answer = normalize_answer(golden_answer)
            if golden_answer in normalized_prediction:
                score = 1.0
                break
        return score

    def calculate_metric(self, pred_list, answer_list):

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, answer_list)
        ]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return sub_em_score, metric_score_list


class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: str):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, pred_list, answer_list):

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, answer_list)
        ]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return f1, metric_score_list


class Recall_Score(F1_Score):
    """Token-level Recall score"""

    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, pred_list, answer_list):
        
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, answer_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return precision, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, pred_list, answer_list):
        
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, answer_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return precision, metric_score_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"

    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge

        self.scorer = Rouge()

    def calculate_rouge(self, pred, answer_list):
        output = {}
        for answer in answer_list:
            scores = self.scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        return output


class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, pred_list, answer_list):

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(pred_list, answer_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return score, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, pred_list, answer_list):

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(pred_list, answer_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return score, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, pred_list, answer_list):

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(pred_list, answer_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return score, metric_score_list


class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a

        self.tokenizer = Tokenizer13a()
        self.max_order = config["metric_setting"].get("bleu_max_order", 4)
        self.smooth = config["metric_setting"].get("bleu_smooth", False)

    def calculate_metric(self, pred_list, answer_list):
        from ._bleu import compute_bleu

        pred_list = [self.tokenizer(pred) for pred in pred_list]
        golden_answers_list = [
            [self.tokenizer(ans) for ans in golden_answers] for golden_answers in answer_list
        ]
        score = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth,
        )
        (total_bleu, precisions, bp, ratio, translation_length, reference_length) = score

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            pred = [pred]
            golden_answers = [golden_answers]
            score = compute_bleu(
                reference_corpus=golden_answers_list,
                translation_corpus=pred_list,
                max_order=self.max_order,
                smooth=self.smooth,
            )
            (bleu, precisions, bp, ratio, translation_length, reference_length) = score
            score_list.append(bleu)

        return total_bleu, score_list



metric_dict = {
    'em' : ExactMatch,
    'f1' : F1_Score,
    'acc' : Sub_ExactMatch,
    'recall' : Recall_Score,
    'precision' : Precision_Score,
    'rouge_1' : Rouge_1,
    'rouge_2' : Rouge_2,
    'rouge_l' : Rouge_L,
    "bleu" : BLEU
}