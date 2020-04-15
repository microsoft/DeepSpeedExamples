"""
Official evaluation script for v1.1 of the SQuAD dataset.
Also added other defined metric functions.
"""
import json
import re
import string
import sys
import torch
import numpy as np
from collections import Counter


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    :param s: original string
    :return: normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score given prediction and true answer strings.
    :param prediction: prediction string
    :param ground_truth: answer string
    :return: F1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Calculate exact match score given prediction and true answer strings.
    :param prediction: prediction string
    :param ground_truth: answer string
    :return: EM score
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Calculate the maximum metric value when we have multiple ground truths.
    i.e., for each question, we have multiple answers.
    :param metric_fn: the function to calculate metric
    :param prediction: our model predicted answer string
    :param ground_truths: the list of answer strings
    :return: the maximum metric value by comparing our prediction
             to each ground_truth
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    """
    Evaluate performance, calculate metrics EM and F1.
    :param dataset: the dictionary of 'data' in json file.
    :param predictions: the dictionary of our predictions.
                        (k, v) is like (qa['id'], prediction string)
    """
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def evaluate_from_file(dataset_file, prediction_file):
    """
    Load dataset and prediction from two files, and evaluate
    the performance.
    """
    expected_version = '1.1'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    return evaluate(dataset, predictions)


def em_by_begin_end_index(pred_begin, pred_end, begin, end):
    """
    Calculate exact match score given the token index tensors of
    prediction boundary and true answer boundary.
    """
    batch_num = pred_begin.size(0)
    exact_correct_num = torch.sum(
        (pred_begin == begin) * (pred_end == end))
    em = exact_correct_num.item() / batch_num
    return em


def f1_by_begin_end_index(pred_begin, pred_end, begin, end):
    """
    Calculate F1 score given the token index tensors of
    prediction boundary and true answer boundary.
    """
    batch_size = pred_begin.size(0)
    f1_all = []
    for i in range(batch_size):
        pred = range(int(pred_begin[i]), int(pred_end[i] + 1))
        truth = range(int(begin[i]), int(end[i] + 1))
        overlap_len = len(list(set(pred) & set(truth)))
        pred_len = pred_end[i] - pred_begin[i] + 1
        truth_len = end[i] - begin[i] + 1

        precision = overlap_len / pred_len
        recall = overlap_len / truth_len
        if overlap_len == 0:
            f1 = 0
        else:
            f1 = ((2 * precision * recall) / (precision + recall)).item()
        f1_all.append(f1)
    f1 = np.mean(f1_all)
    return f1


def em_by_begin_end_index_max(pred_begin, pred_end, begins, ends):
    batch_size = len(pred_begin)
    em_all = []
    for i in range(batch_size):
        num_answers = len(begins[i])
        em = []
        for j in range(num_answers):
            em.append((pred_begin[i] == begins[i][j]) *
                      (pred_end[i] == ends[i][j]))
        em_all.append(max(em))
    return np.mean(em_all)


def f1_by_begin_end_index_max(pred_begin, pred_end, begins, ends):
    batch_size = len(pred_begin)
    f1_all = []
    for i in range(batch_size):
        num_answers = len(begins[i])
        f1 = []
        for j in range(num_answers):
            pred = range(int(pred_begin[i]), int(pred_end[i] + 1))
            truth = range(int(begins[i][j]), int(ends[i][j] + 1))
            overlap_len = len(list(set(pred) & set(truth)))
            pred_len = pred_end[i] - pred_begin[i] + 1
            truth_len = ends[i][j] - begins[i][j] + 1
            precision = overlap_len / pred_len
            recall = overlap_len / truth_len
            if overlap_len == 0:
                f1_ = 0
            else:
                f1_ = ((2 * precision * recall) / (precision + recall))
            f1.append(f1_)
        f1_all.append(max(f1))
    return np.mean(f1_all)


def convert_tokens(eval_dict, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_dict[str(qid)]["context"]
        spans = eval_dict[str(qid)]["spans"]
        uuid = eval_dict[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate_by_dict(eval_dict, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_dict[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


if __name__ == "__main__":
    pred_begin = [1, 0, 1]
    pred_end = [3, 3, 3]
    begins = [[1, 0], [1, 0], [1, 1]]
    ends = [[3, 2], [3, 3], [3, 3]]
    print(em_by_begin_end_index_max(pred_begin, pred_end, begins, ends))
    print(f1_by_begin_end_index_max(pred_begin, pred_end, begins, ends))
