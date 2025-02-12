import csv
import argparse
import os
from tqdm import tqdm
import json
import numpy as np
from collections import Counter


def main(vqa_data_test_path, output_dir):
    with open(vqa_data_test_path, 'r') as f:
        data_list = json.load(f)
    content = []
    input_eval_path = os.path.join(output_dir, "eval_open_vqa.csv")
    output_eval_path = os.path.join(output_dir, "eval_vqa.json")
    output_eval_summary_path = os.path.join(output_dir, "eval_vqa_summary.json")
    with open(input_eval_path, mode='r') as infile:
        reader = csv.reader(infile, delimiter=",")
        # skip first row
        next(reader)
        for row, sample in tqdm(zip(reader, data_list)):
            pred = str(row[3])
            accuracy = float(row[4])
            answer = sample['answer']
            q_lang = sample['q_lang']
            qid = sample.get('qid', None)
            volume_file_id = sample.get('volume_file_id', None)
            volume_file_dir = sample.get('volume_file_dir', None)
            study_name = sample.get('study_name', None)
            question_clean = sample.get('question_clean', None)
            content_type = sample.get('content_type', None)
            label_name = sample.get('label_name', None)
            content.append({'volume_file_id': volume_file_id, 'volume_file_dir': volume_file_dir, "accuracy": accuracy,
                            'study_name': study_name,'question_clean': question_clean, 'pred': pred, 'answer': answer,
                            'q_lang': q_lang, 'content_type': content_type, 'label_name': label_name, "qid": qid})
    with open(output_eval_path, 'w') as f:
        json.dump(content, f, indent=4)

    summary = {}
    content_scores = {}
    label_scores = {}
    most_common_answers = {}
    most_common_preds = {}
    for values in content:
        content_type = values['content_type']
        label_name = values['label_name']
        pred = values['pred']
        answer = values['answer']
        accuracy = values['accuracy']

        if content_type not in content_scores:
            content_scores[content_type] = {"accuracy": [], "none_count": []}
        content_scores[content_type]["accuracy"].append(accuracy)

        if label_name not in label_scores:
            label_scores[label_name] = {"overall": {"accuracy": [], "none_count": []}}
            most_common_answers[label_name] = {"overall": []}
            most_common_preds[label_name] = {"overall": []}
        if content_type not in label_scores[label_name]:
            label_scores[label_name][content_type] = {"accuracy": [], "none_count": []}
            most_common_answers[label_name][content_type] = []
            most_common_preds[label_name][content_type] = []
        label_scores[label_name]["overall"]["accuracy"].append(accuracy)
        most_common_answers[label_name]["overall"].append(answer)
        most_common_preds[label_name]["overall"].append(pred)
        label_scores[label_name][content_type]["accuracy"].append(accuracy)
        most_common_answers[label_name][content_type].append(answer)
        most_common_preds[label_name][content_type].append(pred)

        if answer.strip().lower() == "none":
            content_scores[content_type]["none_count"].append(1)
            label_scores[label_name]["overall"]["none_count"].append(1)
            label_scores[label_name][content_type]["none_count"].append(1)
        else:
            content_scores[content_type]["none_count"].append(0)
            label_scores[label_name]["overall"]["none_count"].append(0)
            label_scores[label_name][content_type]["none_count"].append(0)

    for content_type in content_scores.keys():
        content_scores[content_type]["accuracy"] = np.mean(content_scores[content_type]["accuracy"])
        content_scores[content_type]["none_count"] = np.mean(content_scores[content_type]["none_count"])
    for label_name in label_scores.keys():
        for content_type in label_scores[label_name].keys():
            label_scores[label_name][content_type]["accuracy"] = np.mean(label_scores[label_name][content_type]["accuracy"])
            label_scores[label_name][content_type]["none_count"] = np.mean(label_scores[label_name][content_type]["none_count"])
            answer_counter = Counter(most_common_answers[label_name][content_type])
            pred_counter = Counter(most_common_preds[label_name][content_type])
            most_common_answer, answer_count = answer_counter.most_common(1)[0]
            most_common_pred, pred_count = pred_counter.most_common(1)[0]
            label_scores[label_name][content_type]["most_common_answer"] = most_common_answer
            label_scores[label_name][content_type]["most_common_answer_count"] = answer_count
            label_scores[label_name][content_type]["most_common_pred"] = most_common_pred
            label_scores[label_name][content_type]["most_common_pred_count"] = pred_count
            label_scores[label_name][content_type]['count'] = sum(answer_counter.values())

    summary['content_scores'] = content_scores
    summary['label_scores'] = label_scores

    with open(output_eval_summary_path, 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    vqa_data_test_path = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json"
    output_dir = "./LaMed/output/LaMed-Phi3-4B-finetune-0000/eval_vqa"
    main(vqa_data_test_path=vqa_data_test_path, output_dir=output_dir)
