import json


def main(pred_file):
    with open(pred_file, "r") as f:
        data = json.load(f)

    # Dictionary to collect counts:
    # {
    #   (label, content): {"correct": int, "total": int}
    # }
    counts = {}

    for item in data:
        label = (item.get("label_name") or "").strip().lower()
        content = (item.get("content_type") or "").strip().lower()
        pred = (item.get("pred") or "").strip().lower()
        answer = (item.get("answer") or "").strip().lower()

        key = (label, content)
        if key not in counts:
            counts[key] = {"correct": 0, "total": 0}

        counts[key]["total"] += 1
        if pred == answer:
            counts[key]["correct"] += 1

    # Now calculate accuracy per (label, content).
    # We'll store results in a nested dict: results[label][content] = accuracy
    results = {}
    for (label, content), vals in counts.items():
        correct = vals["correct"]
        total = vals["total"]
        accuracy = correct / total if total else 0.0

        if label not in results:
            results[label] = {}
        results[label][content] = accuracy

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str,
                        default="/local2/amvepa91/M3D/LaMed/output/LaMed-Phi3-4B-finetune-freeze-brats-ped-viz-0000/eval_vqa/eval_vqa.json",
                        help="Path to predictions JSON file")
    args = parser.parse_args()
    pred_file = args.pred_file
    main(pred_file=pred_file)
