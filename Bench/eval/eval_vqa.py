import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.eval.metrics import compute_exact_match, qa_f1_score
from LaMed.src.dataset.multi_dataset import VQABratsDataset
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM


bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="GoodBaiBai88/M3D-LaMed-Llama-2-7B")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--vqa_data_test_path', type=str, default="/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/eval_vqa/")

    parser.add_argument('--proj_out_num', type=int, default=1024)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def get_tokenizer(model_path):
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer
        
          
def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = get_tokenizer(args.model_name_or_path)
    """
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        trust_remote_code=True
    )
    """
    if 'llama' in args.model_name_or_path.lower():
        model = LamedLlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
        )
    elif 'phi3' in args.model_name_or_path.lower():
        model = LamedPhi3ForCausalLM.from_pretrained(
            args.model_name_or_path,
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")
    print("model: ", model.__class__)
    model = model.to(device=device)

    #test_dataset = VQADataset(args, tokenizer=tokenizer, close_ended=args.close_ended, mode='test')
    test_dataset = VQABratsDataset(args, tokenizer=tokenizer, mode='test')

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=32,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, "eval_open_vqa.csv")
    print(f"Output path: {output_path}")
    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question Type", "Question", "Answer", "Pred", "accuracy", "bleu", "rouge1", "meteor", "bert_f1"])
        for sample in tqdm(test_dataloader):
            question = sample["question"][0]
            question_type = sample["question_type"][0]
            answer = sample['answer']

            image = sample["image"].to(device=device)
            input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)

            with torch.inference_mode():
                generation = model.generate(images=image, inputs=input_id, max_new_tokens=args.max_new_tokens,
                                            do_sample=args.do_sample, top_p=args.top_p,
                                            temperature=args.temperature)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            result = dict()
            decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)

            result["accuracy"] = compute_exact_match(decoded_preds, decoded_labels)

            try:
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            except:
                bleu_score = {'bleu': np.nan}
            result["bleu"] = bleu_score['bleu']

            try:
                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
            except:
                rouge_score = {'rouge1': np.nan}
            result["rouge1"] = rouge_score['rouge1']

            try:
                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            except:
                meteor_score = {'meteor': np.nan}
            result["meteor"] = meteor_score['meteor']

            try:
                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])
            except:
                result["bert_f1"] = np.nan

            writer.writerow(
                [question_type, question, answer[0], generated_texts[0], result["accuracy"], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"]])

if __name__ == "__main__":
    main()
       