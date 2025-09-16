import os
import json
import re
import evaluate
from collections import Counter
from utils import parser_number

# 加载提示词列表（每个 txt 文件）
def load_prompts(prompt_dir):
    def readfile(name):
        return list(map(str.strip, open(os.path.join(prompt_dir, name)).readlines()))

    return {
        'whatyouhear': readfile('whatyouhear.txt'),
        'a_what': readfile('a_whatyouhear.txt'),
        'transeng': readfile('transctibe_english.txt'),
        'dialect': readfile('dialect.txt'),
        'disorders': readfile('disorder.txt'),
        'a_nodisorders': readfile('a_nodisorder.txt'),
        'q_age': readfile('age_q.txt'),
        'q_gender': readfile('gender_q.txt'),
        'a_gender_boy': readfile('gender_boy.txt'),
        'a_gender_girl': readfile('gender_girl.txt'),
        'binary_disorder': readfile('disorder_binary_q.txt'),
        'binary_disorder_y': readfile('binary_disorder_binary_isdis.txt'),
        'binary_disorder_n': readfile('binary_disorder_binary_nodis.txt'),
        'multi_disorder': readfile('disorder_multi_q.txt'),
    }

# 提取参与转录任务的索引
def extract_disorder_transcript_labels(test_json, binary_disorder, binary_disorder_n, transeng):
    with open(test_json, 'r') as f:
        dataset = json.load(f)

    disorder_label_set = set()
    st = 0
    for sample in dataset:
        record_trans = False
        for i in range(len(sample['conversation'])):
            if sample['conversation'][i]['value'] in binary_disorder:
                if sample['conversation'][i+1]['value'] in binary_disorder_n:
                    record_trans = True
        for i in range(len(sample['conversation'])):
            if sample['conversation'][i]['value'] in transeng and record_trans:
                disorder_label_set.add(st + int(i / 2))
        st += int(len(sample['conversation']) / 2)
    return disorder_label_set

# 处理单条数据
def evaluate_line(prompt, answer, gt, idx, disorder_label_set, prompts, stats):
    """
    stats：记录各种计数的状态字典
    """
    if prompt in prompts['whatyouhear']:
        stats['gen_des_total'] += 1
        for a in ['boy', 'girl']:
            if a in answer:
                stats['gen_total'] += 1
                if a in gt:
                    stats['gen_correct'] += 1
                    stats['gen_des_correct'] += 1

    elif prompt in prompts['q_gender']:
        stats['gen_total'] += 1
        boy_words = ['boy', 'male']
        girl_words = ['girl', 'female']
        an_l, gt_l = None, None
        for b in boy_words:
            if b in answer:
                an_l = 'b'
        for g in girl_words:
            if g in answer:
                an_l = 'g'
        for b in boy_words:
            if b in gt:
                gt_l = 'b'
        for g in girl_words:
            if g in gt:
                gt_l = 'g'
        if an_l and gt_l:
            stats['gen_correct'] += int(an_l == gt_l)

    elif prompt in prompts['dialect']:
        stats['dial_total'] += 1
        if "The speaker sounds to be " in answer:
            dialect_region = answer.split('The speaker sounds to be ')[1].strip()
        else:
            dialect_region = "unknown"
        dialect_gt = gt.split('The speaker sounds to be ')[1].strip()
        stats['dial_correct'] += int(dialect_region == dialect_gt)

    elif prompt in prompts['binary_disorder']:
        stats['dis_total'] += 1
        if answer in prompts['binary_disorder_n'] and gt in prompts['binary_disorder_n']:
            stats['dis_correct'] += 1
        elif answer in prompts['binary_disorder_y'] and gt in prompts['binary_disorder_y']:
            stats['dis_correct'] += 1

    elif prompt in prompts['multi_disorder']:
        stats['multi_dis_total'] += 1
        disorder_keywords = [
            'inconsistent phonological disorder', 'phonological disorder',
            'childhood apraxia of speech', 'phonological delay',
            'vowel disorder', 'articulation disorder'
        ]
        for dis in disorder_keywords:
            if dis in answer and dis in gt:
                stats['multi_dis_correct'] += 1

    elif prompt in prompts['transeng']:
        if idx not in disorder_label_set:
            return
        try:
            trans = answer.split("This is the english transcription, ")[1].strip()
        except:
            trans = answer.strip()
        try:
            gt_trans = gt.split("This is the english transcription, ")[1].strip()
        except:
            gt_trans = gt.strip()
        if trans and gt_trans:
            stats['trans_eng'].append(trans)
            stats['trans_gt'].append(gt_trans)

    elif prompt in prompts['q_age']:
        stats['age_total'] += 1
        an_age = ''.join(re.findall(r'\d+', answer)) or parser_number(answer) or -1
        gt_age = ''.join(re.findall(r'\d+', gt)) or parser_number(gt) or -1
        try:
            an_age = int(an_age)
            gt_age = int(gt_age)
        except:
            return
        stats['age_values'].append(gt_age)
        mid = 5
        if 0 < an_age <= mid and gt_age <= mid:
            stats['age_range_correct'] += 1
        elif an_age > mid and gt_age > mid:
            stats['age_range_correct'] += 1
        if an_age == gt_age:
            stats['age_exact_correct'] += 1

# 主评估函数
def evaluate_model_output(full_file, test_json, prompt_dir):
    prompts = load_prompts(prompt_dir)
    disorder_label_set = extract_disorder_transcript_labels(
        test_json, prompts['binary_disorder'], prompts['binary_disorder_n'], prompts['transeng']
    )
    with open(full_file) as f:
        lines = f.readlines()

    stats = {
        'gen_total': 0, 'gen_correct': 0, 'gen_des_total': 0, 'gen_des_correct': 0,
        'dial_total': 0, 'dial_correct': 0,
        'dis_total': 0, 'dis_correct': 0,
        'multi_dis_total': 0, 'multi_dis_correct': 0,
        'age_total': 0, 'age_exact_correct': 0, 'age_range_correct': 0,
        'trans_eng': [], 'trans_gt': [], 'age_values': []
    }

    for idx, line in enumerate(lines):
        splits = line.strip().split('|')
        if len(splits) != 3:
            print(f"⚠️ Warning: Skipping line {idx} due to unexpected format.")
            continue
        prompt, answer, gt = map(str.strip, splits)
        evaluate_line(prompt, answer, gt, idx, disorder_label_set, prompts, stats)

    metric = evaluate.load("wer")
    metric_c = evaluate.load("cer")
    wer = round(100 - 100 * metric.compute(predictions=stats['trans_eng'], references=stats['trans_gt']), 2)
    cer = round(100 - 100 * metric_c.compute(predictions=stats['trans_eng'], references=stats['trans_gt']), 2)

    # 结果统计
    result = {
        'gender': round(stats['gen_correct'] / stats['gen_total'] * 100, 2) if stats['gen_total'] else 0.0,
        'binary_disorder': round(stats['dis_correct'] / stats['dis_total'] * 100, 2) if stats['dis_total'] else 0.0,
        'multi_disorder': round(stats['multi_dis_correct'] / stats['multi_dis_total'] * 100, 2) if stats['multi_dis_total'] else 0.0,
        'dialect': round(stats['dial_correct'] / stats['dial_total'] * 100, 2) if stats['dial_total'] else 0.0,
        'wer': wer,
        'cer': cer,
    }
    if stats['age_total']:
        result['age'] = round(stats['age_exact_correct'] / stats['age_total'] * 100, 2)
        result['age_range'] = round(stats['age_range_correct'] / stats['age_total'] * 100, 2)

    with open('result_all.json', 'w') as f:
        json.dump(result, f, indent=4)
    return result

if __name__ == "__main__":
    full_dir = '/home/jingchen/kidspeak/code/result/qwen-audio/epoch_10.txt'
    test_file = '/home/jingchen/PandaGPT_expts/PandaGPT_ours/rough/merged/kids_full_new_test_1_3_bi_multi_disorder.json'
    prompt_dir = '/home/jingchen/kidspeak/dataset/prompts'  # 你的 prompts 文件夹

    results = evaluate_model_output(full_dir, test_file, prompt_dir)
    print("📊 Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}%")
