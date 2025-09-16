# evaluate_kids_model.py
import os
import json
import argparse
import evaluate
import re
from collections import Counter
from utils import parser_number

# 加载所有 prompt 文件内容到字典中
def load_prompt_texts(prompt_paths):
    return {name: list(map(str.strip, open(path).readlines())) for name, path in prompt_paths.items()}

# 生成 disorder_label 与 disorder_label1，用于限制 transeng 评估范围
def extract_disorder_labels(test_json, transeng_prompts, binary_disorder_prompts, binary_disorder_no):
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    disorder_label = {}
    disorder_label1 = []
    offset = 0
    for sample in test_data:
        record_trans = 0
        for i in range(len(sample['conversation'])):
            if sample['conversation'][i]['value'] in binary_disorder_prompts:
                if sample['conversation'][i + 1]['value'] in binary_disorder_no:
                    record_trans = 1
        for i in range(len(sample['conversation'])):
            if sample['conversation'][i]['value'] in transeng_prompts and record_trans == 1:
                disorder_label[offset] = int(offset + i / 2 + 1) - 1
                disorder_label1.append(int(offset + i / 2 + 1) - 1)
        offset += int(len(sample['conversation']) / 2)
    return disorder_label, disorder_label1

# 判定属于哪一类
def match_disorder_label(text, disorder_yes, disorder_no):
    for p in disorder_yes:
        if p in text:
            return 'y'
    for p in disorder_no:
        if p in text:
            return 'n'
    return None

# 年龄分组函数
def get_age_group(age):
    """
    将年龄分为5个组别：
    Toddler: 0–3 岁
    Preschool: 4–5 岁
    Early school age: 6–8 岁
    Later school age: 9–12 岁
    Teenagers: 13–17 岁
    """
    if 0 <= age <= 3:
        return 'Toddler'
    elif 4 <= age <= 5:
        return 'Preschool'
    elif 6 <= age <= 8:
        return 'Early school age'
    elif 9 <= age <= 12:
        return 'Later school age'
    elif 13 <= age <= 17:
        return 'Teenagers'
    else:
        return 'Unknown'

# 解析每一条预测记录，对应不同子任务的准确率统计
def evaluate_prediction(prompt, answer, gt, index, prompts, disorder_label1, stats):
    (gen_total, gen_correct, gen_des_total, gen_des_correct,
     dis_total, dis_correct, dia_total, dia_correct,
     trans_predictions, trans_references,
     age_total, age_correct, age_exact_correct,
     dis_total1, multi_dis_total, multi_dis_correct) = stats

    # 判断生成性别任务
    if prompt in prompts['whatyouhear']:
        keywords = ['boy', 'girl']
        gen_des_total += 1
        for k in keywords:
            if k in answer:
                gen_total += 1
                gen_correct += k in gt
                gen_des_correct += k in gt

    # 判断分类性别任务
    if prompt in prompts['q_gender']:
        male_terms = ['boy', 'male']
        female_terms = ['girl', 'female']
        gen_total += 1
        an_l = gt_l = ''
        for k in male_terms:
            if k in answer:
                an_l = 'b'
        for k in female_terms:
            if k in answer:
                an_l = 'g'
        for k in male_terms:
            if k in gt:
                gt_l = 'b'
        for k in female_terms:
            if k in gt:
                gt_l = 'g'
        try:
            gen_correct += (an_l == gt_l)
        except:
            pass

    # 判断方言
    elif prompt in prompts['dialect']:
        dia_total += 1
        try:
            dialect_ans = answer.split('The speaker sounds to be ')[1].strip()
            dialect_gt = gt.split('The speaker sounds to be ')[1].strip()
            dia_correct += (dialect_ans == dialect_gt)
        except:
            pass

    # 判断二分类障碍任务
    # elif prompt in prompts['binary_disorder']:
    #     dis_total += 1
    #     if (answer in prompts['binary_disorder_n'] and gt in prompts['binary_disorder_n']) or \
    #        (answer in prompts['binary_disorder_y'] and gt in prompts['binary_disorder_y']):
    #         dis_correct += 1
    # elif prompt in prompts['binary_disorder']:
    #     dis_total += 1
    #     matched = False
    #     for p in prompts['binary_disorder_n']:
    #         if p in answer and p in gt:
    #             matched = True
    #             break
    #     for p in prompts['binary_disorder_y']:
    #         if p in answer and p in gt:
    #             matched = True
    #             break
    #     if matched:
    #         dis_correct += 1
    elif prompt in prompts['binary_disorder']:
        dis_total += 1

        ans_label = match_disorder_label(answer, prompts['binary_disorder_y'], prompts['binary_disorder_n'])
        gt_label = match_disorder_label(gt, prompts['binary_disorder_y'], prompts['binary_disorder_n'])

        if ans_label and gt_label and ans_label == gt_label:
            dis_correct += 1


    # 判断多分类障碍任务
    elif prompt in prompts['multi_disorder']:
        disorder_types = [
            'inconsistent phonological disorder',
            'phonological disorder',
            'childhood apraxia of speech',
            'phonological delay',
            'vowel disorder',
            'articulation disorder'
        ]
        multi_dis_total += 1
        for disorder in disorder_types:
            if disorder in answer:
                multi_dis_correct += disorder in gt

    # 判断英语转录任务
    elif prompt in prompts['transeng']:
        # if index not in disorder_label1:
        #     return stats
        try:
            trans = answer.split('This is the english transcription, ')[1].strip()
        except:
            try:
                trans = answer.split('This is the english transcription ')[1].strip()
            except:
                try:
                    trans = answer.split('This is the engagement-based transcription, ')[1].strip()
                except:
                    return stats
        try:
            gt_trans = gt.split('This is the english transcription, ')[1].strip()
        except:
            return stats
        trans_predictions.append(trans)
        trans_references.append(gt_trans)

    # 判断年龄任务
    elif prompt in prompts['q_age']:
        age_total += 1
        ans_age = ''.join(re.findall(r'\d+', answer)) or parser_number(answer) or -1
        gt_age = ''.join(re.findall(r'\d+', gt)) or parser_number(gt) or -1
        try:
            ans_age = int(ans_age)
            gt_age = int(gt_age)
        except:
            return stats
        # 使用新的年龄分组进行评估
        ans_age_group = get_age_group(ans_age)
        gt_age_group = get_age_group(gt_age)
        if ans_age_group == gt_age_group and ans_age_group != 'Unknown':
            age_correct += 1
        if ans_age == gt_age:
            age_exact_correct += 1

    return (gen_total, gen_correct, gen_des_total, gen_des_correct,
            dis_total, dis_correct, dia_total, dia_correct,
            trans_predictions, trans_references,
            age_total, age_correct, age_exact_correct,
            dis_total1, multi_dis_total, multi_dis_correct)

# 主函数：处理模型输出，评估各任务的准确率和误差率
def evaluate_model_output(full_file, test_file, prompt_paths):
    prompts = load_prompt_texts(prompt_paths)
    disorder_label, disorder_label1 = extract_disorder_labels(
        test_file, prompts['transeng'], prompts['binary_disorder'], prompts['binary_disorder_n']
    )
    print("DEBUG: disorder_label1 = ", disorder_label1)
    with open(full_file) as f:
        all_lines = f.readlines()

    stats = (0, 0, 0, 0, 0, 0, 0, 0, [], [], 0, 0, 0, 0, 0, 0)

    for idx, line in enumerate(all_lines):
        parts = line.strip().split('|')
        if len(parts) == 3:
            prompt, answer, gt = map(str.strip, parts)
            stats = evaluate_prediction(prompt, answer, gt, idx, prompts, disorder_label1, stats)

    (gen_total, gen_correct, gen_des_total, gen_des_correct,
     dis_total, dis_correct, dia_total, dia_correct,
     trans_predictions, trans_references,
     age_total, age_correct, age_exact_correct,
     dis_total1, multi_dis_total, multi_dis_correct) = stats

    # 加载转录评估指标
    # metric_wer = evaluate.load("wer")
    # metric_cer = evaluate.load("cer")
    # wer = round(100 - 100 * metric_wer.compute(predictions=trans_predictions, references=trans_references), 2)
    # cer = round(100 - 100 * metric_cer.compute(predictions=trans_predictions, references=trans_references), 2)
    # 加载转录评估指标
    print(f"DEBUG: trans prediction = {trans_predictions}, gt = {trans_references}")
    if trans_predictions and trans_references:
        metric_wer = evaluate.load("wer")
        metric_cer = evaluate.load("cer")
        wer = round(100 - 100 * metric_wer.compute(predictions=trans_predictions, references=trans_references), 2)
        cer = round(100 - 100 * metric_cer.compute(predictions=trans_predictions, references=trans_references), 2)
    else:
        wer = cer = 0.00

    # 汇总所有结果
    results = {
        'gender': round((gen_correct / gen_total) * 100, 2) if gen_total else 0.00,
        'binary_disorder': round((dis_correct / dis_total) * 100, 2) if dis_total else 0.00,
        'multi_disorder': round((multi_dis_correct / multi_dis_total) * 100, 2) if multi_dis_total else 0.00,
        'dialect': round((dia_correct / dia_total) * 100, 2) if dia_total else 0.00,
        'wer': wer,
        'cer': cer,
    }
    if age_total:
        results['age'] = round((age_exact_correct / age_total) * 100, 2)
        results['age_range'] = round((age_correct / age_total) * 100, 2)

    with open('result_all.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results

# CLI 接口入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_dir', type=str, required=True, help='模型输出文件路径')
    parser.add_argument('--test_file', type=str, required=True, help='JSON 测试文件路径')
    parser.add_argument('--prompt_dir', type=str, required=True, help='Prompt 文本所在目录')
    args = parser.parse_args()

    prompt_paths = {
        'whatyouhear': os.path.join(args.prompt_dir, 'whatyouhear.txt'),
        'a_what': os.path.join(args.prompt_dir, 'a_whatyouhear.txt'),
        'transeng': os.path.join(args.prompt_dir, 'transctibe_english.txt'),
        'dialect': os.path.join(args.prompt_dir, 'dialect.txt'),
        'disorders': os.path.join(args.prompt_dir, 'disorder.txt'),
        'a_nodisorders': os.path.join(args.prompt_dir, 'a_nodisorder.txt'),
        'q_age': os.path.join(args.prompt_dir, 'age_q.txt'),
        'q_gender': os.path.join(args.prompt_dir, 'gender_q.txt'),
        'a_gender_boy': os.path.join(args.prompt_dir, 'gender_boy.txt'),
        'a_gender_girl': os.path.join(args.prompt_dir, 'gender_girl.txt'),
        'binary_disorder': os.path.join(args.prompt_dir, 'disorder_binary_q.txt'),
        'binary_disorder_y': os.path.join(args.prompt_dir, 'binary_disorder_binary_isdis.txt'),
        'binary_disorder_n': os.path.join(args.prompt_dir, 'binary_disorder_binary_nodis.txt'),
        'multi_disorder': os.path.join(args.prompt_dir, 'disorder_multi_q.txt'),
        'a_disorders': os.path.join(args.prompt_dir, 'a_disorder.txt'),
    }

    results = evaluate_model_output(args.full_dir, args.test_file, prompt_paths)
    for key, value in results.items():
        print(f'{key}: {value:.2f}%')