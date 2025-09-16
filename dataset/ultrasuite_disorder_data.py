# import os
# import json
# import random
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm


# # ==== 加载 prompts ====
# disorders_multi_q = [line.strip() for line in open('./prompts/disorder_multi_q.txt')]
# a_disorders_multi = [line.strip() for line in open('./prompts/a_disorder.txt')]
# binary_disorders_q = [line.strip() for line in open('./prompts/disorder_binary_q.txt')]
# binary_disorders_a_is = [line.strip() for line in open('./prompts/binary_disorder_binary_isdis.txt')]

# # 新增：性别和年龄相关 prompts
# age_q = [line.strip() for line in open('./prompts/age_q.txt')]
# age_a = [line.strip() for line in open('./prompts/age_a.txt')]
# gender_q = [line.strip() for line in open('./prompts/gender_q.txt')]
# gender_boy = [line.strip() for line in open('./prompts/gender_boy.txt')]
# gender_girl = [line.strip() for line in open('./prompts/gender_girl.txt')]

# # ==== 加载 speaker 信息 ====
# speaker_df = pd.read_csv('/home/jingchen/kidspeak/KIDS/ultrasuite_disorder/speakers', sep='\t')
# speaker2label = dict(zip(speaker_df['speaker_id'], speaker_df['ssd_subtype']))
# speaker2age = dict(zip(speaker_df['speaker_id'], speaker_df['age']))
# speaker2sex = dict(zip(speaker_df['speaker_id'], speaker_df['sex']))

# print(f"📋 Loaded {len(speaker2label)} speakers with disorder, age, and sex info.")

# # ==== 主函数 ====
# def collect_data_by_speaker(data_dir):
#     conversations_by_speaker = {}

#     for root, dirs, files in os.walk(data_dir):
#         wav_files = list(filter(lambda x: x.endswith('.wav'), files))
#         if not wav_files:
#             continue

#         stage = root.split('/')[-1]
#         if 'BL' not in stage:
#             continue

#         speaker_id = root.split('/')[-2]
#         if speaker_id not in speaker2label:
#             continue

#         for audio_file in wav_files:
#             audio_path = os.path.join(root, audio_file)
#             txt_path = audio_path.replace('.wav', '.txt')
#             if not os.path.exists(txt_path):
#                 continue

#             try:
#                 sentence = open(txt_path).readline().strip()
#             except:
#                 continue

#             # 构造对话 prompts
#             conv = []
#             # 语言障碍问题（二分类 + 多分类）
#             conv.append({'from': 'human', 'value': random.choice(binary_disorders_q)})
#             conv.append({'from': 'gpt', 'value': random.choice(binary_disorders_a_is)})
#             conv.append({'from': 'human', 'value': random.choice(disorders_multi_q)})
#             conv.append({'from': 'gpt', 'value': random.choice(a_disorders_multi).format(speaker2label[speaker_id])})

#             # 年龄问答
#             if speaker_id in speaker2age:
#                 age = round(float(speaker2age[speaker_id]))
#                 conv.append({'from': 'human', 'value': random.choice(age_q)})
#                 conv.append({'from': 'gpt', 'value': random.choice(age_a).format(age)})

#             # 性别问答
#             if speaker_id in speaker2sex:
#                 conv.append({'from': 'human', 'value': random.choice(gender_q)})
#                 if speaker2sex[speaker_id].lower() == 'female':
#                     conv.append({'from': 'gpt', 'value': random.choice(gender_girl)})
#                 else:
#                     conv.append({'from': 'gpt', 'value': random.choice(gender_boy)})

#             # 洗牌 QA 顺序
#         #     conv = shuffle_qa_pairs(conv)

#             basepath = audio_path.split('/home/jingchen/kidspeak/')[1]
#             item = {
#                 "audio_name": basepath,
#                 "conversation": conv
#             }

#             if speaker_id not in conversations_by_speaker:
#                 conversations_by_speaker[speaker_id] = []

#             conversations_by_speaker[speaker_id].append(item)

#     return conversations_by_speaker


# # ========== 主执行入口 ==========
# if __name__ == '__main__':
#     data_dir = '/home/jingchen/kidspeak/KIDS/ultrasuite_disorder/core-upx/core'
#     conversations_by_speaker = collect_data_by_speaker(data_dir)

#     speaker_ids = list(conversations_by_speaker.keys())
#     train_ids, temp_ids = train_test_split(speaker_ids, test_size=0.3, random_state=42)
#     val_ids, test_ids = train_test_split(temp_ids, test_size=1/3, random_state=42)

#     subsets = {
#         'train': sum([conversations_by_speaker[s] for s in train_ids], []),
#         'val': sum([conversations_by_speaker[s] for s in val_ids], []),
#         'test': sum([conversations_by_speaker[s] for s in test_ids], [])
#     }

#     # 输出 JSON 文件
#     os.makedirs('./output', exist_ok=True)
#     for split in ['train', 'val', 'test']:
#         with open(f'./output/ultrasuite_disorder_{split}.json', 'w', encoding='utf-8') as f:
#             json.dump(subsets[split], f, ensure_ascii=False, indent=4)

#     # 输出统计
#     print("\n✅ 数据划分完成！样本数量如下：")
#     for k, v in subsets.items():
#         print(f"{k.capitalize()}: {len(v)}")

#     # 输出统计
#     print("\n✅ 数据划分完成！样本数量如下：")
#     for split_name in ['train', 'val', 'test']:
#         split_data = subsets[split_name]
#         # 统计样本数量
#         num_samples = len(split_data)
#         # 统计唯一 speaker 数量（从路径中解析 speaker_id）
#         speaker_ids_in_split = set([
#                 os.path.normpath(item['audio_name']).split(os.sep)[4]
#                 for item in split_data
#                 ])
#         num_speakers = len(speaker_ids_in_split)

#         print(f"{split_name.capitalize()}:")
#         print(f"  🔹 样本数: {num_samples}")
#         print(f"  👤 Speaker 数: {num_speakers}")

import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==== 加载 prompts ====
disorders_multi_q = [line.strip() for line in open('./prompts/disorder_multi_q.txt')]
a_disorders_multi = [line.strip() for line in open('./prompts/a_disorder.txt')]
binary_disorders_q = [line.strip() for line in open('./prompts/disorder_binary_q.txt')]
binary_disorders_a_is = [line.strip() for line in open('./prompts/binary_disorder_binary_isdis.txt')]

age_q = [line.strip() for line in open('./prompts/age_q.txt')]
age_a = [line.strip() for line in open('./prompts/age_a.txt')]
gender_q = [line.strip() for line in open('./prompts/gender_q.txt')]
gender_boy = [line.strip() for line in open('./prompts/gender_boy.txt')]
gender_girl = [line.strip() for line in open('./prompts/gender_girl.txt')]

# ==== 加载 speaker 信息 ====
speaker_df = pd.read_csv('/home/jingchen/kidspeak/KIDS/ultrasuite_disorder/speakers', sep='\t')
speaker2label = dict(zip(speaker_df['speaker_id'], speaker_df['ssd_subtype']))
speaker2age = dict(zip(speaker_df['speaker_id'], speaker_df['age']))
speaker2sex = dict(zip(speaker_df['speaker_id'], speaker_df['sex']))

print(f"📋 Loaded {len(speaker2label)} speakers with disorder, age, and sex info.")

# ==== 主函数 ====
def collect_all_conversations(data_dir):
    all_conversations = []

    for root, dirs, files in os.walk(data_dir):
        wav_files = list(filter(lambda x: x.endswith('.wav'), files))
        if not wav_files:
            continue

        stage = root.split('/')[-1]
        if 'BL' not in stage:
            continue

        speaker_id = root.split('/')[-2]
        if speaker_id not in speaker2label:
            continue

        for audio_file in wav_files:
            audio_path = os.path.join(root, audio_file)
            txt_path = audio_path.replace('.wav', '.txt')
            if not os.path.exists(txt_path):
                continue

            try:
                sentence = open(txt_path).readline().strip()
            except:
                continue

            conv = []
            conv.append({'from': 'human', 'value': random.choice(binary_disorders_q)})
            conv.append({'from': 'gpt', 'value': random.choice(binary_disorders_a_is)})
            conv.append({'from': 'human', 'value': random.choice(disorders_multi_q)})
            conv.append({'from': 'gpt', 'value': random.choice(a_disorders_multi).format(speaker2label[speaker_id])})

            if speaker_id in speaker2age:
                age = round(float(speaker2age[speaker_id]))
                conv.append({'from': 'human', 'value': random.choice(age_q)})
                conv.append({'from': 'gpt', 'value': random.choice(age_a).format(age)})

            if speaker_id in speaker2sex:
                conv.append({'from': 'human', 'value': random.choice(gender_q)})
                if speaker2sex[speaker_id].lower() == 'female':
                    conv.append({'from': 'gpt', 'value': random.choice(gender_girl)})
                else:
                    conv.append({'from': 'gpt', 'value': random.choice(gender_boy)})

            basepath = audio_path.split('/home/jingchen/kidspeak/')[1]
            item = {
                "audio_name": basepath,
                "conversation": conv
            }

            all_conversations.append(item)

    return all_conversations

# ========== 主执行入口 ==========
if __name__ == '__main__':
    data_dir = '/home/jingchen/kidspeak/KIDS/ultrasuite_disorder/core-upx/core'
    all_samples = collect_all_conversations(data_dir)

    # ✅改：直接对所有样本做划分
    train_samples, temp_samples = train_test_split(all_samples, test_size=0.3, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=1/3, random_state=42)

    subsets = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

    # 输出 JSON 文件
    os.makedirs('./output', exist_ok=True)
    for split in ['train', 'val', 'test']:
        with open(f'./output/ultrasuite_disorder_{split}.json', 'w', encoding='utf-8') as f:
            json.dump(subsets[split], f, ensure_ascii=False, indent=4)

    # 打印统计信息
    print("\n✅ 数据划分完成！样本和 speaker 统计如下：")
    for split_name in ['train', 'val', 'test']:
        split_data = subsets[split_name]
        num_samples = len(split_data)

        speaker_ids_in_split = set([
            os.path.normpath(item['audio_name']).split(os.sep)[4]
            for item in split_data
        ])
        num_speakers = len(speaker_ids_in_split)

        print(f"{split_name.capitalize()}:")
        print(f"  🔹 样本数: {num_samples}")
        print(f"  👤 Speaker数: {num_speakers}")
