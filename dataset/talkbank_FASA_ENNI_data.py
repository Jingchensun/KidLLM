# import os
# import json
# import ast
# import random
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# # ========== 工具函数 ==========

# def load_prompt_lines(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return [line.strip() for line in f.readlines()]

# def shuffle_qa_pairs(qa_list):
#     paired_list = [(qa_list[i], qa_list[i + 1]) for i in range(0, len(qa_list), 2)]
#     random.shuffle(paired_list)
#     return [item for pair in paired_list for item in pair]

# # ========== 加载 Prompt 模板 ==========

# prompt_dir = './prompts/'

# prompts = {
#     'what_you_hear': load_prompt_lines(prompt_dir + 'whatyouhear.txt'),
#     'a_what': load_prompt_lines(prompt_dir + 'a_whatyouhear.txt'),
#     'transcribe_eng': load_prompt_lines(prompt_dir + 'transctibe_english.txt'),
#     'dialect': load_prompt_lines(prompt_dir + 'dialect.txt'),
#     'disorder_yes': load_prompt_lines(prompt_dir + 'a_disorder.txt'),
#     'disorder_no': load_prompt_lines(prompt_dir + 'a_nodisorder.txt'),
#     'binary_disorder_q': load_prompt_lines(prompt_dir + 'disorder_binary_q.txt'),
#     'binary_disorder_a_yes': load_prompt_lines(prompt_dir + 'binary_disorder_binary_isdis.txt'),
#     'binary_disorder_a_no': load_prompt_lines(prompt_dir + 'binary_disorder_binary_nodis.txt'),
#     'age_q': load_prompt_lines(prompt_dir + 'age_q.txt'),
#     'age_a': load_prompt_lines(prompt_dir + 'age_a.txt'),
#     'gender_q': load_prompt_lines(prompt_dir + 'gender_q.txt'),
#     'gender_boy': load_prompt_lines(prompt_dir + 'gender_boy.txt'),
#     'gender_girl': load_prompt_lines(prompt_dir + 'gender_girl.txt'),
# }

# # ========== 主函数 ==========

# def create_conversations(data_dir, metadata_csv, add_dialect=True):
#     conversations_by_speaker = {}
    
#     metadata_df = pd.read_csv(metadata_csv)
#     metadata_df['id_index'] = metadata_df['audio_file'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
#     print(f"\n📁 开始处理数据目录：{data_dir}")
#     for root, _, files in tqdm(list(os.walk(data_dir)), desc="📊 数据处理进度"):
#         mp3_files = [f for f in files if f.endswith('.mp3')]
#         for audio_file in mp3_files:
#             folder_id = os.path.basename(root)
#             audio_path = os.path.join(root, audio_file)
#             transcript_path = audio_path.replace('.mp3', '.txt')

#             try:
#                 meta_row = metadata_df[metadata_df['id_index'] == folder_id].iloc[0]
#                 metadata = ast.literal_eval(meta_row['metadata'])[0]
#             except Exception as e:
#                 print(f"❌ Metadata 错误：{folder_id} — {e}")
#                 continue

#             speaker_id = metadata.get('speaker', folder_id)
#             audio_rel_path = audio_path.split('/data/jingchen/')[1]
#             conv = []

#             # 初始化统计标志
#             has_transcribe = has_age = has_gender = False
#             is_td = is_sli = False

#             # Transcription 对话
#             if os.path.exists(transcript_path):
#                 sentence = open(transcript_path, 'r').readline().strip()
#                 conv.extend([
#                     {'from': 'human', 'value': random.choice(prompts['what_you_hear'])},
#                     {'from': 'gpt', 'value': random.choice(prompts['a_what']).format('child')},
#                     {'from': 'human', 'value': random.choice(prompts['transcribe_eng'])},
#                     {'from': 'gpt', 'value': f'This is the english transcription, {sentence}'}
#                 ])
#                 has_transcribe = True

#             if add_dialect:
#                 dialect_label = "American English"
#                 conv.extend([
#                     {'from': 'human', 'value': random.choice(prompts['dialect'])},
#                     {'from': 'gpt', 'value': f'The speaker sounds to be {dialect_label}'}
#                 ])

#             # 年龄
#             if metadata.get('age_in_days') is not None:
#                 age_years = int(round(metadata['age_in_days'] / 365))
#                 conv.extend([
#                     {'from': 'human', 'value': random.choice(prompts['age_q'])},
#                     {'from': 'gpt', 'value': random.choice(prompts['age_a']).format(age_years)}
#                 ])
#                 has_age = True

#             # 性别
#             if metadata.get('sex') and metadata['sex'] != '-':
#                 conv.append({'from': 'human', 'value': random.choice(prompts['gender_q'])})
#                 gender_reply = random.choice(prompts['gender_girl'] if metadata['sex'] == 'female' else prompts['gender_boy'])
#                 conv.append({'from': 'gpt', 'value': gender_reply})
#                 has_gender = True

#             # 语言障碍信息
#             if meta_row['group_type'] == 'TD':
#                 conv.extend([
#                     {'from': 'human', 'value': random.choice(prompts['binary_disorder_q'])},
#                     {'from': 'gpt', 'value': random.choice(prompts['binary_disorder_a_no'])}
#                 ])
#                 is_td = True
#             elif meta_row['group_type'] == 'SLI':
#                 conv.extend([
#                     {'from': 'human', 'value': random.choice(prompts['binary_disorder_q'])},
#                     {'from': 'gpt', 'value': random.choice(prompts['binary_disorder_a_yes'])}
#                 ])
#                 is_sli = True

#             # conv = shuffle_qa_pairs(conv)

#             if speaker_id not in conversations_by_speaker:
#                 conversations_by_speaker[speaker_id] = []

#             conversations_by_speaker[speaker_id].append({
#                 'audio_name': audio_rel_path,
#                 'conversation': conv,
#                 'stat': {
#                     'transcribe': has_transcribe,
#                     'age': has_age,
#                     'gender': has_gender,
#                     'td': is_td,
#                     'sli': is_sli
#                 }
#             })

#     return conversations_by_speaker

# def compute_statistics(conversations):
#     stats = {
#         'total': len(conversations),
#         'transcribe': 0,
#         'age': 0,
#         'gender': 0,
#         'td': 0,
#         'sli': 0
#     }
#     for conv in conversations:
#         stat = conv.get('stat', {})
#         if stat.get('transcribe'): stats['transcribe'] += 1
#         if stat.get('age'): stats['age'] += 1
#         if stat.get('gender'): stats['gender'] += 1
#         if stat.get('td'): stats['td'] += 1
#         if stat.get('sli'): stats['sli'] += 1
#     return stats

# # ========== 主执行入口 ==========

# if __name__ == '__main__':
#     data_dir = '/data/jingchen/KIDS/talkbank_dataset/v1.3/official_v1.3/usable/FASA_ENNI/out'
#     metadata_csv = '/data/jingchen/KIDS/talkbank_dataset/v1.3/official_v1.3/talkbank_childes.csv'

#     conversations_by_speaker = create_conversations(data_dir, metadata_csv, add_dialect=False)

#     speaker_ids = list(conversations_by_speaker.keys())
#     train_ids, temp_ids = train_test_split(speaker_ids, test_size=0.3, random_state=42)
#     val_ids, test_ids = train_test_split(temp_ids, test_size=1/3, random_state=42)

#     subsets = {
#         'train': sum([conversations_by_speaker[s] for s in train_ids], []),
#         'val': sum([conversations_by_speaker[s] for s in val_ids], []),
#         'test': sum([conversations_by_speaker[s] for s in test_ids], [])
#     }
#     subset_speakers = {
#     'train': train_ids,
#     'val': val_ids,
#     'test': test_ids
#             }
#     os.makedirs('./output', exist_ok=True)

#     statistics = {}

#     for subset_name, subset_data in subsets.items():
#         # ✅ 提前统计
#         statistics[subset_name] = compute_statistics(subset_data)

#         # 🧹 删除 stat 字段再写入 JSON
#         for item in subset_data:
#             item.pop('stat', None)

#         with open(f'./output/talkban_v1_3_enni_post_{subset_name}.json', 'w', encoding='utf-8') as f_out:
#             json.dump(subset_data, f_out, ensure_ascii=False, indent=4)


#     # 保存统计信息
#     with open('./output/statistic.json', 'w', encoding='utf-8') as f_stat:
#         json.dump(statistics, f_stat, indent=4)

#     print("\n✅ 数据处理完成！各子集样本数如下：")
#     for k, v in statistics.items():
#         print(f"{k.capitalize()}: {v} (Speakers: {len(subset_speakers[k])})")

import os
import json
import ast
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========== 工具函数 ==========

def load_prompt_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def shuffle_qa_pairs(qa_list):
    paired_list = [(qa_list[i], qa_list[i + 1]) for i in range(0, len(qa_list), 2)]
    random.shuffle(paired_list)
    return [item for pair in paired_list for item in pair]

# ========== 加载 Prompt 模板 ==========

prompt_dir = './prompts/'

prompts = {
    'what_you_hear': load_prompt_lines(prompt_dir + 'whatyouhear.txt'),
    'a_what': load_prompt_lines(prompt_dir + 'a_whatyouhear.txt'),
    'transcribe_eng': load_prompt_lines(prompt_dir + 'transctibe_english.txt'),
    'dialect': load_prompt_lines(prompt_dir + 'dialect.txt'),
    'disorder_yes': load_prompt_lines(prompt_dir + 'a_disorder.txt'),
    'disorder_no': load_prompt_lines(prompt_dir + 'a_nodisorder.txt'),
    'binary_disorder_q': load_prompt_lines(prompt_dir + 'disorder_binary_q.txt'),
    'binary_disorder_a_yes': load_prompt_lines(prompt_dir + 'binary_disorder_binary_isdis.txt'),
    'binary_disorder_a_no': load_prompt_lines(prompt_dir + 'binary_disorder_binary_nodis.txt'),
    'age_q': load_prompt_lines(prompt_dir + 'age_q.txt'),
    'age_a': load_prompt_lines(prompt_dir + 'age_a.txt'),
    'gender_q': load_prompt_lines(prompt_dir + 'gender_q.txt'),
    'gender_boy': load_prompt_lines(prompt_dir + 'gender_boy.txt'),
    'gender_girl': load_prompt_lines(prompt_dir + 'gender_girl.txt'),
}

# ========== 主函数 ==========

def create_conversations(data_dir, metadata_csv, add_dialect=True):
    all_conversations = []  # ✅改成直接收集所有样本

    metadata_df = pd.read_csv(metadata_csv)
    metadata_df['id_index'] = metadata_df['audio_file'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    print(f"\n📁 开始处理数据目录：{data_dir}")
    for root, _, files in tqdm(list(os.walk(data_dir)), desc="📊 数据处理进度"):
        mp3_files = [f for f in files if f.endswith('.mp3')]
        for audio_file in mp3_files:
            folder_id = os.path.basename(root)
            audio_path = os.path.join(root, audio_file)
            transcript_path = audio_path.replace('.mp3', '.txt')

            try:
                meta_row = metadata_df[metadata_df['id_index'] == folder_id].iloc[0]
                metadata = ast.literal_eval(meta_row['metadata'])[0]
            except Exception as e:
                print(f"❌ Metadata 错误：{folder_id} — {e}")
                continue

            speaker_id = metadata.get('speaker', folder_id)
            audio_rel_path = audio_path.split('/data/jingchen/')[1]
            conv = []

            has_transcribe = has_age = has_gender = False
            is_td = is_sli = False

            if os.path.exists(transcript_path):
                sentence = open(transcript_path, 'r').readline().strip()
                conv.extend([
                    {'from': 'human', 'value': random.choice(prompts['what_you_hear'])},
                    {'from': 'gpt', 'value': random.choice(prompts['a_what']).format('child')},
                    {'from': 'human', 'value': random.choice(prompts['transcribe_eng'])},
                    {'from': 'gpt', 'value': f'This is the english transcription, {sentence}'}
                ])
                has_transcribe = True

            if add_dialect:
                dialect_label = "American English"
                conv.extend([
                    {'from': 'human', 'value': random.choice(prompts['dialect'])},
                    {'from': 'gpt', 'value': f'The speaker sounds to be {dialect_label}'}
                ])

            if metadata.get('age_in_days') is not None:
                age_years = int(round(metadata['age_in_days'] / 365))
                conv.extend([
                    {'from': 'human', 'value': random.choice(prompts['age_q'])},
                    {'from': 'gpt', 'value': random.choice(prompts['age_a']).format(age_years)}
                ])
                has_age = True

            if metadata.get('sex') and metadata['sex'] != '-':
                conv.append({'from': 'human', 'value': random.choice(prompts['gender_q'])})
                gender_reply = random.choice(prompts['gender_girl'] if metadata['sex'] == 'female' else prompts['gender_boy'])
                conv.append({'from': 'gpt', 'value': gender_reply})
                has_gender = True

            if meta_row['group_type'] == 'TD':
                conv.extend([
                    {'from': 'human', 'value': random.choice(prompts['binary_disorder_q'])},
                    {'from': 'gpt', 'value': random.choice(prompts['binary_disorder_a_no'])}
                ])
                is_td = True
            elif meta_row['group_type'] == 'SLI':
                conv.extend([
                    {'from': 'human', 'value': random.choice(prompts['binary_disorder_q'])},
                    {'from': 'gpt', 'value': random.choice(prompts['binary_disorder_a_yes'])}
                ])
                is_sli = True

            # conv = shuffle_qa_pairs(conv)

            all_conversations.append({
                'audio_name': audio_rel_path,
                'conversation': conv,
                'speaker_id': speaker_id,  # ✅保存 speaker_id 方便后面统计
                'stat': {
                    'transcribe': has_transcribe,
                    'age': has_age,
                    'gender': has_gender,
                    'td': is_td,
                    'sli': is_sli
                }
            })

    return all_conversations

def compute_statistics(conversations):
    stats = {
        'total': len(conversations),
        'transcribe': 0,
        'age': 0,
        'gender': 0,
        'td': 0,
        'sli': 0
    }
    for conv in conversations:
        stat = conv.get('stat', {})
        if stat.get('transcribe'): stats['transcribe'] += 1
        if stat.get('age'): stats['age'] += 1
        if stat.get('gender'): stats['gender'] += 1
        if stat.get('td'): stats['td'] += 1
        if stat.get('sli'): stats['sli'] += 1
    return stats

# ========== 主执行入口 ==========

if __name__ == '__main__':
    data_dir = '/data/jingchen/KIDS/talkbank_dataset/v1.3/official_v1.3/usable/FASA_ENNI/out'
    metadata_csv = '/data/jingchen/KIDS/talkbank_dataset/v1.3/official_v1.3/talkbank_childes.csv'

    all_conversations = create_conversations(data_dir, metadata_csv, add_dialect=False)

    # ✅改：直接对所有样本按比例划分
    train_data, temp_data = train_test_split(all_conversations, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

    subsets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    os.makedirs('./output', exist_ok=True)

    statistics = {}

    for subset_name, subset_data in subsets.items():
        # ✅ 提前统计
        statistics[subset_name] = compute_statistics(subset_data)

        # 🧹 删除 stat 和 speaker_id 字段后保存
        for item in subset_data:
            item.pop('stat', None)
            item.pop('speaker_id', None)

        with open(f'./output/talkbank_v1_3_enni_post_{subset_name}.json', 'w', encoding='utf-8') as f_out:
            json.dump(subset_data, f_out, ensure_ascii=False, indent=4)

    # 保存统计信息
    with open('./output/statistic.json', 'w', encoding='utf-8') as f_stat:
        json.dump(statistics, f_stat, indent=4)

    print("\n✅ 数据处理完成！各子集样本数如下：")
    for k, v in statistics.items():
        print(f"{k.capitalize()}: {v['total']} samples")