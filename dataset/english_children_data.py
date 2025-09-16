# import os
# import json
# import random
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# # 加载 prompt 模板
# def load_prompt(path):
#     return [line.strip() for line in open(path).readlines()]

# whatyouhear = load_prompt('prompts/whatyouhear.txt')
# a_what = load_prompt('prompts/a_whatyouhear.txt')
# transeng = load_prompt('prompts/transctibe_english.txt')
# dialect = load_prompt('prompts/dialect.txt')
# gender_q = load_prompt('prompts/gender_q.txt')
# gender_boy = load_prompt('prompts/gender_boy.txt')
# gender_girl = load_prompt('prompts/gender_girl.txt')

# # 主函数
# def collect_conversations(DIR, only_studio_sentences=False, add_dialect=True):
#     speaker_data = {}

#     for root, dirs, files in os.walk(DIR):
#         # 如果限定仅处理 studio_mic/sentences 路径
#         if only_studio_sentences and ('studio_mic/sentences' not in root.replace('\\', '/')):
#             continue

#         for audio in files:
#             if not audio.endswith('.wav'):
#                 continue

#             a = os.path.join(root, audio)

#             # 从路径中提取 speaker_id（即 speaker 文件夹名）
#             parts = a.split('/')
#             if 'english_words_sentences' in a:
#                 # 假设结构是 /.../english_words_sentences/01_M_native/studio_mic/sentences/xxx.wav
#                 try:
#                     speaker_index = parts.index('english_words_sentences') + 1
#                     speaker_id = parts[speaker_index]
#                 except:
#                     speaker_id = os.path.basename(os.path.dirname(a))
#             else:
#                 speaker_id = os.path.basename(os.path.dirname(a))

#             # 判断性别
#             if '_M_' in a:
#                 person = 'boy'
#                 gender_answer = random.choice(gender_boy)
#             elif '_F_' in a:
#                 person = 'girl'
#                 gender_answer = random.choice(gender_girl)
#             else:
#                 person = 'child'
#                 gender_answer = random.choice(gender_boy + gender_girl)

#             # 判断方言
#             if '_native' in a:
#                 dr = 'from UK'
#             elif '_nonNative' in a:
#                 dr = 'not from UK'
#             else:
#                 dr = 'unknown'

#             sentence2 = os.path.basename(a).split('.')[0].strip()
#             sentence2 = ' '.join(sentence2.split('_'))

#             prompts = []
#             prompts.append({'from': 'human', 'value': random.choice(whatyouhear)})
#             prompts.append({'from': 'gpt', 'value': random.choice(a_what).format(person)})
#             prompts.append({'from': 'human', 'value': random.choice(transeng)})
#             prompts.append({'from': 'gpt', 'value': f'This is the english transcription, {sentence2}'})

#             if add_dialect:
#                 prompts.append({'from': 'human', 'value': random.choice(dialect)})
#                 prompts.append({'from': 'gpt', 'value': f'The speaker sounds to be {dr}'})

#             prompts.append({'from': 'human', 'value': random.choice(gender_q)})
#             prompts.append({'from': 'gpt', 'value': gender_answer})

#             audio_rel_path = a.split('/home/jingchen/kidspeak/')[1]

#             conv_obj = {
#                 "audio_name": audio_rel_path,
#                 "conversation": prompts
#             }

#             if speaker_id not in speaker_data:
#                 speaker_data[speaker_id] = []
#             speaker_data[speaker_id].append(conv_obj)

#     return speaker_data


# # 收集所有数据
# data_paths = [
#     ('/home/jingchen/kidspeak/KIDS/english_children/english_free_speech/files_cut_by_sentences', False),
#     ('/home/jingchen/kidspeak/KIDS/english_children/english_words_sentences', True)  # 只保留 studio_mic/sentences
# ]

# all_conversations = {}
# for path, only_studio in data_paths:
#     partial = collect_conversations(path, only_studio_sentences=only_studio)
#     for speaker_id, convs in partial.items():
#         if speaker_id not in all_conversations:
#             all_conversations[speaker_id] = []
#         all_conversations[speaker_id].extend(convs)

# # Train/Val/Test 划分
# speaker_ids = list(all_conversations.keys())
# train_ids, temp_ids = train_test_split(speaker_ids, test_size=0.3, random_state=42)
# val_ids, test_ids = train_test_split(temp_ids, test_size=1/3, random_state=42)

# subsets = {
#     'train': sum([all_conversations[s] for s in train_ids], []),
#     'val': sum([all_conversations[s] for s in val_ids], []),
#     'test': sum([all_conversations[s] for s in test_ids], [])
# }

# # 输出
# os.makedirs('./output', exist_ok=True)

# for subset_name, subset_data in subsets.items():
#     with open(f'./output/english_children_{subset_name}.json', 'w', encoding='utf-8') as f_out:
#         json.dump(subset_data, f_out, ensure_ascii=False, indent=4)

# # 打印统计信息
# print("\n✅ 数据处理完成！各子集样本与Speaker数量如下：")
# for name, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
#     print(f"{name.capitalize()}: Samples = {len(subsets[name])}, Speakers = {len(ids)}")


import os
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 加载 prompt 模板
def load_prompt(path):
    return [line.strip() for line in open(path).readlines()]

whatyouhear = load_prompt('prompts/whatyouhear.txt')
a_what = load_prompt('prompts/a_whatyouhear.txt')
transeng = load_prompt('prompts/transctibe_english.txt')
dialect = load_prompt('prompts/dialect.txt')
gender_q = load_prompt('prompts/gender_q.txt')
gender_boy = load_prompt('prompts/gender_boy.txt')
gender_girl = load_prompt('prompts/gender_girl.txt')

# 主函数
def collect_conversations(DIR, only_studio_sentences=False, add_dialect=True):
    all_samples = []  # ✅改：不按 speaker_id 收集，直接收集所有样本

    for root, dirs, files in os.walk(DIR):
        if only_studio_sentences and ('studio_mic/sentences' not in root.replace('\\', '/')):
            continue

        for audio in files:
            if not audio.endswith('.wav'):
                continue

            a = os.path.join(root, audio)

            parts = a.split('/')
            if 'english_words_sentences' in a:
                try:
                    speaker_index = parts.index('english_words_sentences') + 1
                    speaker_id = parts[speaker_index]
                except:
                    speaker_id = os.path.basename(os.path.dirname(a))
            else:
                speaker_id = os.path.basename(os.path.dirname(a))

            if '_M_' in a:
                person = 'boy'
                gender_answer = random.choice(gender_boy)
            elif '_F_' in a:
                person = 'girl'
                gender_answer = random.choice(gender_girl)
            else:
                person = 'child'
                gender_answer = random.choice(gender_boy + gender_girl)

            if '_native' in a:
                dr = 'from UK'
            elif '_nonNative' in a:
                dr = 'not from UK'
            else:
                dr = 'unknown'

            sentence2 = os.path.basename(a).split('.')[0].strip()
            sentence2 = ' '.join(sentence2.split('_'))

            prompts = []
            prompts.append({'from': 'human', 'value': random.choice(whatyouhear)})
            prompts.append({'from': 'gpt', 'value': random.choice(a_what).format(person)})
            prompts.append({'from': 'human', 'value': random.choice(transeng)})
            prompts.append({'from': 'gpt', 'value': f'This is the english transcription, {sentence2}'})

            if add_dialect:
                prompts.append({'from': 'human', 'value': random.choice(dialect)})
                prompts.append({'from': 'gpt', 'value': f'The speaker sounds to be {dr}'})

            prompts.append({'from': 'human', 'value': random.choice(gender_q)})
            prompts.append({'from': 'gpt', 'value': gender_answer})

            audio_rel_path = a.split('/home/jingchen/kidspeak/')[1]

            conv_obj = {
                "audio_name": audio_rel_path,
                "conversation": prompts
            }

            all_samples.append(conv_obj)  # ✅改：直接添加到 all_samples

    return all_samples  # ✅改：返回所有样本，而不是以 speaker_id 分组

# 收集所有数据
data_paths = [
    ('/home/jingchen/kidspeak/KIDS/english_children/english_free_speech/files_cut_by_sentences', False),
    ('/home/jingchen/kidspeak/KIDS/english_children/english_words_sentences', True)
]

all_samples = []
for path, only_studio in data_paths:
    partial_samples = collect_conversations(path, only_studio_sentences=only_studio)
    all_samples.extend(partial_samples)

# ✅改：直接对所有样本做随机划分
train_samples, temp_samples = train_test_split(all_samples, test_size=0.3, random_state=52)
val_samples, test_samples = train_test_split(temp_samples, test_size=1/3, random_state=62)

subsets = {
    'train': train_samples,
    'val': val_samples,
    'test': test_samples
}

# 输出
os.makedirs('./output', exist_ok=True)

for subset_name, subset_data in subsets.items():
    with open(f'./output/english_children_{subset_name}.json', 'w', encoding='utf-8') as f_out:
        json.dump(subset_data, f_out, ensure_ascii=False, indent=4)

# 打印统计信息
print("\n✅ 数据处理完成！各子集样本数量如下：")
for name in ['train', 'val', 'test']:
    print(f"{name.capitalize()}: Samples = {len(subsets[name])}")
