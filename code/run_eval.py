import os
import torch
import json
import sys
from tqdm import tqdm
import argparse
from model.openllama import OpenLLAMAPEFTModel

# Set UTF-8 encoding for stdout and stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def parse_text(text):
    """格式化模型输出文本，处理特殊字符"""
    lines = [line for line in text.split("\n") if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            lines[i] = f'<pre><code class="language-{items[-1]}">' if count % 2 == 1 else f'<br></code></pre>'
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", "\`").replace("<", "&lt;").replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;").replace("*", "&ast;").replace("_", "&lowbar;")
                line = line.replace("-", "&#45;").replace(".", "&#46;").replace("!", "&#33;")
                line = line.replace("(", "&#40;").replace(")", "&#41;").replace("$", "&#36;")
            lines[i] = "<br>" + line
    return "".join(lines)


def predict(input_text, audio_path, image_path, video_path, thermal_path, model, max_length, top_p, temperature, history, modality_cache):
    if not any([image_path, audio_path, video_path, thermal_path]):
        return [(input_text, "There is no input data provided! Please upload your data and start the conversation.")]
    
    prompt_text = ""
    for idx, (q, a) in enumerate(history):
        prefix = "" if idx == 0 else " Human: "
        prompt_text += f"{prefix}{q}\n### Assistant: {a}\n###"
    prompt_text += f" Human: {input_text}" if history else input_text

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    history.append((input_text, response))
    return parse_text(input_text), parse_text(response), history


def init_model(vicuna_hf_repo, delta_ckpt_path, lora_r, lora_alpha, lora_dropout, stage, whisper_pretrained='small'):
    args = {
        'model': 'openllama_peft',
        'vicuna_hf_repo': vicuna_hf_repo,
        'delta_ckpt_path': delta_ckpt_path,
        'stage': stage,
        'max_tgt_len': 1200,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'whisper_pretrained': whisper_pretrained,
    }
    model = OpenLLAMAPEFTModel(**args)
    
    # 加载优化后的checkpoint（只包含LoRA + 可训练参数）
    print(f'[INFO] Loading optimized checkpoint from {delta_ckpt_path}')
    delta_ckpt = torch.load(delta_ckpt_path, map_location='cpu')
    
    # 分别加载LoRA参数和其他可训练参数
    lora_state_dict = {}
    other_state_dict = {}
    
    for k, v in delta_ckpt.items():
        if k.startswith('llama_model.'):
            # LoRA参数，需要去掉前缀
            lora_key = k[len('llama_model.'):]
            lora_state_dict[lora_key] = v
        else:
            # 其他可训练参数（音频处理模块）
            # 处理不同版本的参数映射
            if k == 'llama_proj.weight':
                # 旧版本单一投影层 → 新版本主投影路径
                other_state_dict['audio_to_llama_main.weight'] = v
                print(f'[INFO] Mapped old parameter: {k} → audio_to_llama_main.weight')
            elif k == 'llama_proj.bias':
                other_state_dict['audio_to_llama_main.bias'] = v
                print(f'[INFO] Mapped old parameter: {k} → audio_to_llama_main.bias')
            elif k.startswith('audio_down.') or k.startswith('audio_ln1.') or k.startswith('gated_blocks.'):
                # 旧复杂版本的参数，跳过不兼容的组件
                print(f'[INFO] Skipping incompatible old parameter: {k}')
                continue
            elif k == 'audio_to_llama.weight':
                # 中间版本 → 新版本主投影路径
                other_state_dict['audio_to_llama_main.weight'] = v
                print(f'[INFO] Mapped parameter: {k} → audio_to_llama_main.weight')
            elif k == 'audio_to_llama.bias':
                other_state_dict['audio_to_llama_main.bias'] = v
                print(f'[INFO] Mapped parameter: {k} → audio_to_llama_main.bias')
            else:
                # 新版本参数，直接使用
                other_state_dict[k] = v
    
    # 加载LoRA参数
    if lora_state_dict:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.llama_model, lora_state_dict)
        print(f'[INFO] Loaded {len(lora_state_dict)} LoRA parameters')
    
    # 加载其他参数（使用strict=False以兼容新增模块）
    if other_state_dict:
        missing_keys, unexpected_keys = model.load_state_dict(other_state_dict, strict=False)
        print(f'[INFO] Loaded {len(other_state_dict)} other trainable parameters')
        
        # 过滤掉已知的冻结参数，只显示真正缺失的关键参数
        critical_missing = []
        for key in missing_keys:
            # 跳过Whisper编码器参数（被冻结）
            if key.startswith('audio_encoder.'):
                continue
            # 跳过LLaMA基础模型参数（被LoRA冻结）
            if key.startswith('llama_model.base_model.model.model.') and not 'lora' in key:
                continue
            # 只关注真正的可训练参数
            critical_missing.append(key)
        
        if critical_missing:
            print(f'[WARNING] Critical missing trainable parameters: {critical_missing}')
        else:
            print('[INFO] All expected trainable parameters loaded successfully')
            
        if unexpected_keys:
            print(f'[WARNING] Unexpected keys (ignored): {unexpected_keys}')
    
    # 检查架构配置兼容性
    if '_architecture_config' in delta_ckpt:
        arch_config = delta_ckpt['_architecture_config']
        print(f'[INFO] Loaded architecture config: {arch_config}')
    else:
        print('[INFO] No architecture config found (older checkpoint)')
    
    # 打印模型参数统计
    model.print_trainable_parameters()
    
    print('[INFO] Optimized checkpoint loaded successfully')
    return model.eval().half().cuda()


def run_inference(model, test_file, save_file_path, base_audio_dir, max_length, top_p, temperature):
    with open(test_file, 'r') as f:
        test_set = json.load(f)
        # test_set = test_set[:max(1, len(test_set) // 100)]  # 取前 1%，至少保留 1 条数据

    count = 0
    all_ans = []
    for conv in tqdm(test_set):
        count += 1
        audio_path = os.path.join(base_audio_dir, conv['audio_name'])
        conversation = conv['conversation']
        history = []

        for i in range(0, len(conversation), 2):
            assert conversation[i]['from'] == 'human'
            assert conversation[i+1]['from'] == 'gpt'

            question = conversation[i]['value']
            inputf, responsef, history = predict(
                input_text=question,
                audio_path=audio_path,
                image_path='',
                video_path='',
                thermal_path='',
                model=model,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                history=history,
                modality_cache=[],
            )

            gt = conversation[i+1]['value']
            all_ans.append([question, responsef, gt])

        if count % 10 == 0:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            with open(save_file_path, 'a', encoding='utf-8') as w:
                w.write('\n'.join('|'.join(i) for i in all_ans))
                w.write('\n')
            all_ans = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--vicuna_hf_repo', type=str, required=True, help='Hugging Face repository for vicuna model')
    parser.add_argument('--delta_ckpt_base_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--save_file_path', type=str, required=True)
    parser.add_argument('--base_audio_dir', type=str, required=True)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--whisper_pretrained', type=str, default='small', 
                       help='whisper model size: ours, tiny, base, small, medium, large, or HF model names')

    args = parser.parse_args()
    delta_ckpt_path = os.path.join(args.delta_ckpt_base_path, f'pytorch_model_{args.epoch}.pt')
    
    print(f'[INFO] Loading model from epoch {args.epoch}...')
    print(f'[INFO] Using Whisper model: {args.whisper_pretrained}')
    model = init_model(
        vicuna_hf_repo=args.vicuna_hf_repo,
        delta_ckpt_path=delta_ckpt_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        stage=args.stage,
        whisper_pretrained=args.whisper_pretrained,
    )
    print('[INFO] Model loaded successfully.')

    run_inference(
        model=model,
        test_file=args.test_file,
        save_file_path=args.save_file_path,
        base_audio_dir=args.base_audio_dir,
        max_length=args.max_length,
        top_p=args.top_p,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()
