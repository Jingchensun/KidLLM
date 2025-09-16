from .header import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from . import audio_utils
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import whisper


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0:  # the first human turn
            assert role == 'human'
            text = '</Img> ' + turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id)
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                role = 'gpt'
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]

    targets_decode = target_ids.clone().detach()
    targets_decode[targets_decode == -100] = 0

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


PROMPT_START = '### Human: <Img>'


# -----------------------------
# 门控前馈(Gated MLP)模块：支持 'glu' / 'swiglu' / 'gelu' / 'none'
# 输入: [B, T', D_in] → 输出: [B, T', D_in]（带残差+LN）
# -----------------------------
class GatedFFN(nn.Module):
    def __init__(self, dim, mult=4, variant='swiglu', dropout=0.1):
        super().__init__()
        self.variant = variant.lower()
        hid = int(mult * dim)

        if self.variant == 'glu':
            # a ⊙ σ(b)
            self.fc = nn.Linear(dim, hid * 2)
            self.proj = nn.Linear(hid, dim)
        elif self.variant == 'swiglu':
            # a ⊙ SiLU(b)
            self.fc = nn.Linear(dim, hid * 2)
            self.proj = nn.Linear(hid, dim)
        elif self.variant == 'gelu':
            self.fc = nn.Linear(dim, hid)
            self.act = nn.GELU()
            self.proj = nn.Linear(hid, dim)
        elif self.variant == 'none':
            self.proj = nn.Linear(dim, dim)  # 直通小映射
        else:
            raise ValueError(f'Unknown ff_variant: {variant}')

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        residual = x
        if self.variant == 'glu':
            uv = self.fc(x)                     # [B, T, 2H]
            u, v = uv.chunk(2, dim=-1)
            y = u * torch.sigmoid(v)           # GLU
            y = self.proj(self.dropout(y))
        elif self.variant == 'swiglu':
            uv = self.fc(x)
            u, v = uv.chunk(2, dim=-1)
            y = u * F.silu(v)                  # SwiGLU
            y = self.proj(self.dropout(y))
        elif self.variant == 'gelu':
            y = self.proj(self.dropout(self.act(self.fc(x))))
        else:  # 'none'
            y = self.proj(self.dropout(x))
        return self.norm(residual + y)


class OpenLLAMAPEFTModel(nn.Module):
    """
    音频-文本多模态模型：Whisper 编码器 + (Learnable Downsampling + Gated MLP) + LLaMA
    使用 LoRA 进行高效微调
    """

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        self.max_tgt_len = args['max_tgt_len']

        # ====== 1) Whisper 音频编码器 ======
        whisper_pretrained = args.get('whisper_pretrained', 'small')
        whisper_path = args.get('whisper_path', '')

        if whisper_pretrained == 'ours':
            if not whisper_path:
                whisper_path = '.checkpoints/whisper_ckpt/best_mhattn_contrastive_3x3_d1024_h16_55ep_phword_newtimit.ckpt'
            print(f'Loading custom Whisper model from {whisper_path}')
            self.audio_encoder = whisper.load_model('small')
            checkpoint = torch.load(whisper_path, map_location='cpu')
            checkpoint = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v
                          for k, v in checkpoint.items()}
            self.audio_encoder.load_state_dict(checkpoint, strict=False)
            self.audio_encoder = self.audio_encoder.encoder.float()
        else:
            print(f'Loading Whisper {whisper_pretrained} model from OpenAI')
            try:
                self.audio_encoder = whisper.load_model(whisper_pretrained).encoder.float()
            except Exception as e:
                print(f'Warning: Failed to load {whisper_pretrained}: {e}')
                print('Falling back to "small" model')
                whisper_pretrained = 'small'
                self.audio_encoder = whisper.load_model(whisper_pretrained).encoder.float()

        whisper_dims = {
            'tiny': 384, 'tiny.en': 384,
            'base': 512, 'base.en': 512,
            'small': 768, 'small.en': 768,
            'medium': 1024, 'medium.en': 1024,
            'large': 1280, 'large-v1': 1280, 'large-v2': 1280, 'large-v3': 1280, 'large-v3-turbo': 1280,
            'turbo': 1024
        }
        if whisper_pretrained == 'ours':
            whisper_dim = whisper_dims['small']
        else:
            whisper_dim = whisper_dims.get(whisper_pretrained, 768)

        print(f'Audio encoder initialized with model: {whisper_pretrained}')
        print(f'Whisper output dimension: {whisper_dim}')

        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        print('Audio encoder (Whisper) parameters frozen')

        # ====== 2) LLaMA 解码器 + LoRA ======
        vicuna_hf_repo = args.get('vicuna_hf_repo', 'jsun39/kidspeak_vicuna')
        print(f'Initializing language decoder from Hugging Face: {vicuna_hf_repo}')

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(
            vicuna_hf_repo,
            torch_dtype="auto",
            device_map=None
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        print('LLaMA base model parameters frozen, only LoRA adapters are trainable')

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_hf_repo, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print('Language decoder initialized.')

        # ====== 3) 改进的音频处理模块 ======
        # 可配置超参数
        self.conv_kernel = int(args.get('conv_kernel', 5))      # 减小卷积核，减少信息损失
        self.conv_stride = int(args.get('conv_stride', 6))      # 保持下采样率
        self.conv_padding = int(args.get('conv_padding', 2))    # 精确控制padding
        self.ff_variant = str(args.get('ff_variant', 'gelu')).lower()  # 使用更稳定的GELU
        self.ffn_mult = float(args.get('ffn_mult', 2.0))       # 减小MLP容量，防止过拟合
        self.audio_dropout_p = float(args.get('audio_dropout', 0.05))  # 降低dropout
        self.num_gated_blocks = int(args.get('num_gated_blocks', 1))
        self.use_residual_proj = bool(args.get('use_residual_proj', True))  # 新增：残差投影

        # 改进1：更温和的下采样 - 使用平均池化 + 1x1卷积
        self.audio_avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=0)  # 无信息损失的下采样
        self.audio_conv1x1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=1, bias=False)  # 特征重组
        
        # 改进2：预投影层归一化
        self.audio_pre_ln = nn.LayerNorm(whisper_dim)
        
        # 改进3：轻量级门控块（减少复杂度）
        self.gated_block = GatedFFN(
            whisper_dim, 
            mult=self.ffn_mult, 
            variant=self.ff_variant, 
            dropout=self.audio_dropout_p
        )
        
        # 改进4：残差投影连接
        if self.use_residual_proj:
            # 主投影路径
            self.audio_to_llama_main = nn.Linear(whisper_dim, self.llama_model.config.hidden_size)
            # 残差路径（更简单的映射）
            self.audio_to_llama_res = nn.Linear(whisper_dim, self.llama_model.config.hidden_size)
            # 门控权重
            self.proj_gate = nn.Linear(whisper_dim, 1)
        else:
            self.audio_to_llama = nn.Linear(whisper_dim, self.llama_model.config.hidden_size)
        
        # 改进5：输出层归一化
        self.audio_post_ln = nn.LayerNorm(self.llama_model.config.hidden_size)

        # 改进6：智能参数初始化
        self._init_audio_modules()
        
        # 设备
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    
    def _init_audio_modules(self):
        """智能初始化音频处理模块"""
        # 1x1卷积使用单位矩阵初始化（保持特征）
        nn.init.eye_(self.audio_conv1x1.weight.squeeze())
        
        # 投影层使用较小的标准差初始化
        if self.use_residual_proj:
            nn.init.normal_(self.audio_to_llama_main.weight, std=0.02)
            nn.init.normal_(self.audio_to_llama_res.weight, std=0.01)  # 残差路径更小
            nn.init.zeros_(self.audio_to_llama_main.bias)
            nn.init.zeros_(self.audio_to_llama_res.bias)
            # 门控初始化为0.5（平衡两个路径）
            nn.init.constant_(self.proj_gate.weight, 0.0)
            nn.init.constant_(self.proj_gate.bias, 0.0)  # sigmoid(0) = 0.5
        else:
            nn.init.normal_(self.audio_to_llama.weight, std=0.02)
            nn.init.zeros_(self.audio_to_llama.bias)
        
        print('[INFO] Audio modules initialized with improved strategies')
    
    def print_trainable_parameters(self):
        """打印所有可训练参数的详细信息"""
        print('\n=== 可训练参数详情 ===')
        total_params = 0
        audio_params = 0
        lora_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_params += param_count
                
                if name.startswith('audio_') or name.startswith('gated_') or name.startswith('proj_'):
                    audio_params += param_count
                    print(f'[AUDIO] {name}: {param.shape} ({param_count:,} 参数)')
                elif name.startswith('llama_model.') and 'lora' in name:
                    lora_params += param_count
                    print(f'[LoRA] {name}: {param.shape} ({param_count:,} 参数)')
        
        print(f'\n总可训练参数: {total_params:,}')
        print(f'  - 音频模块: {audio_params:,} ({audio_params/total_params*100:.1f}%)')
        print(f'  - LoRA适配器: {lora_params:,} ({lora_params/total_params*100:.1f}%)')
        print('=== 参数统计完成 ===\n')

    def encode_audio(self, audio_paths):
        """
        改进的音频编码方法：
        1. 使用平均池化替代卷积下采样
        2. 添加残差投影连接
        3. 更稳定的归一化和门控
        """
        whisper_model_name = self.args.get('whisper_pretrained', 'small')
        inputs = audio_utils.load_and_transform_audio_data(audio_paths, self.device, whisper_model_name)
        
        # Whisper编码器提取特征 [B, 1500, D]
        with torch.no_grad():
            audio_embeds = self.audio_encoder(inputs)
        
        # 改进1：使用平均池化进行温和下采样（保持信息完整性）
        x = audio_embeds.transpose(1, 2)                    # [B, D, 1500]
        x = self.audio_avgpool(x)                           # [B, D, 250] - 平均池化
        x = self.audio_conv1x1(x)                           # [B, D, 250] - 特征重组
        x = x.transpose(1, 2)                               # [B, 250, D]
        
        # 改进2：预处理归一化
        x = self.audio_pre_ln(x)
        
        # 改进3：轻量级特征增强（单层门控）
        x_enhanced = self.gated_block(x)                    # [B, 250, D]
        
        # 改进4：残差投影连接
        if self.use_residual_proj:
            # 主路径：经过门控处理的特征
            main_proj = self.audio_to_llama_main(x_enhanced)
            # 残差路径：直接投影原始特征
            res_proj = self.audio_to_llama_res(x)
            # 自适应门控权重
            gate_weight = torch.sigmoid(self.proj_gate(x))   # [B, 250, 1]
            # 加权融合
            inputs_llama = gate_weight * main_proj + (1 - gate_weight) * res_proj
        else:
            inputs_llama = self.audio_to_llama(x_enhanced)
        
        # 改进5：输出归一化
        inputs_llama = self.audio_post_ln(inputs_llama)
        
        atts_llama = torch.ones(inputs_llama.size()[:2], dtype=torch.long, device=inputs_llama.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, audio_embeds, input_ids, target_ids, attention_mask):
        """将音频特征与文本prompt融合"""
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        batch_size = audio_embeds.shape[0]

        # 构建prompt前缀
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)

        # 文本嵌入
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)

        # BOS
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)

        # 拼接: [BOS] + [prompt] + [AUDIO] + [TEXT]
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, audio_embeds, p_after_embeds], dim=1)

        # 构建 targets（对 [BOS]+[prompt]+[AUDIO] 屏蔽）
        k = audio_embeds.shape[1]
        empty_targets = torch.ones([batch_size, 1 + p_before_embeds.size(1) + k],
                                   dtype=torch.long, device=self.device).fill_(-100)
        targets = torch.cat([empty_targets, target_ids], dim=1)
        assert inputs_embeds.size(1) == targets.size(1)

        # 注意力掩码
        atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size(1) + k], dtype=torch.long, device=self.device)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size()

        return inputs_embeds, targets, attention_mask

    def forward(self, inputs):
        """模型前向传播"""
        # 1) 音频前缀 token
        audio_paths = inputs['audio_paths']
        audio_embeds, _ = self.encode_audio(audio_paths)

        # 2) 文本输入
        output_texts = inputs['output_texts']
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len
        )

        # 3) 拼接
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            audio_embeds, input_ids, target_ids, attention_mask
        )

        # 4) LLaMA 前向 + CE
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        # 5) 生成 token 准确率（仅作监控）
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        try:
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        except:
            gen_acc = 0.0

        return loss, gen_acc

    def extract_audio_feature(self, inputs):
        """提取音频特征（音频前缀 token）"""
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            return audio_embeds
        else:
            raise ValueError("No audio paths provided")

    def prepare_generation_embedding(self, inputs):
        """为生成准备嵌入向量"""
        prompt = inputs['prompt']
        if len(inputs['modality_embeds']) == 1:
            feature_embeds = inputs['modality_embeds'][0]
        else:
            feature_embeds = self.extract_audio_feature(inputs)
            inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]

        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)

        text = '</Img> ' + prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_after_embeds], dim=1)
        return inputs_embeds

    def generate(self, inputs):
        """生成文本回复"""
        input_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.llama_tokenizer.pad_token_id
        )

        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text
