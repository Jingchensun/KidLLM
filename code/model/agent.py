from .header import *

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        if args['stage'] == 2:
            self.load_stage_1_parameters(args["delta_ckpt_path"])
            print(f'[!] load stage 1 checkpoint from {args["delta_ckpt_path"]}')

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model, 
            model_parameters=self.model.parameters(),
            config_params=ds_params, 
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc

    def eval_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.eval()
        loss, mle_acc = self.ds_engine(batch)

        #self.ds_engine.backward(loss)
        #self.ds_engine.step()
        pbar.set_description(f'[!] val_loss: {round(loss.item(), 4)}; val_token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; val_loss: {round(loss.item(), 4)}; val_token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc
    
    def save_model(self, path, epoch):
        # 只保存LoRA适配器参数和其他可训练参数
        save_dict = {}
        
        # 1. 保存LoRA适配器参数
        if hasattr(self.ds_engine.module, 'llama_model'):
            llama_model = self.ds_engine.module.llama_model
            if hasattr(llama_model, 'peft_config'):
                # 使用PEFT的保存方法，只保存适配器参数
                peft_state_dict = get_peft_model_state_dict(llama_model)
                for k, v in peft_state_dict.items():
                    save_dict[f'llama_model.{k}'] = v
                print(f'[!] Saved {len(peft_state_dict)} LoRA parameters')
        
        # 2. 保存其他可训练参数（音频处理模块）
        state_dict = self.ds_engine.module.state_dict()
        trainable_params = []
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad and not k.startswith('llama_model.') and not k.startswith('audio_encoder.'):
                # 保存新的音频处理模块参数
                save_dict[k] = state_dict[k]
                trainable_params.append(k)
                print(f'[!] Saved trainable parameter: {k} (shape: {v.shape})')
        
        print(f'[!] Total trainable audio modules saved: {len(trainable_params)}')
        
        # 检查关键模块是否都被保存
        if hasattr(self.ds_engine.module, 'use_residual_proj') and self.ds_engine.module.use_residual_proj:
            expected_modules = ['audio_conv1x1', 'audio_pre_ln', 'gated_block', 'audio_to_llama_main', 'audio_to_llama_res', 'proj_gate', 'audio_post_ln']
        else:
            expected_modules = ['audio_conv1x1', 'audio_pre_ln', 'gated_block', 'audio_to_llama', 'audio_post_ln']
            
        for module in expected_modules:
            found = any(module in param for param in trainable_params)
            print(f'[!] Module {module}: {"✓ Found" if found else "✗ Missing"}')
        
        # 保存架构配置信息
        config_info = {
            'use_residual_proj': getattr(self.ds_engine.module, 'use_residual_proj', True),
            'ff_variant': getattr(self.ds_engine.module, 'ff_variant', 'gelu'),
            'ffn_mult': getattr(self.ds_engine.module, 'ffn_mult', 2.0),
            'audio_dropout_p': getattr(self.ds_engine.module, 'audio_dropout_p', 0.05),
        }
        save_dict['_architecture_config'] = config_info
        print(f'[!] Saved architecture config: {config_info}')

        filename = f'{path}/pytorch_model_{epoch}.pt'
        torch.save(save_dict, filename)
        
        # 保存tokenizer和配置（这些文件很小）
        self.model.llama_tokenizer.save_pretrained(path)
        self.model.llama_model.config.save_pretrained(path)
        
        # 计算保存的文件大小
        import os
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f'[!] Save optimized model into {path}')
        print(f'[!] Model file size: {file_size:.2f} MB (only LoRA + trainable params)')
        print(f'[!] Total parameters saved: {len(save_dict)}')

    def load_stage_1_parameters(self, path):
        print(f'[INFO] Loading stage 1 optimized checkpoint from {path}')
        delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
        
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
            set_peft_model_state_dict(self.model.llama_model, lora_state_dict)
            print(f'[INFO] Loaded {len(lora_state_dict)} LoRA parameters from stage 1')
        
        # 加载其他参数（使用strict=False以兼容新增模块）
        if other_state_dict:
            missing_keys, unexpected_keys = self.model.load_state_dict(other_state_dict, strict=False)
            print(f'[INFO] Loaded {len(other_state_dict)} other trainable parameters from stage 1')
            
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
                print('[INFO] All expected trainable parameters loaded from stage 1')
                
            if unexpected_keys:
                print(f'[WARNING] Unexpected keys (ignored): {unexpected_keys}')
        
        print('[INFO] Stage 1 optimized checkpoint loaded successfully')


