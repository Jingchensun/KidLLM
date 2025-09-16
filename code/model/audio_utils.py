#!/usr/bin/env python3
"""
音频处理工具模块 - 从ImageBind提取的核心功能
"""

import torch
import torch.nn as nn
import torchaudio
import whisper
import torchvision
import sys
from packaging import version


def fix_torchvision_imports():
    """
    修复torchvision兼容性问题
    """
    torchvision_version = version.parse(torchvision.__version__)
    
    # For torchvision >= 0.15.0, functional_tensor was moved/renamed
    if torchvision_version >= version.parse("0.15.0"):
        # Create a compatibility module for pytorchvideo
        import torchvision.transforms.functional as F
        
        # Create a mock functional_tensor module
        class FunctionalTensorMock:
            def __getattr__(self, name):
                # Redirect to the new functional module
                return getattr(F, name)
        
        # Inject the mock module into torchvision.transforms
        if not hasattr(torchvision.transforms, 'functional_tensor'):
            torchvision.transforms.functional_tensor = FunctionalTensorMock()
        
        # Also handle the case where pytorchvideo tries to import directly
        sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional_tensor


def patch_whisper_layernorm():
    """
    修复Whisper LayerNorm的dtype问题
    """
    def fixed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure both input and weights are in the same dtype
        if x.dtype != self.weight.dtype:
            # Convert input to match weight dtype
            x = x.to(self.weight.dtype)
        
        return nn.LayerNorm.forward(self, x)
    
    # Replace the forward method of Whisper's LayerNorm
    whisper.model.LayerNorm.forward = fixed_forward


def load_and_transform_audio_data(audio_paths, device, whisper_model_name='small'):
    """
    加载和转换音频数据为mel频谱图
    
    Args:
        audio_paths: 音频文件路径列表
        device: 目标设备
        whisper_model_name: Whisper模型名称，用于确定mel频谱维度
        
    Returns:
        torch.Tensor: mel频谱图张量 [batch_size, mel_bins, time_steps]
    """
    sample_rate = 16000
    
    # 根据Whisper模型确定mel频谱维度
    whisper_mel_dims = {
        # OpenAI官方模型
        'tiny': 80,
        'tiny.en': 80,
        'base': 80, 
        'base.en': 80,
        'small': 80,
        'small.en': 80,
        'medium': 80,
        'medium.en': 80,
        'large': 128,
        'large-v1': 128,
        'large-v2': 128,
        'large-v3': 128,
        'large-v3-turbo': 80,  # turbo版本使用80维
        'turbo': 80,
        'ours': 80  # 自定义模型使用small的mel维度
    }
    
    n_mels = whisper_mel_dims.get(whisper_model_name, 80)
    mels = []
    
    for audio_path in audio_paths:
        if audio_path != '':
            # 加载音频文件
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            # print("waveform.shape", waveform.shape) #torch.Size([2, 68208])
            # print("sr", sr) #16000
            
            # 重采样到16kHz（如果需要）
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # 使用Whisper预处理
            waveform = whisper.pad_or_trim(waveform.flatten())
            # print("waveform.shape", waveform.shape) #torch.Size([480000])
            # 根据模型生成相应维度的mel频谱
            if n_mels == 128:
                # 对于large模型，使用更高维度的mel频谱
                mel = whisper.log_mel_spectrogram(waveform, n_mels=128).to(device)
            else:
                # 对于其他模型，使用标准80维mel频谱
                mel = whisper.log_mel_spectrogram(waveform).to(device)
                # print("mel.shape", mel.shape) #torch.Size([80, 3000])
        else:
            # 空音频路径时创建零张量
            mel = torch.zeros(n_mels, 3000).to(device)
        
        mels.append(mel)
    
    mels = torch.stack(mels, dim=0)
    return mels


# 在模块导入时应用修复
fix_torchvision_imports()
patch_whisper_layernorm()
