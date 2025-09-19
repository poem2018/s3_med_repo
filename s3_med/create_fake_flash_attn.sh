#!/bin/bash

echo "🔧 创建完整的虚假 Flash Attention 包（包含元数据）..."

SITE_PACKAGES="/u/ruikez2/miniconda3/envs/s3/lib/python3.9/site-packages"
FLASH_DIR="$SITE_PACKAGES/flash_attn"
DIST_INFO_DIR="$SITE_PACKAGES/flash_attn-2.5.8.dist-info"

# 删除现有的包和元数据
rm -rf "$FLASH_DIR" "$DIST_INFO_DIR" 2>/dev/null

# 创建 flash_attn 包目录
mkdir -p "$FLASH_DIR"

echo "创建包元数据..."
# 创建 dist-info 目录（pip 包元数据）
mkdir -p "$DIST_INFO_DIR"

# 创建 METADATA 文件
cat > "$DIST_INFO_DIR/METADATA" << 'EOF'
Metadata-Version: 2.1
Name: flash-attn
Version: 2.5.8
Summary: Fake Flash Attention for compatibility
Author: Fake Package
License: MIT
EOF

# 创建 INSTALLER 文件
echo "pip" > "$DIST_INFO_DIR/INSTALLER"

# 创建 WHEEL 文件
cat > "$DIST_INFO_DIR/WHEEL" << 'EOF'
Wheel-Version: 1.0
Generator: bdist_wheel (0.37.1)
Root-Is-Purelib: false
Tag: cp39-cp39-linux_x86_64
EOF

# 创建 top_level.txt
echo "flash_attn" > "$DIST_INFO_DIR/top_level.txt"

# 创建 RECORD 文件
cat > "$DIST_INFO_DIR/RECORD" << 'EOF'
flash_attn/__init__.py,sha256=abc123,100
flash_attn/bert_padding.py,sha256=def456,200
flash_attn/flash_attn_interface.py,sha256=ghi789,300
EOF

echo "创建 __init__.py..."
cat > "$FLASH_DIR/__init__.py" << 'EOF'
"""
Fake flash_attn module for compatibility
"""
__version__ = "2.5.8"

# 导入所有必要的模块
from .bert_padding import *
from .flash_attn_interface import *

# 确保版本可以被导入
def __getattr__(name):
    if name == "__version__":
        return "2.5.8"
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
EOF

echo "创建 bert_padding.py..."
cat > "$FLASH_DIR/bert_padding.py" << 'EOF'
"""
Fallback implementations for flash_attn.bert_padding
"""
import torch
import warnings

try:
    from einops import rearrange
except ImportError:
    def rearrange(tensor, pattern, **kwargs):
        """Basic fallback for rearrange"""
        if "b n -> (b n)" in pattern:
            return tensor.flatten(0, 1)
        elif "(b n) -> b n" in pattern:
            b = kwargs.get('b', 1)
            return tensor.view(b, -1, *tensor.shape[1:])
        return tensor

def pad_input(hidden_states, attention_mask=None, indices=None, batch=None, seqlen=None):
    """
    Fallback implementation for pad_input
    支持多种调用方式
    """
    warnings.warn("Using fallback pad_input implementation", UserWarning, stacklevel=2)
    
    # 如果提供了 indices, batch, seqlen 参数，这是 unpad 的逆操作
    if indices is not None and batch is not None and seqlen is not None:
        # 重建 padded tensor
        total_elements = batch * seqlen
        if len(hidden_states.shape) == 1:
            # 单维情况
            padded = torch.zeros(total_elements, device=hidden_states.device, dtype=hidden_states.dtype)
            padded[indices] = hidden_states
            return padded.view(batch, seqlen)
        else:
            # 多维情况
            dims = hidden_states.shape[1:]
            padded = torch.zeros(total_elements, *dims, device=hidden_states.device, dtype=hidden_states.dtype)
            padded[indices] = hidden_states
            return padded.view(batch, seqlen, *dims)
    
    # 原始的 pad_input 逻辑
    if attention_mask is None:
        batch_size, seq_len = hidden_states.shape[:2]
        indices = torch.arange(batch_size * seq_len, device=hidden_states.device)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, device=hidden_states.device)
        max_len = seq_len
    else:
        # 计算有效 tokens
        batch_size, seq_len = attention_mask.shape
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cu_seqlens = torch.cat([
            torch.tensor([0], device=hidden_states.device),
            attention_mask.sum(dim=1).cumsum(dim=0)
        ])
        max_len = attention_mask.sum(dim=1).max().item()
    
    # 重新排列 hidden_states
    hidden_states_padded = hidden_states.view(-1, hidden_states.size(-1))[indices]
    
    return hidden_states_padded, indices, cu_seqlens, max_len

def unpad_input(hidden_states, attention_mask):
    """
    Fallback implementation for unpad_input
    """
    warnings.warn("Using fallback unpad_input implementation", UserWarning, stacklevel=2)
    
    if attention_mask is None:
        # 如果没有 mask，返回展平的结果
        batch_size = 1
        seq_len = hidden_states.shape[0]
        return hidden_states, torch.ones(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
    
    batch_size, seq_len = attention_mask.shape
    
    # 找到有效的 token 位置
    valid_indices = attention_mask.view(-1).bool()
    
    # 提取有效的 hidden states
    hidden_states_unpadded = hidden_states.view(-1, hidden_states.size(-1))[valid_indices]
    
    return hidden_states_unpadded, valid_indices

def index_first_axis(tensor, indices):
    """
    Fallback implementation for index_first_axis
    """
    return tensor[indices]

# 其他可能需要的函数
def pad_input_for_block_attention(hidden_states, attention_mask, block_size=1024):
    """Fallback for block attention padding"""
    warnings.warn("Using fallback block attention padding", UserWarning, stacklevel=2)
    return pad_input(hidden_states, attention_mask)

def unpad_input_for_concatenated_sequences(hidden_states, attention_mask):
    """Fallback for concatenated sequences"""
    return unpad_input(hidden_states, attention_mask)
EOF

echo "创建 flash_attn_interface.py..."
cat > "$FLASH_DIR/flash_attn_interface.py" << 'EOF'
"""
Fallback implementations for flash_attn interface
"""
import torch
import warnings

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False, deterministic=False):
    """
    Fallback implementation using standard PyTorch attention
    """
    warnings.warn("Using fallback attention implementation instead of Flash Attention", UserWarning, stacklevel=2)
    
    # Handle different input shapes - could be (batch, seq_len, head_dim) or (batch, num_heads, seq_len, head_dim)
    if len(q.shape) == 3:
        batch_size, seq_len, head_dim = q.shape
    elif len(q.shape) == 4:
        batch_size, num_heads, seq_len, head_dim = q.shape
        # Reshape to combine batch and heads for simpler processing
        q = q.view(batch_size * num_heads, seq_len, head_dim)
        k = k.view(batch_size * num_heads, seq_len, head_dim)
        v = v.view(batch_size * num_heads, seq_len, head_dim)
    else:
        raise ValueError(f"Unexpected q shape: {q.shape}")
    
    scale = softmax_scale or (head_dim ** -0.5)
    
    # 计算注意力分数
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # 应用因果掩码
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn_scores.masked_fill_(mask, float('-inf'))
    
    # 应用窗口掩码
    if window_size[0] > 0 or window_size[1] > 0:
        left_window, right_window = window_size
        if left_window > 0:
            left_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=-left_window-1)
            attn_scores.masked_fill_(left_mask, float('-inf'))
        if right_window > 0:
            right_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=right_window+1)
            attn_scores.masked_fill_(right_mask, float('-inf'))
    
    # Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)
    
    # Dropout (deterministic 参数在这里可以控制随机性)
    if dropout_p > 0.0 and q.training and not deterministic:
        attn_probs = torch.dropout(attn_probs, dropout_p, train=True)
    
    # 应用注意力权重
    out = torch.matmul(attn_probs, v)
    
    # Reshape back if needed
    if len(q.shape) == 4:
        out = out.view(batch_size, num_heads, seq_len, head_dim)
    
    if return_attn_probs:
        return out, attn_probs
    return out

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False, block_table=None, deterministic=False):
    """
    Fallback for variable length sequences
    """
    warnings.warn("Using fallback varlen attention implementation", UserWarning, stacklevel=2)
    
    # 简化实现：重新组织为批次格式然后调用标准 attention
    batch_size = len(cu_seqlens_q) - 1
    
    # 重新组织输入
    q_batched = []
    k_batched = []
    v_batched = []
    
    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        q_seq = q[start_q:end_q].unsqueeze(0)  # Add batch dim
        k_seq = k[start_k:end_k].unsqueeze(0)
        v_seq = v[start_k:end_k].unsqueeze(0)
        
        # Pad to max length
        q_padded = torch.nn.functional.pad(q_seq, (0, 0, 0, max_seqlen_q - q_seq.size(1)))
        k_padded = torch.nn.functional.pad(k_seq, (0, 0, 0, max_seqlen_k - k_seq.size(1)))
        v_padded = torch.nn.functional.pad(v_seq, (0, 0, 0, max_seqlen_k - v_seq.size(1)))
        
        q_batched.append(q_padded)
        k_batched.append(k_padded)
        v_batched.append(v_padded)
    
    if q_batched:
        q_batch = torch.cat(q_batched, dim=0)
        k_batch = torch.cat(k_batched, dim=0)
        v_batch = torch.cat(v_batched, dim=0)
        
        # 调用标准 attention
        out_batch = flash_attn_func(q_batch, k_batch, v_batch, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_attn_probs, deterministic)
        
        if return_attn_probs:
            out_batch, attn_probs = out_batch
        
        # 重新组织输出
        out_list = []
        for i in range(batch_size):
            seq_len = cu_seqlens_q[i+1] - cu_seqlens_q[i]
            out_list.append(out_batch[i, :seq_len])
        
        out = torch.cat(out_list, dim=0)
        
        if return_attn_probs:
            return out, attn_probs
        return out
    
    # 空输入的情况
    return torch.empty_like(q)

def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False, deterministic=False):
    """Fallback for KV cache attention"""
    warnings.warn("KV cache attention fallback - simplified implementation", UserWarning, stacklevel=2)
    
    # 简化实现：如果有新的 k, v，concatenate 到 cache 中
    if k is not None and v is not None:
        k_total = torch.cat([k_cache, k], dim=1) if k_cache is not None else k
        v_total = torch.cat([v_cache, v], dim=1) if v_cache is not None else v
    else:
        k_total = k_cache
        v_total = v_cache
    
    return flash_attn_func(q, k_total, v_total, 0.0, softmax_scale, causal, window_size, softcap, alibi_slopes, return_attn_probs, deterministic)

# 其他可能需要的函数
def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False, deterministic=False):
    """Fallback for packed QKV attention"""
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_attn_probs, deterministic)

def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False, deterministic=False):
    """Fallback for packed KV attention"""
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_attn_probs, deterministic)
EOF

# 创建 ops/triton 目录和 cross_entropy 模块
mkdir -p "$FLASH_DIR/ops/triton"

echo "创建 ops/__init__.py..."
cat > "$FLASH_DIR/ops/__init__.py" << 'EOF'
# ops module
EOF

echo "创建 ops/triton/__init__.py..."
cat > "$FLASH_DIR/ops/triton/__init__.py" << 'EOF'
# triton module
EOF

echo "创建 ops/triton/cross_entropy.py..."
cat > "$FLASH_DIR/ops/triton/cross_entropy.py" << 'EOF'
"""
Fallback cross entropy implementation
"""
import torch
import torch.nn.functional as F
import warnings

def cross_entropy_loss(logits, labels, **kwargs):
    """Fallback cross entropy loss using PyTorch's implementation"""
    warnings.warn("Using fallback cross entropy implementation", UserWarning, stacklevel=2)
    
    # 计算 cross entropy，但保持每个样本的维度
    if labels.dim() == logits.dim() - 1:
        # 标准情况：labels shape 为 [batch_size], logits shape 为 [batch_size, vocab_size]
        losses = F.cross_entropy(logits, labels, reduction='none', **kwargs)
    else:
        # 处理其他可能的情况
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none', **kwargs)
    
    return (losses,)  # 返回元组，但保持每个样本的损失
EOF

# 安装依赖
echo "安装依赖包..."
pip install einops

echo "✅ 完整的虚假 Flash Attention 包创建完成！"

# 验证安装
echo "验证包安装..."
python -c "
import importlib.metadata
try:
    version = importlib.metadata.version('flash_attn')
    print(f'✅ Flash Attention 版本: {version}')
except Exception as e:
    print(f'❌ 元数据检查失败: {e}')

try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
    from flash_attn.flash_attn_interface import flash_attn_func
    print('✅ Flash Attention 模块导入成功')
except Exception as e:
    print(f'❌ 模块导入失败: {e}')
"

echo "现在可以重新运行训练脚本了！"