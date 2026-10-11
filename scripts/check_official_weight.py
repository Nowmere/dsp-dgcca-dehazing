import torch
import sys

ckpt = torch.load('ots_train_ffa_3_19.pk', map_location='cpu')
print('='*60)
print('文件: ots_train_ffa_3_19.pk')
print('文件类型:', type(ckpt))
print('='*60)

if isinstance(ckpt, dict):
    print('Keys:', list(ckpt.keys()))
    print()
    for k in list(ckpt.keys())[:5]:
        v = ckpt[k]
        if isinstance(v, dict):
            print(f'  {k}: dict with keys {list(v.keys())[:5]}... (共{len(v)}项)')
        elif isinstance(v, torch.Tensor):
            print(f'  {k}: Tensor shape {v.shape}')
        else:
            print(f'  {k}: {type(v).__name__} = {str(v)[:80]}')

elif hasattr(ckpt, 'keys'):
    print('类字典格式(OrderedDict)')
    keys = list(ckpt.keys())
    print('参数数量:', len(keys))
    print('前10个key:', keys[:10])
    print('后10个key:', keys[-10:])
    
    # 检查是否包含 FFA-Net 的关键模块
    has_pre_conv = any('pre_conv' in k or 'conv1' in k for k in keys)
    has_g1 = any('g1' in k for k in keys)
    has_g2 = any('g2' in k for k in keys)
    has_g3 = any('g3' in k for k in keys)
    has_depth = any('depth' in k for k in keys)
    has_fusion = any('fusion' in k or 'cross' in k or 'attn' in k for k in keys)
    has_post_conv = any('post_conv' in k or 'conv' in k for k in keys[-10:])
    has_ca = any('calayer' in k or 'ca' in k.lower() for k in keys)
    has_pa = any('palayer' in k or 'pa' in k.lower() for k in keys)
    
    print()
    print('模块检测:')
    print(f'  pre_conv (输入层): {"✅" if has_pre_conv else "❌"}')
    print(f'  g1/g2/g3 (FFA组): {"✅" if has_g1 and has_g2 and has_g3 else "❌"}')
    print(f'  post_conv (输出层): {"✅" if has_post_conv else "❌"}')
    print(f'  CA/PA (注意力): {"✅" if has_ca and has_pa else "❌"}')
    print(f'  depth分支: {"✅" if has_depth else "❌"}')
    print(f'  融合模块: {"✅" if has_fusion else "❌"}')
    
    print()
    if not has_depth and not has_fusion and has_pre_conv and has_g1 and has_ca:
        print('🔍 结论: 这是 纯FFA-Net (Baseline) 的训练权重')
        print('   不包含深度特征提取和跨模态融合模块')
        print('   与我们 checkpoints_baseline/ 中的权重是同一种模型')
    elif has_depth and has_fusion:
        print('🔍 结论: 这是 FFA_DGCCA_Net (深度增强版) 的训练权重')
        print('   包含深度引导跨模态注意力模块')
    else:
        print('🔍 结论: 未知网络结构，请进一步检查')
else:
    print('其他格式:', str(type(ckpt)))
    if isinstance(ckpt, (list, tuple)):
        print(f'长度: {len(ckpt)}')
        for i, item in enumerate(ckpt[:3]):
            print(f'  [{i}]: {type(item).__name__}')

# 检查文件大小
import os
size_mb = os.path.getsize('ots_train_ffa_3_19.pk') / 1e6
print(f'\n文件大小: {size_mb:.2f} MB')