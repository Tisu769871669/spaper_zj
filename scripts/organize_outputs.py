#!/usr/bin/env python
"""
训练输出文件整理脚本

将训练完成的模型文件从 src/ 目录整理到 outputs/ 目录
保持清晰的目录结构
"""

import os
import shutil
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
outputs_dir = project_root / 'outputs'

print(f"\n{'='*60}")
print(f"  训练输出文件整理工具")
print(f"{'='*60}")
print(f"源目录: {src_dir}")
print(f"目标目录: {outputs_dir}")
print(f"{'='*60}\n")

# 确保outputs目录存在
(outputs_dir / 'models' / 'BiARL').mkdir(parents=True, exist_ok=True)
(outputs_dir / 'models' / 'VanillaPPO').mkdir(parents=True, exist_ok=True)
(outputs_dir / 'models' / 'LSTM').mkdir(parents=True, exist_ok=True)
(outputs_dir / 'checkpoints' / 'BiARL').mkdir(parents=True, exist_ok=True)
(outputs_dir / 'checkpoints' / 'VanillaPPO').mkdir(parents=True, exist_ok=True)
(outputs_dir / 'checkpoints' / 'LSTM').mkdir(parents=True, exist_ok=True)
(outputs_dir / 'results').mkdir(parents=True, exist_ok=True)

moved_files = []
errors = []

def move_file(src_path, dst_path, description):
    """移动文件并记录"""
    try:
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            moved_files.append((src_path.name, str(dst_path)))
            print(f"✅ {description}")
            print(f"   {src_path.name} → {dst_path.relative_to(outputs_dir)}")
            return True
        else:
            print(f"⚠️  文件不存在: {src_path.name}")
            return False
    except Exception as e:
        errors.append((src_path.name, str(e)))
        print(f"❌ 错误: {src_path.name} - {e}")
        return False

# 1. 整理 Bi-ARL 模型
print("\n1️⃣  整理 Bi-ARL 模型...")
print("-" * 60)
for seed in [42, 101, 202]:
    seed_dir = outputs_dir / 'models' / 'BiARL' / f'seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Attacker 模型
    src_file = src_dir / f'attacker_bilevel_seed{seed}.pth'
    dst_file = seed_dir / 'attacker.pth'
    move_file(src_file, dst_file, f'Bi-ARL Attacker (seed {seed})')
    
    # Defender 模型
    src_file = src_dir / f'defender_bilevel_seed{seed}.pth'
    dst_file = seed_dir / 'defender.pth'
    move_file(src_file, dst_file, f'Bi-ARL Defender (seed {seed})')

# 2. 整理 Vanilla PPO 模型
print("\n2️⃣  整理 Vanilla PPO 模型...")
print("-" * 60)
for seed in [42, 101, 202]:
    seed_dir = outputs_dir / 'models' / 'VanillaPPO' / f'seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    src_file = src_dir / f'defender_vanilla_ppo_seed{seed}.pth'
    dst_file = seed_dir / 'model.pth'
    move_file(src_file, dst_file, f'Vanilla PPO (seed {seed})')

# 3. 整理 LSTM-IDS 模型
print("\n3️⃣  整理 LSTM-IDS 模型...")
print("-" * 60)
for seed in [42, 101, 202]:
    seed_dir = outputs_dir / 'models' / 'LSTM' / f'seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    src_file = src_dir / f'lstm_ids_seed{seed}.pth'
    dst_file = seed_dir / 'model.pth'
    move_file(src_file, dst_file, f'LSTM-IDS (seed {seed})')

# 4. 整理检查点文件
print("\n4️⃣  整理检查点文件...")
print("-" * 60)
checkpoints_src = src_dir / 'checkpoints'
if checkpoints_src.exists():
    for seed_dir in checkpoints_src.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith('seed'):
            seed = seed_dir.name.replace('seed', '')
            
            # 假设是Bi-ARL的检查点
            dst_seed_dir = outputs_dir / 'checkpoints' / 'BiARL' / f'seed{seed}'
            
            for ckpt_file in seed_dir.glob('*.pth'):
                dst_file = dst_seed_dir / ckpt_file.name
                move_file(ckpt_file, dst_file, f'Checkpoint {ckpt_file.name}')
    
    # 删除空的checkpoints目录
    try:
        if checkpoints_src.exists() and not any(checkpoints_src.iterdir()):
            checkpoints_src.rmdir()
            print(f"✅ 已删除空目录: src/checkpoints")
    except:
        pass

# 5. 整理实验结果CSV
print("\n5️⃣  整理实验结果...")
print("-" * 60)
src_file = src_dir / 'experiment_results.csv'
dst_file = outputs_dir / 'results' / 'experiment_results.csv'
move_file(src_file, dst_file, 'Experiment Results CSV')

# 6. 清理src目录下的其他.pth文件
print("\n6️⃣  清理其他临时文件...")
print("-" * 60)
for pth_file in src_dir.glob('*.pth'):
    if pth_file.is_file():
        print(f"⚠️  发现未分类的模型文件: {pth_file.name}")
        # 可以选择移动到outputs/models/未分类/
        unkown_dir = outputs_dir / 'models' / 'Unknown'
        unkown_dir.mkdir(parents=True, exist_ok=True)
        dst_file = unkown_dir / pth_file.name
        move_file(pth_file, dst_file, f'未分类文件 {pth_file.name}')

# 总结
print(f"\n{'='*60}")
print(f"  整理完成!")
print(f"{'='*60}")
print(f"✅ 成功移动: {len(moved_files)} 个文件")
if errors:
    print(f"❌ 错误: {len(errors)} 个文件")
else:
    print(f"❌ 错误: 0 个文件")

if moved_files:
    print(f"\n📋 已移动的文件:")
    for filename, dst in moved_files[:10]:  # 只显示前10个
        print(f"  • {filename}")
    if len(moved_files) > 10:
        print(f"  ... 还有 {len(moved_files) - 10} 个文件")

print(f"\n📁 输出目录结构:")
print(f"  {outputs_dir}/")
print(f"  ├── models/")
print(f"  │   ├── BiARL/")
print(f"  │   ├── VanillaPPO/")
print(f"  │   └── LSTM/")
print(f"  ├── checkpoints/")
print(f"  └── results/")

print(f"\n{'='*60}\n")
