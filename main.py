# -*- coding: utf-8 -*-
"""
main.py - 总调度器脚本

该脚本负责自动循环调用 FsAttack.py 来执行一系列的攻击实验。
"""
import os
import random

def main():
    # =====================================================================================
    # 1. 批量实验配置
    # =====================================================================================
    NUM_BEHAVIORS_TO_ATTACK = 10
    NUM_SEEDS = 3
    # 其他参数可以在 FsAttack.py 的 argparse 中设置默认值，或在这里指定
    # 例如： BATCH_SIZE = 32
    # command_template += f" --batch_size {BATCH_SIZE}"
    
    # 假设您的指令列表总共有 200 个
    TOTAL_INSTRUCTIONS =200
    
    # =====================================================================================
    # 2. 运行批量攻击循环
    # =====================================================================================
    
    # 随机选择要攻击的目标指令索引
    behaviors_to_test_indices = random.sample(range(TOTAL_INSTRUCTIONS), k=NUM_BEHAVIORS_TO_ATTACK)
    
    print(f"🚀 将要对 {len(behaviors_to_test_indices)} 个目标进行攻击, 每个目标运行 {NUM_SEEDS} 个随机种子。")
    print(f"攻击目标索引: {behaviors_to_test_indices}")
    
    total_runs = len(behaviors_to_test_indices) * NUM_SEEDS
    current_run = 0

    for target_index in behaviors_to_test_indices:
        for seed in range(NUM_SEEDS):
            current_run += 1
            print(f"\n--- [正在执行: {current_run}/{total_runs}] | 目标索引: {target_index} | 随机种子: {seed}] ---")
            
            # 构建命令行指令来调用 FsAttack.py
            command = (
                f"python FsAttack.py "
                f"--user_prompt_index {target_index} "
                f"--seed {seed}"
                # 您可以在这里添加或覆盖 FsAttack.py 中的任何默认参数
                # f" --num_steps 256" 
            )
            
            print(f"执行命令: {command}")
            os.system(command)

    print("\n\n🎉 所有批量实验执行完毕！")

if __name__ == "__main__":
    main()

