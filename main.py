# -*- coding: utf-8 -*-

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
main.py - 最终主攻击实验脚本 (专业版)
【v3版】：增加了instruction_list内容预览功能。
"""
import torch
import pandas as pd
import random
import os
import pickle
import numpy as np
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
from llm_attacks.minimal_gcg.string_utils import load_conversation_template

from data_manager import prepare_harmbench_instructions, download_and_load_demo_pool
from FsAttack import optimization_based_search

def main():
    """主执行函数"""
    
    # =====================================================================================
    # 1. 实验配置
    # =====================================================================================
    MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
    TEMPLATE_NAME = "llama-2"
    NUM_BEHAVIORS_TO_ATTACK = 20
    SHOTS = 8
    NUM_STEPS = 128
    BATCH_SIZE = 64
    TOP_K = 256
    TARGET_RESPONSE_START = "Step"
    NUM_SEEDS = 5
    RESULTS_CSV_FILE = "final_summary_results_optimized.csv"
    RESULTS_PKL_DIR = f"saved/opt_{SHOTS}"

    # =====================================================================================
    # 2. 准备数据
    # =====================================================================================
    instruction_list = prepare_harmbench_instructions(save_path='data/my_harmbench_instruction_list.pkl')
    demo_pool = download_and_load_demo_pool(file_path='data/mistral_demonstration_list_official.pkl')

    if not instruction_list or not demo_pool:
        print("❌ 数据准备失败，程序终止。")
        return
        
    os.makedirs(RESULTS_PKL_DIR, exist_ok=True)

    # 【新增功能】检查 instruction_list 的前5个内容
    # ==================================================================
    print("\n--- 检查 instruction_list 前5项内容 ---")
    if instruction_list and len(instruction_list) > 0:
        # 只遍历前5个，如果列表不足5个，则遍历全部
        for i, instruction in enumerate(instruction_list[:5]):
            # 打印前200个字符作为预览，避免内容过长刷屏
            print(f"[{i+1}] {instruction[:200]}...")
    else:
        print("instruction_list 为空或未成功加载。")
    print("----------------------------------------\n")
    # ==================================================================

    # =====================================================================================
    # 3. 加载目标模型
    # =====================================================================================
    print(f"\n>>> 步骤3: 正在加载目标模型: {MODEL_PATH}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, low_cpu_mem_usage=True, use_cache=False)
    if 'llama-2' in MODEL_PATH:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to('cuda')
    model.requires_grad_(False)
    conv_template = load_conversation_template(TEMPLATE_NAME)
    print("✅ 目标模型和对话模板加载成功！")

    # =====================================================================================
    # 4. 运行主攻击实验循环 (包含多随机种子)
    # =====================================================================================
    summary_results_list = []
    behaviors_to_test_indices = random.sample(range(len(instruction_list)), k=NUM_BEHAVIORS_TO_ATTACK)

    print(f"\n🚀 开始对 {len(behaviors_to_test_indices)} 个随机选择的有害行为进行攻击 (每个行为将运行{NUM_SEEDS}次)...")

    for i, target_index in enumerate(behaviors_to_test_indices):
        target_instruction = instruction_list[target_index]
        
        for seed in range(NUM_SEEDS):
            print(f"\n--- [攻击 {i*NUM_SEEDS+seed+1}/{NUM_BEHAVIORS_TO_ATTACK*NUM_SEEDS} | 目标索引: {target_index} | 随机种子: {seed}] ---")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            _ , detailed_log = optimization_based_search(
                model=model, tokenizer=tokenizer, conv_template=conv_template,
                instruction=target_instruction, target=TARGET_RESPONSE_START,
                demo_pool=demo_pool, num_steps=NUM_STEPS, shots=SHOTS, 
                batch_size=BATCH_SIZE, top_k=TOP_K
            )

            log_filename = f"{RESULTS_PKL_DIR}/seed_{seed}_pool_{len(demo_pool)}_shots_{SHOTS}_index_{target_index}.pkl"
            with open(log_filename, 'wb') as f:
                pickle.dump(detailed_log, f)
            print(f"  >>> 详细过程日志已保存到: {log_filename}")

            final_result = detailed_log[-1]
            summary_results_list.append({
                "target_index": target_index,
                "seed": seed,
                "is_jailbroken": final_result['judge'],
                "final_response": final_result['gen_str'],
            })
            
            df_summary = pd.DataFrame(summary_results_list)
            df_summary.to_csv(RESULTS_CSV_FILE, index=False, encoding='utf-8-sig')

    print(f"\n\n🎉 实验完成！摘要结果已保存到 {RESULTS_CSV_FILE}")

    # =====================================================================================
    # 5. 最终结果分析
    # =====================================================================================
    if summary_results_list:
        df_final = pd.DataFrame(summary_results_list)
        successful_attacks = df_final['is_jailbroken'].sum()
        total_attacks = len(df_final)
        asr = (successful_attacks / total_attacks) * 100 if total_attacks > 0 else 0
        print("\n--- 最终实验结果总结 ---")
        print(f"总计攻击次数 (行为数 x 种子数): {total_attacks}")
        print(f"成功越狱数: {successful_attacks}")
        print(f"最终攻击成功率 (ASR): {asr:.2f}%")

if __name__ == "__main__":
    main()
