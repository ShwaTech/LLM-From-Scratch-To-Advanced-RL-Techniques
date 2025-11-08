import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


part_1 = "Part_01_Core_Transformer_Architecture"
part_2 = "Part_02_Training_A_Tiny_LLM"
part_3 = "Part_03_Modernizing_The_Architecture"
part_4 = "Part_04_Scaling_Up"
part_5 = "Part_05_Mixture_of_Experts"
part_6 = "Part_06_Supervised_Fine_Tuning"
part_7 = "Part_07_Reward_Modeling"



list_of_files = [
    f"{part_1}/attn_mask.py",
    f"{part_1}/attn_numpy_demo.py",
    f"{part_1}/block.py",
    f"{part_1}/demo_mha_shapes.py",
    f"{part_1}/demo_visualize_multi_head.py",
    f"{part_1}/ffn.py",
    f"{part_1}/multi_head.py",
    f"{part_1}/orchestrator.py",
    f"{part_1}/pos_encoding.py",
    f"{part_1}/single_head.py",
    f"{part_1}/vis_utils.py",
    f"{part_1}/tests/test_attn_math.py",
    f"{part_1}/tests/test_causal_mask.py",
    
    f"{part_2}/orchestrator.py",
    f"{part_2}/tokenizer.py",
    f"{part_2}/dataset.py",
    f"{part_2}/tiny.txt",
    f"{part_2}/tiny_hi.txt",
    f"{part_2}/utils.py",
    f"{part_2}/model_gpt.py",
    f"{part_2}/train.py",
    f"{part_2}/sample.py",
    f"{part_2}/eval_loss.py",
    f"{part_2}/tests/test_tokenizer.py",
    f"{part_2}/tests/test_dataset_shift.py",
    
    f"{part_3}/orchestrator.py",
    f"{part_3}/tokenizer.py",
    f"{part_3}/rmsnorm.py",
    f"{part_3}/rope.py",
    f"{part_3}/swiglu.py",
    f"{part_3}/kv_cache.py",
    f"{part_3}/attn_modern.py",
    f"{part_3}/block_modern.py",
    f"{part_3}/model_modern.py",
    f"{part_3}/demo_generate.py",
    f"{part_3}/utils.py",
    f"{part_3}/tests/test_rmsnorm.py",
    f"{part_3}/tests/test_rope_apply.py",
    f"{part_3}/tests/test_kvcache_shapes.py",
    
    f"{part_4}/orchestrator.py",
    f"{part_4}/tokenizer_bpe.py",
    f"{part_4}/dataset_bpe.py",
    f"{part_4}/lr_scheduler.py",
    f"{part_4}/amp_accum.py",
    f"{part_4}/checkpointing.py",
    f"{part_4}/logger.py",
    f"{part_4}/train.py",
    f"{part_4}/sample.py",
    f"{part_4}/tests/test_tokenizer_bpe.py",
    f"{part_4}/tests/test_scheduler.py",
    f"{part_4}/tests/test_resume_shapes.py",
    
    f"{part_5}/orchestrator.py",
    f"{part_5}/README.md",
    f"{part_5}/gating.py",
    f"{part_5}/experts.py",
    f"{part_5}/moe.py",
    f"{part_5}/block_hybrid.py",
    f"{part_5}/demo_moe.py",
    f"{part_5}/tests/test_gate_shapes.py",
    f"{part_5}/tests/test_moe_forward.py",
    f"{part_5}/tests/test_hybrid_block.py",
    
    f"{part_6}/orchestrator.py",
    f"{part_6}/formatters.py",
    f"{part_6}/dataset_sft.py",
    f"{part_6}/collator_sft.py",
    f"{part_6}/curriculum.py",
    f"{part_6}/evaluate.py",
    f"{part_6}/train_sft.py",
    f"{part_6}/sample_sft.py",
    f"{part_6}/tests/test_formatter.py",
    f"{part_6}/tests/test_masking.py",
    
    f"{part_7}/orchestrator.py",
    f"{part_7}/data_prefs.py",
    f"{part_7}/collator_rm.py",
    f"{part_7}/model_reward.py",
    f"{part_7}/loss_reward.py",
    f"{part_7}/train_rm.py",
    f"{part_7}/eval_rm.py",
    f"{part_7}/tests/test_bt_loss.py",
    f"{part_7}/tests/test_reward_forward.py",
    
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")