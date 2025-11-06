import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


part_1 = "Part_01_Core_Transformer_Architecture"
part_2 = "Part_02_Training_A_Tiny_LLM"



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
    f"{part_2}/utils.py",
    f"{part_2}/model_gpt.py",
    f"{part_2}/train.py",
    f"{part_2}/sample.py",
    f"{part_2}/eval_loss.py",
    f"{part_2}/tests/test_tokenizer.py",
    f"{part_2}/tests/test_dataset_shift.py"
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