# Repository layout (Part 9 — RLHF with GRPO)
#
#   Part_09_RLHF_with_GRPO/
#     orchestrator.py          # run unit tests + optional tiny GRPO demo
#     policy.py                # policy = SFT LM generates multiple (group) responses per prompt.
#     rollout.py               # prompt formatting, sampling, logprobs/KL utilities
#     grpo_loss.py             # GRPO clipped objective + Baseline (Group Average) + entropy + KL penalty
#     train_grpo.py            # single‑GPU RLHF loop (tiny, on‑policy)
#     eval_grpo.py             # compare reward vs. reference on a small set
#     tests/
#       test_grpo_loss.py
#
# Run from inside `Part_09_RLHF_with_GRPO/`:
#   cd Part_09_RLHF_with_GRPO
#   python orchestrator.py --demo 
#   pytest -q


import argparse, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent


def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd.split(), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny GRPO demo")
    args = p.parse_args()
    
    # 1) unit tests
    run("python -m pytest -q tests/test_grpo_loss.py")
    
    # 2) optional demo (requires SFT+RM checkpoints from Parts 6 & 7)
    if args.demo:
        run("python train_grpo.py --group_size 4 --policy_ckpt ../Part_06_Supervised_Fine_Tuning/runs/sft-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --steps 200 --batch_prompts 4 --resp_len 128 --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
        run("python eval_grpo.py --policy_ckpt runs/grpo-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
    
    print("\nPart 9 checks complete. ✅")