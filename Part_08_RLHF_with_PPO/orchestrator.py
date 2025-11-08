# Repository layout (Part 8 — RLHF with PPO)
#
#   Part_08_RLHF_with_PPO/
#     orchestrator.py          # run unit tests + optional tiny PPO demo
#     policy.py                # policy = SFT LM + value head (toy head on logits)
#     rollout.py               # prompt formatting, sampling, logprobs/KL utilities
#     ppo_loss.py              # PPO clipped objective + value + entropy + KL penalty
#     train_ppo.py             # single‑GPU RLHF loop (tiny, on‑policy)
#     eval_ppo.py              # compare reward vs. reference on a small set
#     tests/
#       test_ppo_loss.py
#       test_policy_forward.py
#
# Run from inside `Part_08_RLHF_with_PPO/`:
#   cd Part_08_RLHF_with_PPO
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
    p.add_argument("--demo", action="store_true", help="tiny PPO demo")
    args = p.parse_args()
    
    # 1) unit tests
    run("python -m pytest -q tests/test_ppo_loss.py")
    run("python -m pytest -q tests/test_policy_forward.py")
    
    # 2) optional demo (requires SFT+RM checkpoints from Parts 6 & 7)
    if args.demo:
        # run("python train_ppo.py --policy_ckpt ../Part_06_Supervised_Fine_Tuning/runs/sft-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --steps 10 --batch_size 4 --resp_len 128 --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
        
        # run("python train_ppo.py --policy_ckpt ../Part_06_Supervised_Fine_Tuning/runs/sft-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --steps 50 --batch_size 4 --resp_len 128 --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
        
        run("python train_ppo.py --policy_ckpt ../Part_06_Supervised_Fine_Tuning/runs/sft-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --steps 100 --batch_size 4 --resp_len 128 --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
        run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../Part_07_Reward_Modeling/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../Part_04_Scaling_Up/runs/scalingup-demo/tokenizer")
    
    print("\nPart 8 checks complete. ✅")
