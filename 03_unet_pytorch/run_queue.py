"""Sequential run queue for the v5 model comparison.

Runs every (architecture x hard-negative) configuration on the GPU, three
seeds each, and skips any configuration whose metrics_all_seeds.csv already
exists so the queue is resumable after an interruption.

    python 03_unet_pytorch/run_queue.py            # run the queue
    python 03_unet_pytorch/run_queue.py --status   # print progress, run nothing
    python 03_unet_pytorch/run_queue.py --dry-run  # list what would run
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-torch", "Scripts", "python.exe")
TRAIN = os.path.join(ROOT, "03_unet_pytorch", "01_train_unet_v5_torch.py")
RESULTS = os.path.join(ROOT, "results", "03_unet_pytorch")
LOGS = os.path.join(ROOT, "logs")
STATUS = os.path.join(RESULTS, "queue_status.json")

DEV = r"data\tiled_data\partitioned\modelDevelopment"
TEST = r"data\tiled_data\partitioned\Validation"
HN_FULL = r"data\tiled_data\finetuning_hard_negatives_complete"
HN_NEG = r"data\tiled_data\complementary_hard_negatives_v2"

ARCHS = ["scratch", "efficientnet-b0", "resnet34", "dinov2"]
MAX_NEG = 2000          # subsample of the 26,597 binary negative tiles
SEEDS = ["1", "2", "3"]


def jobs():
    out = []
    for hn in (False, True):          # plain runs first: they are the baseline
        for arch in ARCHS:
            name = f"torch_{arch.replace('-', '')}" + ("_hn" if hn else "")
            cmd = [PY, TRAIN, "--dev-dir", DEV, "--test-dir", TEST,
                   "--arch", arch, "--exp-name", name,
                   "--split-seed", "42", "--seeds", *SEEDS]
            if hn:
                cmd += ["--extra-train-dirs", HN_FULL,
                        "--negative-train-dirs", HN_NEG,
                        "--max-negatives", str(MAX_NEG)]
            out.append({"name": name, "arch": arch, "hard_negatives": hn,
                        "cmd": cmd})
    return out


def done(name):
    return os.path.exists(os.path.join(RESULTS, name, "metrics_all_seeds.csv"))


def write_status(js, current=None, note=""):
    os.makedirs(RESULTS, exist_ok=True)
    with open(STATUS, "w") as f:
        json.dump({
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "current": current, "note": note,
            "done": [j["name"] for j in js if done(j["name"])],
            "pending": [j["name"] for j in js
                        if not done(j["name"]) and j["name"] != current],
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    js = jobs()

    if a.status or a.dry_run:
        for j in js:
            print(f"  [{'done' if done(j['name']) else '    '}] {j['name']}")
        n = sum(done(j["name"]) for j in js)
        print(f"\n{n}/{len(js)} configurations complete")
        if a.dry_run:
            for j in js:
                if not done(j["name"]):
                    print("\n" + " ".join(j["cmd"]))
        return

    os.makedirs(LOGS, exist_ok=True)
    for j in js:
        if done(j["name"]):
            print(f"SKIP {j['name']} (already complete)")
            continue
        log = os.path.join(LOGS, f"{j['name']}.log")
        print(f"\n{'='*70}\nRUN  {j['name']}  ->  {log}\n{'='*70}", flush=True)
        write_status(js, current=j["name"], note="training")
        t0 = dt.datetime.now()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with open(log, "w") as fh:
            rc = subprocess.call(j["cmd"], stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=ROOT, env=env)
        mins = (dt.datetime.now() - t0).total_seconds() / 60
        print(f"{j['name']}: exit {rc} after {mins:.1f} min", flush=True)
        if rc != 0:
            write_status(js, current=None,
                         note=f"{j['name']} FAILED rc={rc}; see {log}")
            print(f"  tail of {log}:", flush=True)
            with open(log) as fh:
                for line in fh.readlines()[-15:]:
                    print("   " + line.rstrip(), flush=True)
    write_status(js, current=None, note="queue finished")
    print("\nqueue finished")


if __name__ == "__main__":
    main()
