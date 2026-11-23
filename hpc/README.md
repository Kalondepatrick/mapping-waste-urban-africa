# HPC — Sapitwa (SLURM)

Submission scripts for the cluster runs. The cluster checkout must mirror this
repository; `PROJ` at the top of each script points at it.

| file | job |
|---|---|
| `factorial/run_factorial.sbatch` + `jobs.txt` | 28-configuration pretraining-domain x label-budget factorial, into `results/04_label_budget_factorial/` |
| `factorial/run_negsweep.sbatch` + `negweights.txt` | not-waste loss weight sweep, into `results/05_negative_weight_sweep/` |
| `factorial/run_v6.sbatch` + `v6_jobs.txt` | RS-specific pretrained backbones, frozen vs fine-tuned, into `results/06_pretrained_backbones/` |
| `factorial/setup_env.sh` | builds `.venv-torch` and pre-stages checkpoints on the login node |
| `factorial/hn_subset_3000.txt`, `hn_rsync_files.txt` | hard-negative tile lists used to transfer that subset |

```bash
sbatch hpc/factorial/run_v6.sbatch
squeue -u $USER
```

## Two things that will bite you

1. **Load the Python module first.** `.venv-torch` was built against
   `Python/3.12.3-GCCcore-13.3.0`. Without it the venv interpreter dies with
   `libffi.so.8: cannot open shared object file`. Every sbatch here does
   `module purge; module load Python/3.12.3-GCCcore-13.3.0` before activating —
   that is required, not cosmetic.
2. **Compute nodes have no outbound network.** Pre-stage Hugging Face checkpoints
   on the login node, then export `HF_HOME="$PROJ/.hf_cache"` and
   `HF_HUB_OFFLINE=1`. The scripts already do this.

Array jobs are capped with `%4` to match the four A100 MIG slices
(`gpu:3g.40gb`, two per node on compute3/compute4). Every script skips a
configuration whose `metrics_all_seeds.csv` already exists, so resubmitting after
a timeout resumes rather than duplicating work.

## Transferring data

Large transfers over this link have failed silently before. Two real failure
modes, both fixed in the file lists here: Python writing CRLF line endings into
the file list, and GNU tar reading a Windows `C:/...` path as a remote host
("Cannot connect to C:"). Write file lists with LF endings and POSIX `/c/...`
paths, and **always verify the file count on the far side** — the first attempt
reported success while shipping an empty 45-byte archive.

## Note after the 2026-11-23 reorganisation

The paths inside these scripts were updated to the new layout
(`03_unet_pytorch/01_train_unet_v5_torch.py`, `results/0N_*`,
`data/pretrained_weights/openearthmap/`). The Sapitwa checkout still has the old
layout, so re-sync the repository there before the next submission, or the
scripts will not find the trainer.
