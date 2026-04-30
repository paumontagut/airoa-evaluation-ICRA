# REPRODUCTION_STEPS — Team 35 (V4R Lab, TU Wien)

This file is the submission note for Team 35. The evaluator should follow the steps below literally. All paths, commands, and environment variable names are exactly what we used for the local smoke test.

---

## 1. Overview

| Item | Value |
|---|---|
| Model summary | π0.5 (3.3B) supervised fine-tune on `task6911` (HSR relocation tasks), 3000 training steps |
| Framework | OpenPI (this branch is `sample-openpi`, no harness changes) |
| Repository | `https://github.com/paumontagut/airoa-evaluation-ICRA` |
| Branch | `sample-openpi` |
| Commit hash | `47b4df568907c12c4a4c9758644dacfc5778c6d5` |
| Checkpoint S3 path | `s3://airoa-icra-team-35/pi05_expert_only_3000/` |
| Expected VRAM | ~9 GB (loads under 16 GB easily) |

---

## 2. Prerequisites

- NVIDIA GPU with ≥ 16 GB VRAM (Blackwell-compatible).
- Docker Engine + Docker Compose v2.
- NVIDIA Container Toolkit.
- R2 (S3-compatible) credentials for `s3://airoa-icra-team-35/`. The team-internal credentials are bundled with this submission email; please use them only to download the checkpoint.
- No `HF_TOKEN` required: all weights are inside the checkpoint directory and the OpenPI image already contains the PaliGemma tokenizer.

---

## 3. Reproduction Steps

### 3.1 Clone and checkout

```bash
git clone --branch sample-openpi --single-branch \
  https://github.com/paumontagut/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git rev-parse HEAD   # should print 47b4df568907c12c4a4c9758644dacfc5778c6d5
```

### 3.2 Download checkpoint from R2

```bash
export AWS_ACCESS_KEY_ID=<team-35 access key>
export AWS_SECRET_ACCESS_KEY=<team-35 secret>
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
export AWS_REGION=auto

mkdir -p ./checkpoints/pi05_expert_only_3000
aws s3 sync s3://airoa-icra-team-35/pi05_expert_only_3000/ \
            ./checkpoints/pi05_expert_only_3000/
```

The directory must contain `params/`, `assets/`, and `_CHECKPOINT_METADATA` (see §6). Total download is ~5.7 GiB.

### 3.3 Environment variables

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/pi05_expert_only_3000
export POLICY_CONFIG_NAME=pi05_hsr_task6891011_level12_v2.5_train_adaptive
```

`POLICY_PYTORCH_DEVICE` is not used (this is a JAX/OpenPI policy).

### 3.4 Start containers

```bash
sudo -E env "POLICY_CHECKPOINT_PATH=${POLICY_CHECKPOINT_PATH}" \
            "POLICY_CONFIG_NAME=${POLICY_CONFIG_NAME}" \
  ./RUN-DOCKER-CONTAINER.sh up
```

`sudo -E` is only needed if Docker is not in your user group; preserves the two `POLICY_*` variables either way.

### 3.5 Verify

```bash
sudo ./RUN-DOCKER-CONTAINER.sh logs policy_server
# Expect a line of the form:
#   Serving policy config=pi05_hsr_task6891011_level12_v2.5_train_adaptive
#   checkpoint=/policy_checkpoint
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

End-to-end synthetic test (the same `test_mode:=true` we ran locally):

```bash
sudo docker exec -it airoa_hsr_client bash -lc \
  "source /opt/ros/noetic/setup.bash && \
   source /root/catkin_ws/devel/setup.bash && \
   roslaunch hsr_policy_client hsr_policy_client.launch test_mode:=true"
```

### 3.6 Stop

```bash
sudo -E env "POLICY_CHECKPOINT_PATH=${POLICY_CHECKPOINT_PATH}" \
            "POLICY_CONFIG_NAME=${POLICY_CONFIG_NAME}" \
  ./RUN-DOCKER-CONTAINER.sh down
```

---

## 4. Files modified relative to the base repo

None on this branch. Our submission uses the upstream `sample-openpi` harness as-is. The customisation lives entirely in the checkpoint and the `POLICY_CONFIG_NAME` we pass in.

| Path | Reason |
|---|---|
| (none) | — |

---

## 5. Important notes

- First startup loads ~3.3 B parameters from the orbax checkpoint and takes 60–90 s before the server is ready. Subsequent inferences are fast.
- The config name `pi05_hsr_task6891011_level12_v2.5_train_adaptive` is registered in `src/openpi/training/config.py` of the upstream `sample-openpi` branch. The checkpoint was trained with it, so loading uses the same dataset stats (`task6891011_level12_v2.5_train`). The asset id inside the checkpoint is `lerobot_datasets/task6891011_level12_v2.5_train`.
- Action space is HSR 11-DoF joint space (5 arm + 1 gripper + 2 head + 3 base). Observations: `head_rgb` (480×640), `hand_rgb` (480×640), 8-D state.
- We confirmed locally that the server starts, the action chunk has finite values, and `test_mode:=true` runs without `shape mismatch` or `sanitized N NaN` warnings.

---

## 6. Checkpoint file layout

OpenPI orbax-style:

```
pi05_expert_only_3000/
├── _CHECKPOINT_METADATA
├── assets/
│   └── lerobot_datasets/
│       └── task6891011_level12_v2.5_train/
│           └── norm_stats.json
└── params/
    ├── _METADATA
    ├── _sharding
    ├── array_metadatas/process_0
    ├── d/...
    ├── manifest.ocdbt
    └── ocdbt.process_0/
        ├── d/...
        └── manifest.ocdbt
```

Total size on disk: ~5.7 GiB, 23 files.

---

## 7. Smoke test expected output

`policy_server` log on a successful boot (abridged):

```
[INFO] Serving policy config=pi05_hsr_task6891011_level12_v2.5_train_adaptive
       checkpoint=/policy_checkpoint
[INFO] policy_server listening on 0.0.0.0:8000
[INFO] Action executed.
[INFO] Action executed.
```

`hsr_policy_client` with `test_mode:=true` produces 100+ frames of `Action executed.` lines without `NaN` or `shape mismatch`.

---

## 8. Contact

- Team: Team 35 — V4R Lab, TU Wien
- Representative: Pau Montagut Bofi (please reply to this email thread)
- Submission date: 2026-04-30
