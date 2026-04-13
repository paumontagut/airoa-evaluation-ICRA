# airoa-evaluation-ICRA

Participant evaluation runtime for ICRA 2026 VLA Workshop Competition.

**Team 35 — V4R-VLA (TU Wien)**
**Model: SmolVLA (450M parameters)**

## Quick Start (Evaluators)

```bash
# 1. Set checkpoint path (REQUIRED)
export POLICY_CHECKPOINT_PATH=/path/to/smolvla_topup_3k

# 2. Start both containers
./RUN-DOCKER-CONTAINER.sh up

# 3. Enter client shell
./RUN-DOCKER-CONTAINER.sh shell

# 4. Run evaluation (inside client container)
roslaunch hsr_policy_client hsr_policy_client.launch
```

The policy server starts automatically, loads the SmolVLA checkpoint, and listens on port 8000.
Look for `WebSocket server listening on 0.0.0.0:8000` in the policy_server logs to confirm startup.

### Download checkpoint from R2

```bash
export AWS_ACCESS_KEY_ID=406d99feb9d2619d89d4730d850fc380
export AWS_SECRET_ACCESS_KEY=f6eaef1720fc598b01fc46cba15f89c3dad636abc24548bcccda6455f2846a72
export AWS_REGION=auto

aws s3 cp s3://airoa-icra-team-35/smolvla_topup_3k/ ./checkpoints/smolvla_topup_3k/ \
  --recursive \
  --endpoint-url https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com

export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/smolvla_topup_3k
```

## 1. Editable Scope

Main implementation targets:

- `server/`
- `src/`

## 2. Host Requirements

- Linux
- Docker Engine
- Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit (`nvidia-container-runtime` must be available)

```bash
docker --version
docker compose version
nvidia-smi
```

## 3. Required Environment Variables

```bash
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
```

No other variables are required. The container handles GPU detection, tokenizer caching,
and model loading automatically. All model assets are pre-cached inside the Docker image.

## 4. Test Flow

Start containers:

```bash
./RUN-DOCKER-CONTAINER.sh up
```

Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

Run launch inside the container:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

By default, `test_mode` is `true`. In this mode, the client uses synthetic random observations
(`head_rgb`, `hand_rgb`, and `state`) in an infinite loop and prints language/action logs.

Expected output in policy_server logs:
```
SmolVLA ready on cuda
Serving SmolVLA checkpoint=/policy_checkpoint on 0.0.0.0:8000
WebSocket server listening on 0.0.0.0:8000
```

Expected output in hsr_client logs:
```
Action executed.
```

## 5. Health Check

The policy server exposes a health check endpoint:

```bash
curl http://localhost:8000/healthz
# Returns: OK
```

## 6. Logs and Stop

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh down
```

## 7. WebSocket I/O Contract

Inference request fields:

- `head_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `hand_rgb`: image array `(H, W, 3)` (current HSR dataset profile: `(480, 640, 3)`)
- `state`: `(8,)`
- `prompt`: `str`

Inference response field:

- `actions` with shape `(T, 11)`, `T >= 1`

Action order:

- `[arm_lift_joint, arm_flex_joint, arm_roll_joint, wrist_flex_joint, wrist_roll_joint, gripper, head_pan_joint, head_tilt_joint, base_x, base_y, base_t]`

Value requirement:

- finite numeric values only

## 8. Architecture

```
+---------------------+     WebSocket (port 8000)    +---------------------+
|   hsr_client (ROS)  | <-------------------------> |  policy_server (GPU) |
|   Reads sensors     |   obs → actions             |  SmolVLA inference   |
|   Executes actions  |                              |  450M params         |
+---------------------+                              +---------------------+
```

The policy server runs SmolVLA with:
- Pre-cached tokenizer (HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- TRANSFORMERS_OFFLINE=1 (no network access needed at runtime)
- Checkpoint mounted read-only at /policy_checkpoint
