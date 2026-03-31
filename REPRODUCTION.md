# Reproduction Steps — Team V4R-VLA (Team 35)

## Model

SmolVLA (500M params) fine-tuned on AIRoA MoMa dataset with top-up training on task6911.

Checkpoint location: `s3://airoa-icra-team-35/smolvla_topup_3k/`

## Setup

### 1. Download checkpoint

```bash
export AWS_ACCESS_KEY_ID=406d99feb9d2619d89d4730d850fc380
export AWS_SECRET_ACCESS_KEY=f6eaef1720fc598b01fc46cba15f89c3dad636abc24548bcccda6455f2846a72
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
export AWS_REGION=auto

mkdir -p checkpoints/smolvla_topup_3k
aws s3 sync s3://airoa-icra-team-35/smolvla_topup_3k/ checkpoints/smolvla_topup_3k/
```

### 2. Set environment variables

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/smolvla_topup_3k
export POLICY_SERVER_HOST=127.0.0.1
export POLICY_SERVER_PORT=8000
export POLICY_PYTORCH_DEVICE=cuda

# Robot network (adjust to your setup)
export HSR_IP=100.119.167.94
export ROS_MASTER_URI=http://100.119.167.94:11311
export ROS_IP=<YOUR_HOST_IP>
```

### 3. Start containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

### 4. Verify policy server is running

```bash
./RUN-DOCKER-CONTAINER.sh logs policy_server
# Should see: "SmolVLA ready on cuda" and "WebSocket server listening on 0.0.0.0:8000"
```

### 5. Run evaluation

```bash
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch test_mode:=false
```

## Test mode (synthetic observations)

```bash
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch
```

## Notes

- The policy server requires a GPU with at least 4GB VRAM.
- The tokenizer (SmolVLM2-500M-Video-Instruct) is pre-downloaded in the Docker image.
- Action output: 11-DoF (arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper, head_pan, head_tilt, base_x, base_y, base_theta).
