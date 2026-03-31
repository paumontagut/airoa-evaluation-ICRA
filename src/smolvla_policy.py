"""SmolVLA policy adapter for the ICRA evaluation WebSocket server.

Loads a SmolVLA checkpoint (model + preprocessor pipeline) and translates
between the evaluation framework's observation/action format and LeRobot's
internal representation.

Eval framework contract:
    Input:  {head_rgb: (480,640,3) uint8, hand_rgb: (480,640,3) uint8,
             state: (8,) float32, prompt: str}
    Output: {actions: (T, 11) float32}

Action order: arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll,
              gripper, head_pan, head_tilt, base_x, base_y, base_theta
"""

import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SmolVLAPolicy:
    """Wraps a SmolVLA checkpoint for the ICRA evaluation server."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA

        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # --- Load model ---
        logger.info("Loading SmolVLA model from %s", checkpoint_path)
        self.policy = LeRobotSmolVLA.from_pretrained(str(checkpoint_path))
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

        # --- Load tokenizer (same one used during training) ---
        logger.info("Loading tokenizer: HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        self.max_token_length = 48

        # --- Load normalizer stats from checkpoint ---
        pre_stats_path = checkpoint_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        post_stats_path = checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

        if pre_stats_path.exists():
            pre_stats = load_file(str(pre_stats_path))
            self.state_mean = pre_stats["observation.state.mean"].to(self.device)
            self.state_std = pre_stats["observation.state.std"].to(self.device)
            logger.info("Loaded state normalization stats (dim=%d)", self.state_mean.shape[0])
        else:
            raise FileNotFoundError(f"Preprocessor stats not found: {pre_stats_path}")

        if post_stats_path.exists():
            post_stats = load_file(str(post_stats_path))
            self.action_mean = post_stats["action.mean"].to(self.device)
            self.action_std = post_stats["action.std"].to(self.device)
            logger.info("Loaded action unnormalization stats (dim=%d)", self.action_mean.shape[0])
        else:
            raise FileNotFoundError(f"Postprocessor stats not found: {post_stats_path}")

        logger.info("SmolVLA ready on %s", self.device)

    @property
    def metadata(self) -> dict:
        return {"model": "smolvla", "type": "vla"}

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a task prompt, matching training pipeline settings."""
        # SmolVLA training appends newline to task text
        if not prompt.endswith("\n"):
            prompt = prompt + "\n"

        tokens = self.tokenizer(
            prompt,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            tokens["input_ids"].to(self.device),                    # (1, max_length)
            tokens["attention_mask"].bool().to(self.device),        # (1, max_length)
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """Normalize state vector using training dataset statistics (MEAN_STD)."""
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device)
        # Truncate or pad to match training state dimension
        stats_dim = self.state_mean.shape[0]
        if state_t.shape[0] > stats_dim:
            state_t = state_t[:stats_dim]
        elif state_t.shape[0] < stats_dim:
            state_t = torch.nn.functional.pad(state_t, (0, stats_dim - state_t.shape[0]))
        normalized = (state_t - self.state_mean) / (self.state_std + 1e-8)
        return normalized.unsqueeze(0)  # (1, state_dim)

    def _unnormalize_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Unnormalize predicted actions back to real scale."""
        # actions shape: (n_action_steps, action_dim) or (batch, n_action_steps, action_dim)
        return (actions * self.action_std + self.action_mean).cpu().numpy()

    def infer(self, obs: dict) -> dict:
        """Run inference on a single observation from the eval framework.

        Args:
            obs: dict with keys head_rgb, hand_rgb, state, prompt

        Returns:
            dict with key 'actions': np.ndarray (T, 11) float32
        """
        head_rgb = obs["head_rgb"]     # (480, 640, 3) uint8
        hand_rgb = obs["hand_rgb"]     # (480, 640, 3) uint8
        state = obs["state"]           # (8,) float32
        prompt = obs.get("prompt", "")

        # --- Images: (H,W,C) uint8 → (1,C,H,W) float32 [0,1] ---
        head_tensor = (
            torch.from_numpy(head_rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        )
        hand_tensor = (
            torch.from_numpy(hand_rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        )

        # --- State: normalize with training stats ---
        state_tensor = self._normalize_state(state)

        # --- Language: tokenize prompt ---
        lang_tokens, lang_mask = self._tokenize(prompt)

        # --- Build batch dict matching SmolVLA's expected keys ---
        # Keys after the rename step: observation.images.camera1/camera2
        # The model's prepare_images() handles resize+padding and [0,1]→[-1,1]
        batch = {
            "observation.images.camera1": head_tensor,
            "observation.images.camera2": hand_tensor,
            "observation.state": state_tensor,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask,
        }

        # --- Forward pass ---
        # predict_action_chunk returns (batch, n_action_steps, action_dim)
        # Unpadding from max_action_dim=32 → 11 happens inside _get_action_chunk.
        with torch.no_grad():
            actions_chunk = self.policy.predict_action_chunk(batch)

        # Remove batch dimension: (1, T, 11) → (T, 11)
        actions = actions_chunk.squeeze(0)

        # --- Unnormalize actions ---
        actions = self._unnormalize_actions(actions)

        return {"actions": actions.astype(np.float32)}

    def reset(self) -> None:
        """Reset the policy's internal action queue."""
        self.policy.reset()
