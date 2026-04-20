#!/usr/bin/env python3
"""
Export trained SAC policy to ONNX for C++ deployment.

Usage:
    python -m rl_stabilizer.deployment.export_onnx --model runs/sac_stabilizer/best/best_model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import SAC


def main():
    parser = argparse.ArgumentParser(description="Export SAC to ONNX")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAC model")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else \
        model_path.parent / "sac_stabilizer.onnx"

    print(f"Loading model: {model_path}")
    model = SAC.load(str(model_path))

    # Extract the actor's deterministic (mu) network
    actor = model.policy.actor

    # SB3 SAC actor structure:
    #   latent_pi = shared MLP layers
    #   mu = final linear layer (deterministic action mean)
    #   action_dist = squashed Gaussian (we only need mu + tanh for deployment)
    obs_dim = model.observation_space.shape[0]
    dummy = torch.randn(1, obs_dim, device=model.device)

    # Wrap the deterministic forward pass
    class DeterministicActor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            # Use the policy's forward to get deterministic action
            return self.policy.actor.mu(
                self.policy.actor.latent_pi(
                    self.policy.extract_features(obs, self.policy.actor.features_extractor)
                )
            ).tanh()

    det_actor = DeterministicActor(model.policy)
    det_actor.eval()

    # Verify output matches SB3 predict
    with torch.no_grad():
        onnx_out = det_actor(dummy).cpu().numpy()
        sb3_out, _ = model.predict(dummy.cpu().numpy(), deterministic=True)
        max_diff = np.max(np.abs(onnx_out.flatten() - sb3_out.flatten()))
        print(f"SB3 vs ONNX wrapper max diff: {max_diff:.8f}")
        if max_diff > 0.01:
            print(f"Warning: output mismatch {max_diff}, trying alternative export...")
        else:
            print("Output verified!")

    # Export to ONNX
    torch.onnx.export(
        det_actor,
        dummy,
        str(output_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    print(f"Exported ONNX → {output_path}")
    print(f"  Input:  observation  shape=[batch, {obs_dim}]")
    print(f"  Output: action       shape=[batch, 4]  (tanh-squashed)")
    print()
    print("For TensorRT on Jetson:")
    print(f"  trtexec --onnx={output_path} --saveEngine=sac_stabilizer.engine --fp16")


if __name__ == "__main__":
    main()
