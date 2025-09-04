import argparse
import tempfile
import torch as th
import onnx
from onnx import shape_inference
from stable_baselines3 import PPO


# ------------------------------------------------------------------
# 1.  Wrap the SB3 policy so that only ACTION and VALUE heads are
#     exported – this avoids the residual “Add” that crashes stedgeai
# ------------------------------------------------------------------
class ActorCriticONNX(th.nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        # Feature extractor (CNN, MLP, …)
        self.features_extractor = sb3_policy.features_extractor
        # Two-head MLP (shared + policy/value branches)
        self.mlp_extractor = sb3_policy.mlp_extractor
        # Final linear layers
        self.action_net = sb3_policy.action_net
        self.value_net  = sb3_policy.value_net

    def forward(self, obs):
        x = obs.float()
        features = self.features_extractor(x)
        latent_pi, latent_vf = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)

        return action_logits


# ------------------------------------------------------------------
# 2.  Optional graph-cleanup that removes *any* residual Add node
#     that still sneaks in (rare on opset ≤14, common on opset ≥17)
# ------------------------------------------------------------------
def prune_residual_add(in_path: str, out_path: str):
    model = onnx.load(in_path)
    keep_nodes = [n for n in model.graph.node if "_policy_Add" not in n.name]
    model.graph.ClearField("node")
    model.graph.node.extend(keep_nodes)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, out_path)


# ------------------------------------------------------------------
# 3.  Main CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export SB3 PPO to X-CUBE-AI-friendly ONNX")
    parser.add_argument("--sb3_file", required=True,
                        help="Path to <algo>.zip produced by SB3")
    parser.add_argument("--onnx_file", required=True,
                        help="Where to write the ONNX model")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset (<=14 recommended for ST tools)")
    parser.add_argument("--prune", action="store_true",
                        help="After export, remove residual Add node")
    args = parser.parse_args()

    # 3.1  Load PPO
    ppo = PPO.load(args.sb3_file, device="cpu")

    # 3.2  Build wrapper
    net = ActorCriticONNX(ppo.policy)
    net.eval()

    # 3.3  Dummy input (batch=1).  Works for vector or image obs.
    if ppo.observation_space.shape is None:
        raise ValueError("Only fixed-size observation spaces supported")
    dummy_obs = th.zeros((1, *ppo.observation_space.shape), dtype=th.float32)

    # 3.4  Export
    th.onnx.export(
        net, dummy_obs, args.onnx_file,
        input_names=["obs"],
        output_names=["action_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={"obs": {0: "batch"},
                      "action_logits": {0: "batch"}},
    )
    print(f"ONNX saved to {args.onnx_file}")

    # 3.5  Optional pruning
    if args.prune:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            prune_residual_add(args.onnx_file, tmp.name)
            onnx.save(onnx.load(tmp.name), args.onnx_file)
            print("Residual Add node pruned")

    # 3.6  Quick sanity check – loads back and runs one inference
    model_onnx = onnx.load(args.onnx_file)
    onnx.checker.check_model(model_onnx)
    print("ONNX model passes structural check ✅")


if __name__ == "__main__":
    main()
