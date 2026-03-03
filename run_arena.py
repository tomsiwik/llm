"""CLI entry point for the micro arena."""

import argparse
import sys

# Ensure micro package is importable
sys.path.insert(0, ".")

from micro.models import list_models
from micro.arena import run_single, run_multidomain, run_comparison, leaderboard, ModelTree


def main():
    parser = argparse.ArgumentParser(description="micro/ MLX Arena")
    parser.add_argument("--model", type=str, help="Single model to train")
    parser.add_argument("--compare", type=str, help="Comma-separated model names to compare")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multidomain"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tree", action="store_true", help="Show model lineage tree")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard from tree")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if args.list:
        print("Available models:", ", ".join(list_models()))
        return

    if args.tree:
        tree = ModelTree()
        print(tree.show_tree())
        return

    if args.leaderboard:
        tree = ModelTree()
        # Show best run per model
        results = []
        from micro.metrics import RunMetrics
        for name, info in tree.data.get("models", {}).items():
            runs = info.get("runs", [])
            if runs:
                best = min(runs, key=lambda r: r.get("val_loss") or r.get("final_loss", 999))
                results.append(RunMetrics(**{k: v for k, v in best.items() if k != "parent" and k != "losses"}))
        if results:
            print(leaderboard(results))
        else:
            print("No results yet. Run --compare first.")
        return

    tree = ModelTree()
    train_kwargs = dict(steps=args.steps, batch_size=args.batch_size, lr=args.lr, log_every=args.log_every)

    if args.compare:
        names = [n.strip() for n in args.compare.split(",")]
        if args.mode == "multidomain":
            train_kwargs["steps_per_domain"] = train_kwargs.pop("steps")
        results = run_comparison(names, mode=args.mode, **train_kwargs)
        for r in results:
            tree.record(r)
        print(f"\n{'='*75}")
        print(leaderboard(results))
    elif args.model:
        if args.mode == "multidomain":
            r = run_multidomain(args.model, steps_per_domain=args.steps, batch_size=args.batch_size,
                                lr=args.lr, log_every=args.log_every)
        else:
            r = run_single(args.model, **train_kwargs)
        tree.record(r)
        vl = f"{r.val_loss:.4f}" if r.val_loss is not None else "N/A"
        print(f"\nResult: {r.model_name} | val_loss={vl} | params={r.param_count:,}")
        if r.forgetting:
            for d, f in r.forgetting.items():
                print(f"  forgetting/{d}: {f['forgetting']:+.3f} ({f['pct']:+.1f}%)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
