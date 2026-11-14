"""Command line entry point: ``strumnet <command> ...``."""

import argparse

import numpy as np


def cmd_generate(args):
    from .baseline import predict_rows
    from .dataset import DIFFICULTIES, NOISE_LEVELS, build, save_csv
    from .tuning import CLASS_INDEX

    rows = build(args.per_group, args.seed)
    save_csv(rows, args.out)
    n = len(rows)
    print(f"wrote {n} examples to {args.out}")
    for split in ("train", "val", "test"):
        k = sum(r["split"] == split for r in rows)
        print(f"  {split:5s} {k:5d} ({k / n:.0%})")

    hits = predict_rows(rows) == np.array([CLASS_INDEX[r["label"]] for r in rows])
    print(f"\npeak-picking baseline: {hits.mean():.2%}")
    for flag in DIFFICULTIES:
        m = np.array([r[flag] for r in rows]) == 1
        print(f"  {flag:16s} {m.mean():6.1%} of examples, baseline {hits[m].mean():.2%}")
    clean = ~np.array([any(r[f] for f in DIFFICULTIES) for r in rows])
    print(f"  {'clean':16s} {clean.mean():6.1%} of examples, baseline {hits[clean].mean():.2%}")
    noise = np.array([r["noise_level"] for r in rows])
    by_noise = ", ".join(f"{lvl}: {hits[noise == lvl].mean():.1%}" for lvl in NOISE_LEVELS)
    print(f"  by noise level   {by_noise}")


def cmd_train(args):
    from sklearn.preprocessing import StandardScaler

    from . import model as mlp
    from .dataset import load_csv
    from .features import FEATURES
    from .tuning import CLASS_INDEX

    df = load_csv(args.data)
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    scaler = StandardScaler().fit(parts["train"][list(FEATURES)])
    X = {s: scaler.transform(d[list(FEATURES)]) for s, d in parts.items()}
    y = {s: d["label"].map(CLASS_INDEX).to_numpy() for s, d in parts.items()}

    model, history = mlp.fit(
        X["train"],
        y["train"],
        X["val"],
        y["val"],
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        lr=args.lr,
        two_heads=args.two_heads,
        epochs=args.epochs,
        seed=args.seed,
    )
    epochs = len(history.history["loss"])
    for split in ("train", "test"):
        acc = np.mean(mlp.predict_classes(model, X[split]) == y[split])
        print(f"{split:5s} accuracy {acc:.2%}")
    print(f"stopped after {epochs} epochs, {model.count_params()} parameters")

    if args.out:
        mlp.save(model, scaler, FEATURES, args.out)
        print(f"saved to {args.out}")


def build_parser():
    p = argparse.ArgumentParser(prog="strumnet", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate the synthetic dataset")
    g.add_argument("--out", default="data/guitar_notes.csv")
    g.add_argument("--per-group", type=int, default=60, help="examples per (class, noise level)")
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train an MLP on a generated CSV")
    t.add_argument("--data", default="data/guitar_notes.csv")
    t.add_argument("--hidden", type=int, nargs="+", default=[16, 8])
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--two-heads", action="store_true", help="separate string and state outputs")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", help="directory to save the model to")
    t.set_defaults(func=cmd_train)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
