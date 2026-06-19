"""Script para entrenar la RNN clásica sobre FatigueSet.

Usa la API de `fatigueset-lib` y la función `train_kfold` implementada en
`fatigueset_lib.fatigueset.rnn`.
"""
import argparse
import json
from pathlib import Path

from fatigueset import train_kfold


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='fatigueset')
    parser.add_argument('--window-size', type=int, default=64)
    parser.add_argument('--step', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='models/rnn')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = train_kfold(
        dataset_path=args.dataset_path,
        window_size=args.window_size,
        step=args.step,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_splits=args.n_splits,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    with open(Path(args.output_dir) / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f'Results saved to {args.output_dir}')


if __name__ == '__main__':
    cli()
