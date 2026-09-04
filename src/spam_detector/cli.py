"""Command line interface: `spam-detector train|predict|serve`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spam_detector.model import CLASSIFIERS
from spam_detector.predict import DEFAULT_MODEL_PATH, SpamDetector
from spam_detector.train import DEFAULT_METRICS_PATH, format_summary, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spam-detector", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and evaluate a model")
    train_parser.add_argument(
        "--dataset",
        default="sms-spam",
        help="'sms-spam' (downloaded), 'sample' (bundled) or a path to a labelled CSV",
    )
    train_parser.add_argument("--classifier", default="linear-svm", choices=list(CLASSIFIERS))
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--cv-folds", type=int, default=5)
    train_parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="tune the decision threshold to reach this spam precision (e.g. 0.99)",
    )
    train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="classify messages")
    predict_parser.add_argument("text", nargs="*", help="messages to classify")
    predict_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument(
        "--file", type=Path, help="read messages from a file, one message per line"
    )
    predict_parser.add_argument("--json", action="store_true", help="emit JSON instead of text")

    serve_parser = subparsers.add_parser("serve", help="run the REST API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def _collect_messages(args) -> list[str]:
    messages = list(args.text)
    if args.file:
        messages.extend(line.strip() for line in args.file.read_text().splitlines() if line.strip())
    if not messages and not sys.stdin.isatty():
        messages.extend(line.strip() for line in sys.stdin.read().splitlines() if line.strip())
    return messages


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        try:
            result = train(
                dataset=args.dataset,
                classifier=args.classifier,
                model_path=args.model_path,
                metrics_path=args.metrics_path,
                test_size=args.test_size,
                target_precision=args.target_precision,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(format_summary(result))
        return 0

    if args.command == "predict":
        messages = _collect_messages(args)
        if not messages:
            print("No messages given. Pass text arguments, --file or pipe stdin.", file=sys.stderr)
            return 2
        try:
            predictions = SpamDetector.load(args.model_path).predict(messages)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "text": p.text,
                            "label": p.label,
                            "spam_probability": round(p.spam_probability, 4),
                        }
                        for p in predictions
                    ],
                    indent=2,
                )
            )
        else:
            for prediction in predictions:
                print(
                    f"[{prediction.label:>4}] {prediction.spam_probability:.4f}  {prediction.text}"
                )
        return 0

    if args.command == "serve":
        import uvicorn

        from spam_detector.api import create_app

        uvicorn.run(create_app(args.model_path), host=args.host, port=args.port)
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
