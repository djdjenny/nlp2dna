import os
import json
import argparse
from pathlib import Path
import subprocess
import sys

MODELS_TO_EVALUATE = [
    {
        "name": "dnabert_nucleotide",
        "extractor": "DNABertExtractor",
        "module": "dnabert",
        "model_path": None,
        "kwargs": {"mode": "nucleotide"},
    },
    {
        "name": "dnabert_3mer",
        "extractor": "DNABertExtractor",
        "module": "dnabert",
        "model_path": None,
        "kwargs": {"mode": "3mer"},
    },
    {
        "name": "dnabert_6mer",
        "extractor": "DNABertExtractor",
        "module": "dnabert",
        "model_path": None,
        "kwargs": {"mode": "6mer"},
    },
    {
        "name": "dnabert_3mer_nonoverlap",
        "extractor": "DNABertExtractor",
        "module": "dnabert",
        "model_path": None,
        "kwargs": {"mode": "3mer_no"},
    },
    {
        "name": "dnabert_6mer_nonoverlap",
        "extractor": "DNABertExtractor",
        "module": "dnabert",
        "model_path": None,
        "kwargs": {"mode": "6mer_no"},
    },
    {
        "name": "dnabert_bpe",
        "extractor": "DNABertBPEExtractor",
        "module": "dnabert",
        "model_path": None,
        "kwargs": {},
    }
]


def find_model_weights(model_dir, pattern):
    model_dir = Path(model_dir)
    for file in model_dir.glob("*.pt"):
        if pattern in file.name and "best" in file.name:
            return str(file)
    return None


def run_geneb_evaluation(
    data_dir,
    output_dir,
    extractor,
    module,
    model_id,
    model_path=None,
    mode=None,
    device="cuda",
    batch_size=8,
    limit=None,
):
    """Run single GENEB evaluation via harness/run_GENEB.py"""

    cmd = [
        sys.executable,
        "harness/run_GENEB.py",
        "--extractor",
        extractor,
        "--module",
        module,
        "--name_model",
        model_id,
        "--model_id",
        model_id,
        "--display",
        model_id,
        "--params",
        "0",
        "--data_dir",
        data_dir,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    if model_path:
        env["DNABERT_MODEL_PATH"] = str(model_path)
    if mode:
        env["DNABERT_MODE"] = mode

    print(f"Running: {' '.join(cmd)}")
    if model_path:
        print(f"Model path: {model_path}")
    if mode:
        print(f"Mode: {mode}")
    result = subprocess.run(cmd, env=env, text=True)

    if result.returncode != 0:
        print(f"Evaluation process exited with code {result.returncode}", flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run GENEB evaluation on multiple DNABERT models and NT-v3")
    parser.add_argument("--data-dir", required=True, help="Path to data directory with task CSVs")
    parser.add_argument("--model-dir", default=None, help="Path to directory with trained model weights")
    parser.add_argument("--model-path", default=None, help="Direct path to specific model weights (.pt file)")
    parser.add_argument("--output-dir", default="./submissions", help="Output directory for results (default: ./submissions)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding extraction")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N tasks")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to run (e.g., dnabert_nucleotide dnabert_3mer nt_v3). If not set, runs all.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mode_patterns = {
        "dnabert_nucleotide": "_nucl_",
        "dnabert_3mer": "_3mer_",
        "dnabert_6mer": "_6mer_",
        "dnabert_3mer_nonoverlap": "_3mer_no_",
        "dnabert_6mer_nonoverlap": "_6mer_no_",
        "dnabert_bpe": "_bpe_",
    }
    models_to_run = MODELS_TO_EVALUATE

    if args.models:
        models_to_run = [m for m in MODELS_TO_EVALUATE if m["name"] in args.models]

    results_summary = {}

    for model_config in models_to_run:
        model_name = model_config["name"]
        extractor = model_config["extractor"]
        module = model_config["module"]
        mode = model_config["kwargs"].get("mode")  # Extract mode from config

        model_path = None
        if model_name.startswith("dnabert"):
            if args.model_path:
                model_path = args.model_path
            elif args.model_dir:
                pattern = mode_patterns.get(model_name)
                model_path = find_model_weights(args.model_dir, pattern)
                if not model_path:
                    continue
            else:
                continue

        success = run_geneb_evaluation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            extractor=extractor,
            module=module,
            model_id=model_name,
            model_path=model_path,
            mode=mode,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        results_summary[model_name] = {"success": success, "model_path": model_path}

    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
