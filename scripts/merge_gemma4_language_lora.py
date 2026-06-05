#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device_map", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--max_memory_per_gpu", default="46GiB")
    parser.add_argument("--cpu_memory", default="700GiB")
    parser.add_argument("--max_shard_size", default="10GB")
    parser.add_argument("--safe_merge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device_map == "auto" and torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        max_memory = {idx: args.max_memory_per_gpu for idx in range(n_visible)}
        max_memory["cpu"] = args.cpu_memory
        device_map = "auto"
    else:
        max_memory = None
        device_map = {"": "cpu"}

    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    merged = model.merge_and_unload(safe_merge=args.safe_merge)
    if hasattr(merged.config, "use_cache"):
        merged.config.use_cache = True
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(output_dir)

    info = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "output_dir": str(output_dir),
        "dtype": "bfloat16",
        "device_map": args.device_map,
        "safe_merge": bool(args.safe_merge),
    }
    (output_dir / "merge_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
