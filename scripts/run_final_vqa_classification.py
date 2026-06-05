#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the unified VQA runner.

Use scripts/run_final_vqa.py for all VQA experiments. This wrapper exists only
so older commands fail less dangerously and still execute the score-visible VQA
prompt path.
"""

import sys
from pathlib import Path


def main() -> int:
    print(
        "[deprecated] scripts/run_final_vqa_classification.py is deprecated; "
        "forwarding to scripts/run_final_vqa.py.",
        file=sys.stderr,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_final_vqa import main as run_final_vqa_main

    run_final_vqa_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
