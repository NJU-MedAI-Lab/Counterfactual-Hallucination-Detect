"""Compatibility entry point for counterfactual batch evaluation."""

import sys

from src.eval import eval_batch


if __name__ == "__main__":
    if "--test_contradictory" not in sys.argv:
        sys.argv.append("--test_contradictory")
    eval_batch.main()
