"""Run a top-level script under a __main__ guard, for vLLM engines that start in spawn mode (Qwen3.5 models):

  spawn_safe.py scripts/loop_generate.py --out ...

A spawned engine process re-imports the main module; here that is this launcher, whose guard keeps the script from
running again in the child."""

import runpy
import sys

if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__script__")  # not "__main__": spawn must re-import this launcher, not the script
