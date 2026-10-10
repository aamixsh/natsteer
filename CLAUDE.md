# natsteer

Context for this project: read [PLAN.md](PLAN.md) first (current plan and the next todo), then
[docs/presentation.md](docs/presentation.md) (current results; small run reports are in `docs/results/`).
Setting up a new server (vast.ai, RunPod, lab): [docs/HANDOFF.md](docs/HANDOFF.md).

- Environment: `source env.sh` first (shared caches in `/workspace/cache`, compiler caches in `.cache/`), then run code with `.venv/bin/python`.
- Outputs go to `data/`, temp files to `scratch/` (not in git; override with `NATSTEER_DATA`, `NATSTEER_SCRATCH`).
- GPU jobs: pick a free GPU explicitly (`CUDA_VISIBLE_DEVICES=N` or `GPU=N`); another project may be using the others.
- Tests: `.venv/bin/python -m pytest -q tests`.
- PLAN.md is the one living plan: rewrite it to the current state, do not append history (history lives in the deck, `docs/results/` and git).
- Keep `docs/presentation.md` and `README.md` compact: update existing slides and sections in place with denser results; add a slide only for a genuinely new topic.
- The harmful-intent grading policy (intent track only) is written by the researchers, not generated.
