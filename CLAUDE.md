# natsteer

Context for this project: read [docs/HANDOFF.md](docs/HANDOFF.md) first, then [PLAN.md](PLAN.md).
Current results: [docs/presentation.md](docs/presentation.md). Moving machines: [docs/MIGRATION.md](docs/MIGRATION.md).

- Environment: `source env.sh` first (keeps caches, temp files and Python inside the project), then run code with `.venv/bin/python`.
- Outputs go to `data/`, temp files to `scratch/` (not in git; override with `NATSTEER_DATA`, `NATSTEER_SCRATCH`).
- Tests: `.venv/bin/python -m pytest -q tests`.
- The harmful-intent grading policy is written by the researchers, not generated.
