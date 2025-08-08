# SGS-PZ: Sanguosha (Identity) PettingZoo AEC Environment

This repository provides a server-authoritative Sanguosha (Three Kingdoms Kill) rules engine exposed as a PettingZoo AEC environment, with FastAPI gateway and RLDS-compatible recorder.

Quickstart:

- Install: `pip install -e .[dev]`
- Test: `pytest`
- Run an episode via API: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`

Roadmap follows M0-M6 milestones described in the task.
