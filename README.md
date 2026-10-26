# tanglish-stress-test

An open-source harness that stress-tests voice agents with Tamil-English
(Tanglish) callers: people who mix languages, pause mid-thought, and change
their mind mid-sentence.

> Status: work in progress. Modules are being built one at a time.

## Setup

Requires Python 3.11. Using [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
pytest
```

ffmpeg is bundled through the `imageio-ffmpeg` package, so no system install is needed.

## Your recordings

Put phone recordings (wav or m4a) in `clips/`. They are converted to 16 kHz
mono on load. `clips/` is git-ignored, so your voice never ends up in the repo.

## Built with

This project was built with [Claude Code](https://claude.com/claude-code).
