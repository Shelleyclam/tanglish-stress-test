# tanglish-stress-test

An open-source harness that stress-tests voice agents with Tamil-English
(Tanglish) callers: people who mix languages, pause mid-thought, and change
their mind mid-sentence.

It plays real caller recordings into an agent **in real time** (20 ms chunks,
full-duplex), records what the agent says and when, and reports where the
agent's design breaks.

## What it tests

| Hypothesis | Measured by |
|---|---|
| H1. A fast first-token model answers before the caller changes their mind | replies started before the caller finished; LLM judge + your label on final vs abandoned intent |
| H2. A 75 ms backchannel filler talks over callers who pause mid-thought | agent audio overlapping caller speech; fillers per minute; fillers in labelled pauses |
| H3. A fixed silence timer cuts off natural pauses | hang-ups before the caller finished; how long the timer actually ran |
| H4. Speech-to-text errors jump on code-switched speech | WER and **key details heard correctly**, split by caller language (needs English/Tamil control clips) |

Also: questions answered (judge, checked against your labels), invented numbers
in replies, reply language.

## Setup

Requires Python 3.11. Using [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"          # harness + tests
pytest                              # runs on synthetic audio, no models needed
```

For the local baseline agent you also need [Ollama](https://ollama.com) and:

```bash
uv pip install -e ".[local]"        # faster-whisper + Piper
ollama pull qwen2.5:1.5b && ollama pull qwen2.5:7b
python -m piper.download_voices --download-dir models en_US-lessac-medium
```

It needs about 6.6 GB of free RAM with all models loaded; `run.py` warns if
there is less.

## Your recordings

1. Put phone recordings (wav or m4a) in `clips/`, named after the scenario id
   (`clips/com_01.m4a`). `clips/` is git-ignored, so your voice stays local.
2. Describe each clip in `scenarios.yaml`: transcript, final / abandoned
   intent, mid-turn pauses, key details (with accepted spellings) and the
   questions you asked. The file's header explains every field.
3. Check VAD against your pause labels: `python scripts/vad_check.py`

## Running

```bash
python run.py --agent local                  # all scenarios, real time
python run.py --agent local --only com_01    # one clip
python run.py --rescore reports/local_<date>.json      # re-score saved calls (new metrics,
                                                       # edited scenarios.yaml) without re-running
python run.py --report-only reports/local_<date>.json  # rebuild report after labelling
```

Each run writes to `reports/`:

- `<agent>_<date>.md`: the report (headline, per-clip tables, pass rates, worst clips, caveats)
- `<agent>_<date>.json`: raw results
- `<agent>_<date>_labels.csv`: fill in `my_label` and `my_questions`; the report then shows
  how often the LLM judge agrees with you. **Don't trust the judge alone**: on the first
  run, small prompt wording changes flipped its verdicts.
- `<agent>_<date>_audio/`: stereo recordings, left = caller, right = agent

## Agents

- `LocalPipelineAgent`: faster-whisper → qwen2.5:1.5b (first sentence) → qwen2.5:7b
  (rest) → Piper, with a backchannel filler, early reply commit and silence timer
  mimicking Vaani's published design. All timings are settings in
  `LocalPipelineConfig`. On a CPU it is too slow to test H1 and H3; those wait for
  a fast agent.
- `VaaniAgent`: Vaani's hosted agent over WebRTC (`uv pip install -e ".[vaani]"`, then put
  `VAANI_API_KEY` and `VAANI_AGENT_ID` in `.env`, which is git-ignored). Each call costs
  Vaani credit, and the transcript only arrives after a few minutes of post-processing.
  Agent speech is cut into turns by loudness and matched to Vaani's transcript; short
  unlogged segments are *counted* as fillers, a guess from duration.

To add an agent, implement `AgentAdapter` in `tanglish/agents/base.py`.

## Built with

This project was built with [Claude Code](https://claude.com/claude-code).
