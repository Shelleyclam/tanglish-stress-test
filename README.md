# tanglish-stress-test

## The problem

Many callers in Tamil Nadu don't speak "English" or "Tamil" on the phone. They speak
Tanglish: *"Saturday room kedaikuma for 2 people… no no, not Saturday, Friday"*. They
switch language mid-sentence, stop to check a date, and change their mind halfway
through. Voice agents are usually demoed on clean, single-language speech. This
harness checks what happens to the details a booking depends on (name, date,
number of guests, place, time) when the caller talks the way people actually talk.

It plays real caller recordings into an agent **in real time** (20 ms audio frames,
full-duplex, like a phone line), records what the agent says and when, and scores
the call.

## Headline metric: key details kept

A key detail is **kept** when the agent both **heard** it correctly *and* **said it
back** correctly in its reply. Hearing "Raghu" and replying "Hello Rahul!" loses the
detail, and so does hearing "4 people" and never repeating it. Each scenario lists
its key details with accepted spellings, including Tamil script ("Friday" as ஃப்ரைடே).

## Results

Four recorded calls (one speaker), one run per agent, same caller audio, same hotel
facts in the system prompt.

| | Local baseline | Vaani (hosted) |
|---|---|---|
| **Key details kept** (heard and said back) | **7/12 (58%)** | **8/12 (67%)** |
| Key details heard correctly | 9/12 (75%) | 11/12 (92%) |
| Caller's questions answered (my hand labels) | 5/8 | 7/8 |
| Reply starts, after the caller finished | 7.1–9.9 s | 4.1–7.1 s |
| Fillers per minute (median) | 23.3 | 0 |
| Clips where the agent talked over the caller (> 0.2 s) | 2 of 4 | 0 of 4 |
| Invented numbers in replies | 0 | 0 |
| Hung up before the caller finished | 0 of 4 | 0 of 4 |

- **Local baseline** (run of 2026-10-27): faster-whisper → qwen2.5 (1.5b for the
  first sentence, 7b for the rest) → Piper, on a CPU-only laptop. It copies Vaani's published
  low-latency design: a backchannel "Mm-hmm" after 75 ms of silence, an early reply
  commit, and a silence timer. Its latencies reflect this laptop, not the design.
- **Vaani** (run of 2026-11-09): Vaani's hosted voice agent,
  reached over WebRTC with its Python SDK and given the same hotel facts as a per-call prompt.

### What happened, per clip

- **cs_01, code-switching, 5 key details.** Vaani's speech-to-text heard
  the name right (ராகு, "Raghu"), and the reply said **"Hello Rahul!"**. The baseline's
  Whisper heard "Rahul" from the start. Neither kept the place: the baseline heard
  "Alegiri", Vaani "LED".
  Of the caller's 5 questions (rooms for 4, early check-in allowed, its cost, tariff,
  a quote), Vaani answered 4 of 5 (not the quote); the baseline 2.
- **com_01, change of mind (Saturday → Friday).** Both answered for Friday, the caller's
  final choice, not the abandoned Saturday. Only Vaani's reply repeated "2 people".
- **pause_01, thinking pause.** Both heard "4 people"; only the baseline said it back.
- **silence_01, a 5.8 s pause to check dates.** Vaani said **"Hello?"** 1 s
  into the pause: a check-in, not an answer. The harness counts it as a reply that
  started before the caller had finished. It didn't hang up and answered correctly
  once the caller came back. In the same clip the baseline played 8 fillers, 6 of
  them inside the caller's pauses.
- **Vaani's replies took 4–7 s, with no filler while it worked.** That
  means 4–7 s of silence on the line after the caller stops. The baseline fills that
  gap with "Mm-hmm", 21–30 times a minute, and talks over the caller doing it.

### Listening notes (qualitative, my own ears)

From listening to the stereo recordings, not measured:

- The baseline's "Mm-hmm" filler sounds more like an "nnn" noise in the gaps than a
  backchannel.
- The baseline's replies come later and say less than Vaani's.
- The baseline speaks with a US English voice (Piper `en_US-lessac-medium`). Vaani's
  voice sounded Indian English, with a Tamil greeting, closer to how the caller speaks.

The baseline is a simple local pipeline on a CPU, built to mimic Vaani's published
design so the harness had something to run against. It is not a competing product,
and these notes compare against it only as a reference point.

## Limits

- **4 clips, 1 speaker, 1 run per agent.** These are observations, not rates. A
  different voice or a second run could change any single number above.
- **WER is not compared.** Vaani writes Tamil words in Tamil script and
  the reference transcripts are romanised, so word error rate would count every Tamil
  word as wrong. Key details are checked with spellings in both scripts instead.
- **"Kept" counts a detail the agent didn't repeat as lost**, even when the reply wasn't
  wrong. Of Vaani's 4 lost details, 1 was said wrong (the name) and 3 were
  not repeated (place in cs_01, place in silence_01, guests in pause_01).
- **The baseline runs on a CPU.** It is too slow to test whether a fast model answers
  before the caller changes their mind, or whether a silence timer cuts off pauses.
- **Fillers on Vaani are inferred.** It sends audio, not labelled events,
  so a short burst of speech with no line in its transcript counts as a filler. This
  run had none.
- **Reply language is a setting**: the prompt asks for English replies.
- **Questions answered is my own judgement**, one listener reading each final reply.
  The built-in LLM judge proved unstable (small prompt changes flipped verdicts), so
  its verdicts aren't used here.

## Try it with the sample clip

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
pytest                                  # synthetic audio, no models needed
```

The local baseline also needs [Ollama](https://ollama.com) and about 6.6 GB of free RAM:

```bash
uv pip install -e ".[local]"
ollama pull qwen2.5:1.5b && ollama pull qwen2.5:7b
python -m piper.download_voices --download-dir models en_US-lessac-medium

python run.py --agent local --scenarios scenarios.sample.yaml --no-judge
```

`clips/sample_synthetic.wav` is a voice-like buzz with a mid-turn pause (regenerate it
with `python scripts/make_sample_clip.py`). It has no words, so it checks the pipeline:
streaming, pause detection, fillers, timing and the report. Understanding scores mean
nothing on it.

## Testing your own calls

1. Record calls (wav or m4a) and put them in `clips/`, named after the scenario id
   (`clips/com_01.m4a`). `clips/` is git-ignored, so your voice stays on your machine.
2. Describe each call in `scenarios.yaml`: transcript, final and abandoned intent,
   mid-turn pauses, key details with accepted spellings, and the questions asked. The
   file's header explains every field.
3. Check pause detection against your labels: `python scripts/vad_check.py`
4. Run, then re-score or rebuild the report without calling the agent again:

```bash
python run.py --agent local                              # all scenarios, real time
python run.py --agent local --only com_01                # one clip
python run.py --rescore reports/local_<date>.json        # new metrics or edited scenarios
python run.py --report-only reports/local_<date>.json    # after filling in labels
```

Each run writes to `reports/` (git-ignored, since reports contain transcripts):
a Markdown report, the raw JSON, a labels CSV where you add your own intent and
question labels, and stereo recordings (left = caller, right = agent).

To run the Vaani adapter you need your own Vaani account and agent:

```bash
uv pip install -e ".[vaani]"
# .env (git-ignored):  VAANI_API_KEY=...  VAANI_AGENT_ID=...
python run.py --agent vaani --only com_01
```

Each call uses Vaani credit. The hotel prompt and greeting are sent per call; the
agent saved in your dashboard is not changed.

To test another agent, implement `AgentAdapter` in `tanglish/agents/base.py`: it receives
20 ms of caller audio at a time and returns what the agent said, with timestamps.
The hotel facts every agent gets are in `facts/resort.yaml`.

## Next steps

- English-only and Tamil-only control calls, to see whether code-switching itself is
  what loses details.
- More speakers, and repeat runs, to turn observations into rates.
- A transliteration step so WER can compare Tamil script with romanised text.
- A fast enough baseline (GPU) to test early replies and silence timers directly.

## License

MIT, see [LICENSE](LICENSE).

## Built with

This project was built with [Claude Code](https://claude.com/claude-code).
