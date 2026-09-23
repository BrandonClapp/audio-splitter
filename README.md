# split-solo

Take a song, isolate the **lead guitar solo** over a time range you specify, and write two MP3s:
the solo, and the backing music behind it.

```bash
split-solo song.mp3 --from 2:14 --to 2:48
# -> song.solo.mp3      the lead
# -> song.backing.mp3   everything else, rhythm guitar included
```

## The one idea

Estimate the solo as well as possible, then define

```
backing = original - solo
```

Everything the solo estimate doesn't confidently claim — rhythm guitar, drums, bass, vocals,
keys, and the separation model's own error — lands in the backing automatically. Nothing is lost
or duplicated, and the two outputs sum back to the original sample-for-sample. Every run prints
that as a **null test**: `max|(solo + backing) - original|`, which is under `1e-6` before MP3
encoding or the run exits non-zero.

This is also why a weak solo estimate degrades gracefully. The backing stays complete and clean
no matter what; only the solo file carries more bleed.

## Honest expectations

`htdemucs_6s` is the one widely available model with a **guitar** source, and that stem contains
**all** the guitars — lead and rhythm — and it is the weakest of that model's six stems. No
off-the-shelf model separates lead from rhythm. Time-gating alone doesn't fix it either, because
rhythm guitar usually plays *during* the solo.

So the refinement masks below are heuristics. Expect a good-but-not-surgical result: a usable
solo track, and a backing track that's clean because it is defined as the exact complement.

## Install

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
split-solo --help
```

The bootstrap creates `.venv`, installs the package, and prefetches the `htdemucs_6s` weights
(a few hundred MB, once). You need `ffmpeg` on PATH. Everything after the weight download is
offline.

Verified on macOS 26.6 / Apple M4 Pro with Homebrew Python 3.14: demucs 4.1.0, torch 2.14.0,
librosa 1.0.0, numpy 2.5.3 — all prebuilt wheels, nothing compiles.

## How it works

1. **Decode** with `ffmpeg` to 44.1 kHz stereo float32, `(2, n)` channels-first throughout.
2. **Separate** with `htdemucs_6s` on MPS, falling back to CPU (automatically, including if MPS
   returns non-finite samples). Stems are **cached on disk** keyed by (audio, model, shifts,
   overlap).
3. **Time gate** the guitar stem to `[--from, --to]` with a raised-cosine fade, so the output is
   exactly zero outside the window and there's nothing to click on at the boundaries.
4. **Build the solo estimate** from the stems named by `--source` (default `guitar`), then
   **refine to lead only** with two independent masks (below).
5. `solo` = the refined signal; `backing` = `original - solo`.
6. **Level safety**: if either file would clip, the *same* gain is applied to both so their
   balance is preserved, targeting -0.3 dBFS true peak. `--no-normalize` opts out.
7. **Encode** both to MP3 (320 kbps CBR by default) with ID3 titles `<track> — guitar solo` and
   `<track> — backing`.

### The two masks

**Center/coherence mask** (on by default, `--center-strength 1.0`). Per STFT bin, score how
close the bin sits to a target pan position, times how phase-coherent it is between left and
right. This is what removes hard-panned and double-tracked rhythm guitars, which is how rhythm
parts are usually recorded. `--center-pan` retargets it if the solo itself sits off-center;
`--center-strength 0` disables it.

**Melody/harmonic mask** (off by default, `--melody-mask`). Track the lead pitch with
`librosa.pyin` and keep only bins near its harmonics, with a floor (`--harm-floor`) so pick
attack and distortion texture survive. Off by default because on heavily distorted guitar the
harmonic series is dense and this can thin the tone. Reach for it when rhythm guitar is *still*
audible after center masking.

## Tuning guide

The window and the knobs are meant to be found by ear. Use `--preview` — it encodes only the
window ±2 s, and the stem cache means only the very first run pays for separation; every round
after that takes a second or two.

```bash
split-solo song.mp3 --from 2:14 --to 2:48 --preview
```

### Which way to push

The two common complaints pull in **opposite** directions, so work out which one you have first:

- **"Rhythm guitar is still in my solo"** → make the solo estimate *claim less*: add
  `--melody-mask`, raise `--center-strength`, raise `--harmonics`.
- **"I can still hear the solo in my backing"** → make it *claim more*: add stems with
  `--source`, and lower `--center-strength`.

That follows directly from `backing = original - solo`. Anything a mask removes from the solo
doesn't vanish — it reappears in the backing. There is no setting that cleans up both at once;
you are choosing where the model's uncertainty goes.

| What you hear | Try |
| --- | --- |
| rhythm guitar still audible in the solo | add `--melody-mask`, then raise `--harmonics` |
| solo still audible in the backing | `--source guitar,other`, then `guitar,vocals,other`; and `--center-strength 0` |
| solo sounds thin, phasey, or "underwater" | lower `--center-strength 0.5`, or raise `--harm-floor 0.25` |
| the solo is panned, not centered | set `--center-pan` (-1 hard left … +1 hard right) |
| drums or keys bleeding into the solo | drop stems from `--source`, raise `--center-strength` |
| clicks, or an abrupt entry/exit | raise `--fade`, or adjust the window and `--pad` |
| general smearing and bleed in both files | `--shifts 5` (≈3× runtime, better separation) |

### `--source`: when the lead isn't only in the guitar stem

`htdemucs_6s` routes a distorted lead inconsistently — parts of it land in `other`, and,
on instrumental tracks, in `vocals`. Whatever it puts there is *not* in your solo file, so it
stays in the backing and you hear the solo bleeding through.

Run once with `--keep-stems` and check where the lead actually went. A quick tell: if the
pitch contour of the `vocals` stem tracks the `guitar` stem, that stem is carrying lead, not
voice. Then widen the estimate:

```bash
split-solo song.mp3 --from 0:06 --to 0:59 --source guitar,vocals,other --center-strength 0
```

On the sample this repo was tuned against — a continuous lead over a full-band backing, 0:06 to
0:59 — that setting left **12 dB less** of the lead's harmonic energy in the backing than the
defaults did, essentially hitting the floor set by drums and bass sitting at those same
frequencies. The cost is that more of the band rides along in the solo file.

Only widen `--source` if you actually have vocals-free material; on a song with singing,
`--source guitar,vocals` puts the singer in your solo file.

### Reading the report

- **mask keep** — how much of the source stems' energy inside the window survived the masks.
  Near 0 dB means the masks are barely doing anything (the stem is already centered and
  coherent, so `--melody-mask` is your lever). Very negative means they're aggressive — and
  that everything they cut went into the backing.
- **solo level** — inside vs. outside the window. Outside is always `-inf` by construction.
- **null test** — must be under `1e-6`, or the run exits non-zero.

## Options

```
split-solo INPUT --from TIME --to TIME [-o OUTDIR]
  --model htdemucs_6s   --device auto|mps|cpu   --shifts 1   --overlap 0.25
  --source guitar       # comma-separated stems the solo is built from
  --fade 30             # ms raised-cosine fade at the window boundaries
  --pad 0               # seconds of context kept around the window
  --center-strength 1.0 --center-pan 0.0
  --melody-mask --harmonics 12 --harm-cents 60 --harm-floor 0.1
  --bitrate 320k | --vbr
  --preview             # encode only window ±2 s, for fast auditioning
  --keep-stems --no-cache --no-normalize -v
```

`--from` / `--to` accept `M:SS`, `M:SS.mmm`, `H:MM:SS`, or bare seconds.

A `--model` whose sources lack `guitar` exits with an error listing that model's actual sources.

## The stem cache

Stems live in `~/.cache/audio-splitter/stems/<key>/` — roughly 6 float32 WAVs per track, about
250 MB per 4-minute song. Delete that directory to reclaim the space, or pass `--no-cache` to
skip writing it.

## Tests

```bash
pytest              # fast: pure DSP, no model needed
pytest -m slow      # end-to-end on a synthesized song (tests/make_fixture.py)
```

The slow suite generates its own fake song — click track, bass, hard-panned rhythm chords, and a
center lead over a known window — so no copyrighted audio is ever committed. Drop your own test
audio in `samples/`; its contents are gitignored.

## Runtime

About 3 s to separate a 65 s track on an M4 Pro at `--shifts 1` on MPS, a few minutes on CPU.
Masking and encoding add a second or two. Every run after the first on the same track hits the
stem cache.
