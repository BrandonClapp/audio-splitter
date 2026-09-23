# Guitar solo splitter — implementation plan

> Status: **plan only, nothing implemented.** The repo is empty (no commits). This document is the
> handoff spec for the agent that will build it.

## Context

The goal is a tool that takes a song, isolates the **lead guitar solo** over a time range the user
specifies, and writes **two MP3s**: the solo and the backing music behind it.

Decisions made by the user during planning:

- The solo is identified by a **manual time window** (`--from` / `--to`), not auto-detected.
- **Rhythm guitar belongs in the backing MP3**, not the solo MP3.
- **CLI only** — no web UI, no GUI wrapper.

The central design idea that makes "rhythm guitar goes to backing" fall out for free:

> Estimate the solo signal as well as possible, then define **`backing = original − solo`**.

Everything the solo estimate doesn't confidently claim — rhythm guitar, drums, bass, vocals, keys, and
the separation model's own error — lands in the backing track automatically. Nothing is lost or
duplicated, and the two outputs sum back to the original. That invariant is directly unit-testable and
is the spine of the design: a weak solo estimate degrades gracefully, because the backing stays
complete either way.

## Environment facts (verified during planning, not assumed)

- `python3` = Homebrew **3.14.7**; `ffmpeg` and `lame` on PATH; macOS 26.6.2; Apple **M4 Pro**, 48 GB.
- `demucs` **4.1.0** and its entire dependency tree resolve on cp314/arm64 as **prebuilt wheels** —
  nothing compiles. Verified with a dry-run resolve of `demucs librosa soundfile numpy` against
  `--python-version 3.14 --only-binary=:all:`: 50 wheels, 0 errors (torch 2.14.0, librosa 1.0.0,
  numba 0.67.0, scipy 1.18.1, numpy 2.5.3). Note that scipy/numba/torch publish `macosx_14_0` /
  `macosx_12_0` wheel tags, so a resolve pinned to `macosx_11_0` will falsely report scipy as missing.
- demucs 4.1.0 calls `torch.load(..., weights_only=False)` explicitly (`demucs/repo.py:69`,
  `demucs/states.py:59`), so it is **not** broken by the torch ≥2.6 `weights_only` default change.
- `htdemucs_6s` is the model with a **guitar** source (`drums, bass, other, vocals, guitar, piano`).
  It's effectively the only widely available model that emits a guitar stem — `audio-separator` reaches
  the same conclusion (`audio-separator -l --list_filter=guitar` → `htdemucs_6s.yaml`), so there is no
  reason to take on that heavier dependency tree.
- demucs 4.1.0 no longer needs `torchaudio` at runtime (it uses `sphn` for audio I/O).
- `demucs.api.Separator.separate_tensor(wav, sr)` returns `(wav, {source: tensor})` where the returned
  `wav` is the same length as the input and the stems are rescaled back to the original level
  (`out = out * std + mean` in `demucs/api.py`). So `backing = wav − solo` is exact and length-matched.
  That same line adds `mean` to *every* stem, which is a second reason to use subtraction rather than
  summing the other five stems.

## What is genuinely hard — state this honestly in the README

`htdemucs_6s`'s guitar stem contains **all** guitars, not just the lead, and it is the weakest of that
model's six stems. Time-gating alone does not fix this, because rhythm guitar usually plays *during* the
solo. Hence the refinement stage in step 4 below. Expect a good-but-not-surgical result: a usable solo
track, and a backing track that's clean because it is defined as the exact complement.

## Approach

### Pipeline

1. **Decode** input (mp3/m4a/flac/wav — anything ffmpeg reads) → 44.1 kHz stereo float32 via `ffmpeg`.
   Represent audio as `(2, n)` float32 throughout, channels-first, matching demucs so arrays and
   tensors pass back and forth without transposing.
2. **Separate** with `htdemucs_6s` on MPS, falling back to CPU. **Cache stems on disk** keyed by
   (audio hash, model, shifts, overlap) under `~/.cache/audio-splitter/stems/<key>/`. This is the key
   usability decision: the user will iterate on the window and mask settings by ear, and the cache makes
   every run after the first take seconds instead of minutes.
3. **Time gate** the guitar stem to `[from, to]` with a raised-cosine fade (`--fade`, default 30 ms) so
   there are no clicks at the boundaries. Outputs stay full-length and sample-aligned with the original.
4. **Refine to lead only** inside the window — two independent, tunable masks on the guitar stem:
   - **Center/coherence mask** (default **on**, `--center-strength 1.0`). STFT L/R (n_fft 4096,
     hop 1024). Per bin, compute a balance term `b = (‖R‖−‖L‖)/(‖R‖+‖L‖+ε)` ∈ [−1, 1] and a pan score
     `clip(1 − |b − pan|, 0, 1)`, times a phase-coherence term `clip(cos(∠L − ∠R), 0, 1)`; raise the
     product to `--center-strength`, apply to both channels, iSTFT with explicit output length. This is
     what removes hard-panned / double-tracked rhythm guitars, which is how rhythm parts are usually
     recorded. `--center-pan` retargets it if the solo itself sits off-center; `--center-strength 0`
     naturally disables it (`x ** 0 == 1`).
   - **Melody/harmonic mask** (default **off**, `--melody-mask`). `librosa.pyin` on the mono sum
     (fmin 80 Hz ≈ low E, fmax 1.4 kHz to allow high frets and bends), with `frame_length=n_fft` and
     `hop_length=hop` so its frames align with the STFT — then trim to the shorter of the two frame
     counts rather than trusting them to match. Build a soft mask keeping bins within `--harm-cents`
     (60) of k·f0 for k = 1..`--harmonics` (12), Gaussian roll-off in cents, combined across harmonics
     by `max`, then `floor + (1 − floor) * mask` with `--harm-floor` (0.1) so pick attack and distortion
     texture survive. Unvoiced frames collapse to the floor.
     Build the mask **in time-blocks** (e.g. 256 frames): the naive `(freq × harmonic × frame)` array is
     ~1 GB for a full-length track.
     Off by default because on heavily distorted guitar the harmonic series is dense and this mask can
     thin the tone. It is the knob to reach for when rhythm guitar is *still* audible after center
     masking.

   Efficiency + correctness detail: slice the guitar stem to the window (plus fade padding), run the
   masks on the slice, apply the fade envelope, then zero-pad back to full length. STFT edge effects
   land exactly where the fades are.
5. **`solo`** = refined signal. **`backing`** = `original − solo`, sample-exact.
6. **Level safety**: measure true peak of both. If either exceeds 0 dBFS, apply **the same** gain to
   both files so their relative balance is preserved, and print the applied gain in dB.
   `--no-normalize` opts out and lets it clip.
7. **Encode** both to MP3 via `ffmpeg -codec:a libmp3lame` (320 kbps CBR default, `--vbr` → `-q:a 0`),
   with ID3 titles `<track> — guitar solo` and `<track> — backing`.
8. **Report** (stdout): model, device, resolved window, runtime, solo RMS inside vs. outside the window,
   any applied gain, and a **null test** — `max|(solo + backing) − original|` at the float stage, which
   must be < 1e-6. That number is the proof the split is lossless before MP3 encoding.

### CLI

```
split-solo INPUT --from 2:14 --to 2:48 [-o OUTDIR]
  --model htdemucs_6s   --device auto|mps|cpu   --shifts 1   --overlap 0.25
  --fade 30             # ms boundary fade
  --pad 0               # seconds of context kept around the window
  --center-strength 1.0 --center-pan 0.0
  --melody-mask --harmonics 12 --harm-cents 60 --harm-floor 0.1
  --bitrate 320 | --vbr
  --preview             # encode only window ±2 s, for fast auditioning
  --keep-stems --no-cache --no-normalize -v
```

`--from` / `--to` accept `M:SS`, `M:SS.mmm`, `H:MM:SS`, or bare seconds. Outputs: `<name>.solo.mp3` and
`<name>.backing.mp3`, next to the input unless `-o` is given. A `--model` whose sources lack `guitar`
exits with a clear error listing that model's actual sources.

## Files to create

```
pyproject.toml                     # hatchling; [project.scripts] split-solo = "audio_splitter.cli:main"
README.md                          # install, usage, tuning guide, honest quality caveats
.gitignore                         # .venv/, __pycache__/, *.mp3, *.wav, out/, samples/*
samples/.gitkeep                   # where the user drops test audio; contents never committed
scripts/bootstrap.sh               # python3 -m venv .venv; pip install -e .; prefetch htdemucs_6s weights
src/audio_splitter/__init__.py     # __version__, SAMPLE_RATE = 44100
src/audio_splitter/cli.py          # argparse, timecode parsing, orchestration
src/audio_splitter/io_audio.py     # ffmpeg decode/encode, float WAV cache I/O, ID3 tags, peak
src/audio_splitter/separate.py     # demucs.api.Separator wrapper, device selection, stem cache
src/audio_splitter/solo.py         # time gate, center/coherence mask, harmonic mask, complement
src/audio_splitter/report.py       # metrics + null test output
tests/test_timecode.py
tests/test_solo_masks.py
tests/test_complement.py
tests/test_integration.py          # marked slow, deselected by default
tests/make_fixture.py              # synthesizes a fake "song" — no copyrighted audio in the repo
```

Dependencies: `demucs>=4.1.0`, `numpy>=2.1`, `librosa>=1.0.0`, `soundfile>=0.13`; `pytest` as a dev
extra. `requires-python = ">=3.12"` (librosa 1.0.0's floor).

### Reuse rather than reimplement

- `demucs.api.Separator` — the supported Python entry point; takes `model`, `device`, `shifts`,
  `overlap`, `segment`, `jobs`, `progress`, `callback`, and returns `(origin, {source: tensor})`. Use
  this rather than shelling out to the `demucs` CLI, because the stem tensors are needed for masking
  and caching.
- demucs' own `--two-stems guitar --other-method minus` proves the complement approach and is worth
  running once as a baseline to A/B against, but it cannot do the lead-vs-rhythm refinement — so only
  step 4 is new code, not the separation.
- `librosa.pyin`, `librosa.stft` / `istft`, `librosa.fft_frequencies` for the masks.
- `ffmpeg` for all decode/encode; `soundfile` only for the float32 stem cache.

## Verification

1. **Bootstrap**: `bash scripts/bootstrap.sh`, then `split-solo --help`. First run downloads the
   `htdemucs_6s` weights once (a few hundred MB); the script does this up front so a later run doesn't
   stall mid-task.
2. **Unit tests** — `pytest -m "not slow"`, all pure DSP, no model needed:
   - timecode parsing, including malformed input and `to <= from`;
   - gate: correct length, exactly zero outside the window, monotonic fades;
   - center mask on a synthetic mix (center 440 Hz tone + hard-L 300 Hz + hard-R 301 Hz) — assert the
     panned parts are attenuated past a threshold while the center tone loses little;
   - harmonic mask on a 220 Hz sawtooth plus a 130/165/196 Hz chord — assert the chord is attenuated;
   - **complement invariant**: for random input and a random mask,
     `max|(solo + backing) − original| < 1e-6`.
3. **Synthetic integration** — `pytest -m slow`: `tests/make_fixture.py` synthesizes a ~20 s fake song
   (click track + bass line + panned rhythm chords + a center lead melody over 8–16 s), runs the full
   pipeline with `--from 8 --to 16`, and asserts: both MP3s exist; durations match the source within one
   frame; decoded `solo + backing` vs. original nulls to better than −40 dB (the MP3-lossy floor); solo
   RMS inside the window is far above solo RMS outside it.
4. **Real-song tuning loop — the user listens, the agent adjusts.** This step is what actually
   determines whether the tool is any good, so treat it as part of the work, not an afterthought.

   The user supplies a sample MP3 (dropped in `samples/`, gitignored so nothing copyrighted is
   committed) and the solo's start/end timestamps. Then iterate:

   1. Run `split-solo samples/<track>.mp3 --from <start> --to <end> --preview`, which encodes only the
      window ±2 s so each round is fast.
   2. On the first round also pass `--keep-stems`, and check *before* blaming the masks whether
      `htdemucs_6s` put the solo in the guitar stem at all — on some tracks a distorted lead leaks into
      `other`. If it did, the fix is a `--source` flag to build the solo from `guitar + other`; add it
      at that point rather than speculatively now.
   3. The user listens to `*.solo.mp3` and `*.backing.mp3` and reports in plain terms ("rhythm guitar
      still in there", "solo sounds underwater", "cymbals bleeding in", "solo cuts off early").
   4. Map that to the knobs and re-run. This mapping also becomes the README's tuning guide:
      - rhythm guitar still audible in the solo → add `--melody-mask`, then raise `--harmonics`;
      - solo sounds thin, phasey, or "underwater" → lower `--center-strength` (try 0.5) or raise
        `--harm-floor` (try 0.25);
      - solo is panned rather than centered → set `--center-pan`;
      - clicks or an abrupt entry/exit → raise `--fade`, or adjust the window and `--pad`;
      - general smearing and bleed in both files → `--shifts 5` (≈3× runtime, better separation).
   5. Once it sounds right, record the settings that worked as the documented defaults (or as a README
      example if they're track-specific) and run the full-length encode.

   The stem cache makes this loop cheap: only round 1 pays for separation; every later round is masking
   plus encoding, on the order of a second or two.
5. **Device sanity**: run once with `--device mps` and once with `--device cpu` and confirm the null test
   and the audible result agree. Check MPS output for non-finite values and fall back to CPU
   automatically if any appear.

## Notes / risks

- **The quality ceiling is the model, not the code.** `htdemucs_6s`'s guitar stem is its weakest, and no
  off-the-shelf model separates *lead* from *rhythm* guitar. Steps 3–4 are heuristics. The
  `backing = original − solo` design means a weak solo estimate degrades gracefully: the backing stays
  complete and clean; the solo file just carries more bleed.
- The stem cache grows at roughly 6 float32 WAVs per track (≈250 MB per 4-minute song). Document
  deleting `~/.cache/audio-splitter` in the README; `--no-cache` skips writing it.
- The one-time model download needs network; everything after that is offline.
- Runtime estimate on this M4 Pro: roughly 30–60 s per 4-minute track on MPS at `--shifts 1`, a few
  minutes on CPU; masking and encoding add a second or two.
