# Developer reference

How the pipeline, the models and the app are built, tested and trained from
the command line, and the research history behind them. Using the app is in
the [README](../README.md); the agents' working rules (GPU hosts, waiting on
long runs) are in [`AGENTS.md`](../AGENTS.md).

## Environment

`scripts/bootstrap_environment.sh` installs the checkout (README); every
command below runs through `scripts/project_env.sh`, which keeps the
environment and caches inside the checkout and numbers GPUs as `nvidia-smi`
does. Where things go is `worm_pose_gen.settings`: one data directory
(`settings.json`, default `user_data/`) with `library/`, `workspaces/`,
`cache/` (flat fields, recording priors) and `runs/` (script outputs).
`scripts/preflight.py` checks an install.

## The app in more detail

What the README leaves out for lab members.

**Install and data.** The checkout and its Python environment take about
5 GB. `--data-dir` is saved in `settings.json`; without it the data
directory is `user_data/` inside the checkout. To move it, move its contents
and run `scripts/bootstrap_environment.sh --data-dir <new folder>`. In the
preflight output a `warn` line is something the app works without (no GPU,
the lab library not reachable, little free space).

| | |
|---|---|
| Data directory | `library/` labels, datasets, models and recording folders; `workspaces/` one per opened recording, with its exports; `cache/` flat fields and body-size priors, rebuilt when missing; `runs/` outputs of command-line scripts |
| Lab library (read-only) | `/storage/fs/store1/shared/worm-pose-models` on flv, `/orcd/data/flavell/001/shared/worm-pose-models` on Engaging |
| The checkout | the code, `.venv`, `settings.json` |

**Serving.** The app listens on 127.0.0.1 only and has no login, so it is
reachable through an ssh tunnel or by anyone logged in to the same machine.
When port 8768 is taken it uses the next free one and prints that. It uses
the GPUs of its machine, the least busy one per analysis (`--gpus 2,3`
limits them; `./worm-pose-app --help` lists the options). On Engaging the
printed `ssh` command goes through the login node to the job's node, and
**Run on → SLURM** sends analyses and training to the cluster, so they keep
running after the interactive job ends.

**Recordings.** A recording folder is searched two levels deep for `.h5`
recordings and `.avi` and `.mp4` videos; each new file is checked in the
background, and one that cannot be read says why (damaged on disk, or an
HDF5 compression plugin that is not installed). A video is read in place
(`frame_reader`, through PyAV): it is indexed once, the first time it is
opened, and the index is cached in the data directory's `cache/video_index`
(rebuilt when the file changes). A frame read out of order takes about
2 ms in an MJPEG `.avi` but 50–300 ms in an H.264 `.mp4` with keyframes far
apart, which decodes from the keyframe before it; reading H.264 forward runs
at about 440 frames/s. The Recordings screen lists a setup's recordings (a
setup is a microscope; the lab's rigs are `NIR behaviour camera (flv
rigs)`), analysed ones first, with their status.

**Workspace.** Four layers (keys 1–4): mask, midline (head a square, tail a
circle), outline, and the A-P field from head to tail. At the bottom are the
transport, the issue track and the curvature kymograph: drag to scrub,
right-drag to select a range for a fix. Flip works on a whole issue or one
frame; Refit chooses its method from the issue's reasons; saving an edited
mask refits around it; Relabel sends keyframes to Labeling and rebuilds the
stretch once they are labeled. **Re-analyse…** runs the analysis again, for
example with another model.

**Exports.** `workspaces/<workspace>/exports/<recording>_<time>/` holds a
Parquet table with one row per frame and `export.json` (units, models,
setup). Columns: frame, time, status (automatic, reviewed, fixed or
unresolved), the midline from head to tail, curvature, width profile, head
and tail positions, centroid and velocity. Lengths are in µm, positions in
image coordinates (origin top left, y down); each column's unit and
description are also in the Parquet metadata.

**Labeling.** A label is one frame's mask and its body (the midline from the
head). Queues: a workspace's Relabel keyframes, **New queue** (frames spread
over chosen recordings, favouring the ones the model is least sure of) and
**Browse labels**. Without a GPU the model's proposal takes 15–20 s per frame
and is made only on **Propose**.

**Training.** Models are scored on a benchmark (mask IoU, head/tail correct,
A-P error). **Use as default** sets a personal default for a setup. **Train
from this** fills the form with the model's recipe (settings, pseudo-label
sets, fibre share); **From scratch** under Start from uses the default body
model's recipe on fresh weights. A dataset says which recordings train,
validate or test a model, combining the lab's labels with yours; a trained
model is evaluated on the setup's benchmarks.

## Tests

Python tests use `unittest`. `scripts/run_tests.py` runs them on the CPU,
one process per test class, twelve at a time with four OpenMP threads each
(torch otherwise takes 128 threads per process and parallel runs slow each
other down several times over), slowest classes first. It tests the code of
the checkout it sits in, so a worktree runs its own copy. `--fast` skips the
25 tests marked `@slow` (`tests/slow.py`): they run the pose optimizer for
ten seconds or more each and make up nearly 90% of the suite's time.

```bash
# everything but @slow, about a minute on one flv machine:
scripts/project_env.sh uv run --no-sync --frozen python scripts/run_tests.py --fast
# the whole suite, about 7 minutes:
scripts/project_env.sh uv run --no-sync --frozen python scripts/run_tests.py
# some modules:
scripts/project_env.sh uv run --no-sync --frozen python scripts/run_tests.py tests.test_pipeline tests.test_fixes
```

Mark a new test `@slow` when it takes ten seconds or more. A single test
runs with `unittest` directly; list modules explicitly, because
`unittest discover` does not work with this layout:

```bash
env PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m unittest tests.test_pipeline.StageTests
```

The browser tests drive the three pages with Playwright against synthetic
CPU apps that each test starts itself (`tests/browser/*_fixture.py`):
`tests/browser/workspace.cjs`, `tests/browser/labeling.cjs` and
`tests/browser/training.cjs`. Run them with `node` from the repository root.
Set `PLAYWRIGHT_MODULE` (a `playwright` or `playwright-core` module directory)
and `CHROMIUM_EXECUTABLE` when they are not installed in the default
locations, `LD_LIBRARY_PATH` when Chromium's libraries are not on the system,
and `PYTHON` when the interpreter is not `.venv/bin/python`. `WS_SHOTS`,
`SCREENSHOTS` and `SCREENSHOT_DIR` (Workspace, Labeling and Training) name a
directory for screenshots of the main states. The header of each test lists
what it covers.

## Developer mode

`./worm-pose-app --dev` adds the research diagnostics to the app.
With `--dev` the Workspace also shows the raw frame, every layer of the
frame payload (the segmenter's probability and the cleanup steps behind the
mask, the fitted and independent tubes, the network's crossings), the
**Details** drawer (classification, the ambiguity flags with their values and
thresholds, width and curvature along the body, mask statistics, the
workspace summary), one extra per-frame series on the timeline, a jobs
drawer, the Refit algorithm and parameter override, and the stage list and
range in the Analyse dialog. Only `--dev` runs the segmenter on a rested
frame, because analysts see the stored mask.

## Libraries: setups, datasets, models

Models, labels, setups and benchmarks live in two libraries with the same
layout (`worm_pose_gen.library`):

- the **lab library**, read-only to the app:
  `/storage/fs/store1/shared/worm-pose-models` on the flv-c machines and
  `/orcd/data/flavell/001/shared/worm-pose-models` on Engaging
  (`library.LAB_LIBRARY_BY_HOST`, `--lab-library`);
- your **personal library**, where everything the app makes goes:
  `library/` in your data directory (`settings.json`, `--library`).

A **setup** is one microscope: its video dataset and flat-field setting, its
pixel size and frame rate, and the default model of each role (`mask`,
`body`; a body-field model as the mask default means the masks come from the
body model). Which recordings belong to a setup is the user's choice: the
recording folders and single files in the personal library
(`recording_folders.json`, `recordings.json`). A personal override of a lab setup's defaults
is logged with who changed it and why. Every label of a setup is in the
setup's **collection** (the lab's labels and your own; your edit of a lab
label is a newer revision). A **dataset** chooses, per recording of the
collection, whether its labels train, validate, test or are not included;
every recording starts not included, so you choose its split in the
Training page's Datasets tab. A **label** revision stores the frame,
its context frames, the mask and the human body fields. A **benchmark** is a
frozen list of test-label revisions. A **pseudo-label set** holds
self-training items: frames the pipeline fitted confidently, with targets
rendered from its fits and their context frames
(`worm_pose_gen.pseudo_labels`, `library.pseudo_sets`). A **model** is a card (`model.json`),
`weights.ckpt`, its training records and its evaluations. Items are named
`lab:<id>` or `mine:<id>`.

The first lab library is built from the old stores (`segmentation_v1`, the
app corpus and their body-field records) and two trained runs:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/migrate_to_library.py \
  --out /path/to/new-lab-library --body-run checkpoints/body_net/runs/<run> [--seed-cache <personal library>]
```

It writes setup `nir-flv`, its label collection, dataset `nir-labels` (splits per recording),
benchmark `nir-v1`, and the model cards of the segmenter run
(`--segmenter-run`, `nir-hand284`) and the body-field net run. Publishing a
personal setup, your labels of a setup (`publish.py labels lab:nir-flv`), a
dataset's splits, a benchmark, a pseudo-label set or a model into the lab
library is a developer step:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/publish.py model mine:copper-ft \
  --default body --reason "fixes heads on copper plates"
```

A published item is frozen: an id the lab already has is refused.
`--default` is repeatable (`--default body --default mask`). Publish what an
item refers to first (the script says what is missing): a model's setup,
parent, datasets and pseudo-label sets, a benchmark's labels. Publishing
writes the flv copy of the lab library; copy it to Engaging afterwards:

```bash
scripts/sync_lab_library.sh <user>@orcd-login.mit.edu
```

## Training and evaluation from the command line

The Training page's jobs run these commands, which also work by hand:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train.py \
  --setup lab:nir-flv --dataset mine:nir-copper --start-from lab:nir-body-lags3 --max-epochs 50
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_model.py \
  --model mine:nir-copper-196 --benchmark lab:nir-v1
```

`scripts/train.py --help` lists the settings (`--context none|short` from
scratch, learning rate, batch size, patience, loss weights and more). A
model keeps the one checkpoint with the lowest validation loss, and the
evaluations go to `models/<id>/evaluations/`. The code is in
`worm_pose_gen.model_training` and `worm_pose_gen.model_eval`.

A body-field run can mix in pseudo-label sets (`--pseudo-labels lab:<id>`,
repeatable, at `--pseudo-fraction` of every epoch) and fibre distractors
on a share of the hand labels (`--fiber-share`). With the form parameters
they make a model's recipe (`model_training.recipe`), which the Train tab
fills in from the model it starts from, or from scratch from the setup's
default body model. The lab default `lab:arch-combo-self-fib` was trained
this way: a fine-tune of `lab:on-food-finetune-329` at learning rate 1e-4
(patience 10, plateau 3) with `lab:self-train-r1` at 0.5 and fibres at 0.5;
its parent was trained from scratch with the short context at the default
settings. Build a new set from analysed workspaces with
`scripts/build_pseudo_labels.py --set <id> --setup lab:nir-flv --workspaces ...`.

## Jobs and the HTTP API

At startup the app finds this node's GPUs and whether SLURM (`sbatch`,
`squeue`, `sacct`, `scancel`) is on the path (`GET /api/compute`). Analyse
and Train then offer one **Run on** choice: **This machine**, or **SLURM**
with partition and time prefilled from the host's defaults
(`compute.SLURM_DEFAULTS_BY_HOST`; on Engaging `ou_bcs_normal`, 12 h, one
GPU). A SLURM job writes a batch script that runs the same command and is
followed by its job id, so it survives an app restart. The app refuses
node-local paths (`/tmp`, `/scratch`, `/dev/shm`) for it. When neither
exists, analysis and training are disabled and the reason is shown. Job
records, progress and logs live under `<workspaces root>/jobs`.
`POST /api/jobs` with `kind: stage` runs one pipeline stage
(`python -m worm_pose_gen.pipeline --workspace ... --stage ...`), and
`kind: command` runs any command. Both take `run_on`, `slurm` and `gpu`.

Everything the pages do goes through the HTTP API (`/api/home`,
`/api/recordings/open`, `/api/workspaces/{name}/…` with `analyse`,
`status`, `frame`, `issues`, `fixes`, `mask`, `kymograph`,
`network-fields` and `export`, then
`/api/labeling`, `/api/queues`, `/api/library`, `/api/training`, `/api/jobs`
and `/api/compute`), so a script can drive the same work. `/docs` is the
generated OpenAPI page.

## Fit poses over a recording

Segment, clean, and fit the body model to every frame of a stretch, in GPU
batches, with an optional overlay video:

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/fit_recording.py \
  --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \
  --start 0 --frames 1200 --video --scale 0.5
```

Cleaning thresholds the probability at `--threshold 0.5` and fills each hole
the mask outlines on all sides, whole, when no 17-pixel square fits in it
(`--hole-radius 8`); a hole open to the background through any gap, a
diagonal one included, stays open, and `--no-fill-holes` skips the fill.
Every connected component is kept: a frame with more than one is reported by
the ambiguity `fragments` flag rather than cut down to its largest piece.

Each run writes `summary.json`, `poses.npz`, and `overlay.mp4` to a
timestamped directory under `runs/poses/` in the data directory.
`--preset fast` (default, about 0.25 s/frame) trades 0.008 median IoU against
`--preset reference`, which reproduces the single-frame fitter exactly at
about 5 s/frame. The width of the tube is a scale times a symmetric template
times a smooth log-space correction (`--width-coefficients`, default 6, 0 for
the symmetric model; `--width-prior` pulls it toward zero), so the two ends
may taper differently; every stored pose is oriented with the thinner end,
the tail, last (`--no-orient` keeps the fitted orientation). In the overlay
the head is a square and the tail a circle. Each run also writes residual
images (mask the tube misses in blue, tube outside the mask in red) for its
`--residual-frames` worst frames and any `--dump-frames`;
`scripts/render_pose_run.py` produces the video and residual images for a
stored run without refitting, and `scripts/compare_pose_runs.py` puts
several runs side by side on the same frames.

`--body-net <weights>` (a body-field model's `models/<id>/weights.ckpt` in a
library) adds the body-field network:
its A-P field (over the front half of the body) and head score every fit
(independent, propagation and track pass), its proposed trace is an extra
start (the independent fit fits it first and the standard starts only where
its IoU stays below 0.95), and the evidence rather than the taper decides
the orientation. With
a recording prior every fit keeps the prior's length: the trace start is
laid from the predicted head along the trace to that length, and the tail
ends wherever the body does (the network's tail, which people label
inconsistently, only orients starts). On the sequence set it cuts the
independent fits' frames below IoU 0.9 from 599 to 107, at about 230 ms
more per frame ([Body-field evidence in the pose fitter](BODY_FIELDS.md#body-field-evidence-in-the-pose-fitter)).
With `--mask-source body_net` the masks also come from that network instead
of the segmenter (`--checkpoint`), cleaned the same way; the segmenter is the
default, as its masks score better on the held-out labels
([Masks from the network](BODY_FIELDS.md#masks-from-the-network)).

The default runtime optimizations preserve fitting schedules
(`scripts/benchmark_pose_runtime.py` measures them). `--compile-energy` additionally compiles the whole
independent fitting loss; it adds startup cost and should be benchmarked on
your workload before enabling.

To look at the poses of a recording, analyse it in the pose app (the
Workspace page; see the [README](../README.md#analyse-a-recording)). With `--dev` it shows every layer
and per-frame series the pipeline produces.

By default the run first bootstraps a recording prior: frames spread over
the whole recording are fit with the hard bounds opened, whole worms (mask
clear of the image border) are kept, and robust medians of body length,
width scale and width profile become Gaussian priors that replace the hard
bounds (`recording_prior.json` in the run directory, cached under
`cache/recording_priors/` in the data directory). A body that
leaves the camera is started at the prior length, extended off camera
through the point where the mask meets the border, and its off-camera part
is censored; the in-view fraction reports how much was seen. Every frame is
started in both orientations and the energy gap between them is stored
(with `--body-net`, a frame where the network sees both ends is started
head first by them only, and has no gap).
`--prior none` restores the hard bounds, `--prior-file` reuses a stored
prior, `--rebootstrap` ignores the cache. Every frame also gets an
ambiguity score (`worm_pose_gen.ambiguity`: low overlap, self-overlap or
missed body by area, self-contact, enclosed holes, fragments, length far
from the prior, a jump since the previous frame); a score of 2 or more marks
a frame whose single-frame answer should not be trusted without its
neighbours. `scripts/ambiguity_report.py` recomputes it for stored runs.
Stretches of such frames are then refit by temporal propagation: the good
pose before the stretch is carried forward through it and the good pose
after it backward, each frame warm-started from its neighbour, all
stretches in lockstep, and per frame the lowest total energy among
independent, forward and backward wins (`source` in `poses.npz`;
`--no-propagate` skips it). Inside a chain each frame is also started from
a first-order prediction of the pose (the last change of shape, rotation
and centroid carried on with damping 0.6, `--prediction-damping`) and the
fit is pulled toward that prediction by a temporal prior
(`--temporal-prior-weight`, default 0.01, sigma half a width). The
propagation pass is the second pass over the ambiguous stretches: it
keeps up to three distinct chain states per direction (`--beam`), refits
each stretch frame's independent pose under the chain schedule with the
stretch's anchor length, and chooses one candidate per frame along the
stretch by dynamic programming over all candidates and their mirrors
(energy over `--path-temperature` plus the squared pose distance in
widths times `--path-distance-weight`, the in-view change times
`--path-inview-weight`, and the squared log length change times
`--path-length-weight`; `--no-path` restores the lowest energy per frame).
`--propagate-preset balanced` runs that preset's steps in the pass; it
was slower and worse on the raw spiral, so the default stays `fast`.
Stretches are also seeded by jumps of the track (a pose jump of more
than a width, or the body entering or leaving the camera at the border;
`--no-jump-seeds`, `--seed-length-fraction` adds length jumps). After
propagation a track length pass refits the frames outside the stretches
whose mask reaches the border and whose length departs from the track
(the median length of the whole bodies within `--track-window` frames)
with that length as their prior (`--track-refit`, `--track-sigma`,
`--track-tolerance`, `--no-track-length`); run before propagation it
disturbed the stretch anchors. The fitter's energy penalises bends tighter than a
radius of `--min-bend-radius` body widths (default 0.5; 0 disables). Each
frame also stores the fraction of the tube covered by mask
(`tube_coverage`), which stays high when a low IoU comes from mask the
tube does not claim, such as a plate streak segmented as worm; the frame
classification tags such frames "mask has extra body (segmentation)". Every
candidate is stored (`hypotheses_*` arrays, `path_*`, `prediction_xy`), with
the path's choice and where it overrode the lowest energy.

The sequence evaluation set, seven 300-frame clips with coils, self-contact,
fragments and camera exits, is the manifest `eval_sets/sequence_eval_set.json`;
`scripts/find_sequence_clips.py` proposes such clips from a mask-only scan
and `scripts/evaluate_sequence_set.py` fits and scores the set
(`eval_sets/heldout_sequence_set.json` holds held-out clips in the same
format, `eval_sets/temporal_eval_set.json` the clips of
`scripts/evaluate_temporal.py`).

## Segment with a fine-tuned network

The segmenter and the body-field network are library models (see
[Libraries](#libraries-setups-datasets-models)), trained and evaluated on
library datasets from the pose app's Training page or from the command line
([Training and evaluation](#training-and-evaluation-from-the-command-line)):

```bash
scripts/project_env.sh uv run --no-sync --frozen python scripts/train.py \
  --setup lab:nir-flv --dataset lab:nir-labels --start-from lab:nir-hand284
scripts/project_env.sh uv run --no-sync --frozen python scripts/evaluate_model.py --model lab:nir-hand284
```

The hand labels of the old store (`segmentation_v1`,
`segmentation_dataset.DEFAULT_DATASET_ROOT`) and of the app corpus were migrated into the lab dataset `nir-labels`
(`scripts/migrate_to_library.py`). New labels are made on the pose app's
Labeling page. They go into your personal dataset for the setup.

A targeted labeling round is a manifest of frames
(`scripts/build_labeling_manifest.py` builds one from the clip-candidate
scans of `scripts/find_sequence_clips.py`); `./worm-pose-app --queue
<manifest>` opens it as a Labeling queue. The splits come from the dataset,
per recording.

## History

The pipeline grew out of a classical, one-frame geometric method (local
darkness mask, skeleton, smooth body, notch repair, endpoint extension),
then a body model fitted to the mask, then the learned segmenter and
body-field network, temporal propagation and the app. The reports and
evidence of each step (the step-by-step pose pipeline plan, the
architecture survey of October 2026, the runtime optimizations, the
classical method's experiments) were removed from the tree in October 2026
and are in the git history.

## Repository layout

- `src/worm_pose_gen/`: the pipeline stages (`pipeline.py`) and the fitting
  behind them, the segmenter and the body-field network, the library
  (`library/`), the pose app (`app/`, its browser UI in `app_ui/`), and
  `settings.py` (where the data directory is).
- `scripts/`: install (`bootstrap_environment.sh`, `preflight.py`), the
  library's command-line steps (`train.py`, `evaluate_model.py`,
  `publish.py`, `sync_lab_library.sh`), fitting and evaluation tools, and the
  test runner (`run_tests.py`).
- `eval_sets/`: the evaluation clip manifests (frame ranges of raw
  recordings).
- `tests/`: unit tests; `tests/browser/` drives the app's pages.
- `docs/`: this reference, the app's design (`APP_SIMPLIFICATION.md`), the
  body-field network (`BODY_FIELDS.md`) and the labeling loop
  (`SEGMENTATION_LABELING.md`).
