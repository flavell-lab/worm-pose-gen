"""The model and label library: setups, label collections, datasets, benchmarks and models (docs/APP_SIMPLIFICATION.md, section 1).

Two libraries share one layout.  The **lab library** holds what the
developer publishes and is read-only to the app; the **personal library**
holds everything the app creates.  Items are named ``lab:<id>`` or
``mine:<id>`` (:mod:`.roots`)::

    <library>/
      setups/<setup>.json                  a microscope: video settings, pixel size, fps, default models
      setups/<setup>.override.json         (personal) the user's defaults for a lab setup
      setups/defaults_log.jsonl            every default change: who, when, why
      recording_folders.json               (personal) folders searched for a setup's recordings
      recordings.json                      (personal) recordings registered to a setup by hand
      labels/<setup>/                      the setup's label collection (this library's part)
        index.json                         every revision of every label, for fast listing
        <recording>/<frame>/<revision>.npz
      datasets/<dataset>/
        dataset.json                       setup, name
        splits.json                        recording -> split; a recording not in it is not included
      benchmarks/<benchmark>.json          a frozen list of test-label revisions
      pseudo_labels/<set>/                 self-training items from the pipeline's confident fits
      models/<model>/
        model.json                         the model card
        weights.ckpt
        training/                          hyperparameters, metrics, the exact label revisions used
        evaluations/<scope>.<benchmark>.json
      evaluations/lab.<model>/             (personal) evaluations of lab models
      cache/body_targets/<builder>/<sha256>.{npz,json}   (personal) body targets of each label revision, per builder model

Modules: :mod:`.roots` (where the libraries are, references, file
helpers), :mod:`.setups` (setups, default models, recording -> setup),
:mod:`.collection` (a setup's labels; saving labels), :mod:`.datasets`
(per-recording splits over a collection), :mod:`.labels` (the label
revision file), :mod:`.targets` (the body
target cache), :mod:`.benchmarks`, :mod:`.models` (cards, weights,
evaluations, recipes), :mod:`.pseudo_sets` (self-training items),
:mod:`.inference` (running a model).

Reading labels for training and evaluation
------------------------------------------

::

    from worm_pose_gen.library import Libraries, labels, load_targets

    libs = Libraries.for_host()                       # or Libraries(lab=..., personal=...)
    for record in labels(libs, "mine:nir-copper", "train"):
        label = record.load()                         # image, image_raw, mask, context, context_valid,
                                                      # nose_xy, nose_valid, trace_xy, head_xy, meta
        targets = load_targets(libs, record)          # (meta, arrays) or None until built
        ...

:func:`labels` gives the setup's current labels (a personal revision of a
lab label is newer, so it wins) of the recordings the dataset puts in the
split.  Each :class:`LabelRecord` carries its split (the recording's in the
dataset), ``status`` (``mask_only``, ``complete`` or ``auto``),
``origin`` and the revision's ``sha256``; ``record.identity`` names the
exact revision and :func:`trained_on` summarizes a run's records for its
model card.  ``benchmark_labels(libs, "lab:nir-v1")`` gives a benchmark's
frozen revisions.  Body targets are built with :func:`build_targets`
(fitter on the GPU; the setup's default mask model, the *builder*, segments
context frames for chain fits) and read with :func:`load_targets`; both
key the cache by the label revision and the builder, so a new default model
means new targets; ``meta["fit_iou"]``, ``meta["self_contact"]`` and the
arrays ``ap``, ``overlap``, ``head_xy``, ``tail_xy``, ``diameter_px`` are
what :class:`body_net.BodyFieldDataset` reads from the old records today.
"""

from .benchmarks import Benchmark, benchmark_labels, freeze_benchmark, get_benchmark, list_benchmarks, write_benchmark
from .collection import Collection
from .datasets import SPLITS, Dataset, create_dataset, fingerprint, labels, list_datasets, trained_on
from .labels import BENCHMARK_ORIGINS, ORIGINS, STATUSES, Label, LabelRecord
from .models import (
    ModelCard, create_model, evaluation_path, evaluations, get_card, list_models, make_inputs, save_evaluation,
    training_dir, weights_path, write_model,
)
from .pseudo_sets import PseudoLabelSet, create_pseudo_labels, get_pseudo_labels, list_pseudo_labels
from .roots import LAB_LIBRARY_BY_HOST, Libraries, make_ref, parse_ref
from .setups import (
    Setup, add_recording_folder, create_setup, defaults_log, get_setup, list_setups,
    recording_folders, recording_id, recording_sources, register_recording, remove_recording_folder, set_default,
    setup_for_recording, write_setup,
)
from .targets import SETUP_DEFAULT, build_targets, cached_meta, load_targets, target_builder

__all__ = [
    "BENCHMARK_ORIGINS", "Benchmark", "Collection", "Dataset", "LAB_LIBRARY_BY_HOST", "Label", "LabelRecord", "Libraries", "ModelCard",
    "ORIGINS", "PseudoLabelSet", "SETUP_DEFAULT", "SPLITS", "STATUSES", "Setup", "add_recording_folder", "benchmark_labels",
    "build_targets", "cached_meta", "create_dataset", "create_model", "create_pseudo_labels", "create_setup", "defaults_log", "evaluation_path",
    "evaluations", "fingerprint", "freeze_benchmark", "get_benchmark", "get_card", "get_pseudo_labels", "get_setup", "labels",
    "list_benchmarks", "list_datasets", "list_models", "list_pseudo_labels", "list_setups", "load_targets", "make_inputs", "make_ref",
    "parse_ref", "recording_folders", "recording_id", "recording_sources", "register_recording", "remove_recording_folder", "save_evaluation", "set_default",
    "setup_for_recording", "target_builder", "trained_on", "training_dir", "weights_path", "write_benchmark", "write_model", "write_setup",
]
