# Google Colab Python 3.12 training notebook

`basic_training_notebook_colab_py312.ipynb` is the Python 3.12 Colab workflow maintained on the `fix/colab-python312` branch of the `CJMarais/micro-wake-word` fork. It does not replace `basic_training_notebook.ipynb`.

## Branch scope

The branch contains the following project changes:

- `notebooks/basic_training_notebook_colab_py312.ipynb` installs and verifies a pinned Python 3.12 environment, synthesizes samples, creates features, trains, validates, converts, and persists the quantized streaming TFLite model.
- `notebooks/requirements_colab_py312.txt` defines the Colab dependency set.
- `microwakeword/data.py` supplies bounded evaluation batches.
- `microwakeword/train.py` evaluates validation and ambient-validation data incrementally, emits periodic progress, and checkpoints the model, optimizer, and completed step.

The notebook clones `REPOSITORY_URL` at `REPOSITORY_REF`. Those settings must identify a revision containing the corresponding library changes.

## Execution and persistence

- `SMOKE_TEST = True` uses reduced sample counts and local temporary storage.
- Full mode mounts Google Drive and uses `MyDrive/micro-wake-word/training_runs/<RUN_NAME>/`.
- `RUN_NAME` is generated as lowercase ASCII snake_case from `TARGET_WORD`, followed by `_v<MODEL_REVISION>`. The JSON and TFLite use the same basename.
- `RUN_ACTION = "new"` requires that generated run name to be unused. Increment `MODEL_REVISION` for another fresh run of the same target.
- `MANIFEST_SCHEMA_VERSION` is separate from `MODEL_REVISION`. It must remain `2`; ESPHome's manifest `version` identifies the JSON schema, not a trained-model revision.
- `RUN_ACTION = "resume"` requires the original target and model version, matching metadata, and a TensorFlow checkpoint.
- Full checkpoints contain model weights, Adam optimizer state, and the completed training step. The three most recent checkpoints are retained.
- Generated speech, downloaded datasets, converted audio, and RaggedMmap features remain in the Colab runtime and are regenerated after runtime replacement.
- The exported model is copied to `<RUN_NAME>.tflite` in the persistent run directory. After that copy is verified, `<RUN_NAME>.json` is written beside it using `TARGET_WORD` as `wake_word`, the actual TFLite filename as `model`, and the configured deployment values. Browser download is used only when the TFLite copy fails.
- ESPHome v2 manifests require both the top-level `model` field and `micro.minimum_esphome_version`. Omitting `model` prevents a manifest-driven loader from resolving and bundling the TFLite file; omitting `minimum_esphome_version` fails v2 schema validation. The notebook emits and validates both fields.
- `MODEL_AUTHOR` is the public person or organisation responsible for the model and must be changed from `Your Name` before distribution. `MODEL_WEBSITE` is optional and, when set, must be an HTTP(S) project, repository, or author URL.
- Execution events and periodic resource samples are appended to JSONL files. The final cell writes a session-specific `run_summary_<SESSION_ID>.json` and refreshes `run_summary.json` with the latest completed session.

## External inputs and environment assumptions

- The runtime must use Python 3.12. GPU training is expected for the full configuration, but Colab hardware type, memory, storage, quotas, and session duration are not fixed.
- Piper source and model files, Hugging Face datasets, MIT room responses, FMA audio, and pre-generated negative features are network inputs. File locations and repository formats can change independently of this repository.
- Piper model selection is controlled by `PIPER_MODEL_DIRECTORY`, `PIPER_MODEL_FILENAME`, `PIPER_MODEL_RELEASE_TAG`, and `PIPER_MODEL_CONFIG_REF`. The generator-ready `.pt` assets are listed on the [Piper Sample Generator release page](https://github.com/rhasspy/piper-sample-generator/releases/tag/v2.0.0), and matching configuration filenames are listed in its [models directory](https://github.com/rhasspy/piper-sample-generator/tree/v3.2.0/models).
- Release `v2.0.0` currently contains `de_DE-mls-medium.pt`, `en_US-libritts_r-medium.pt`, `fr_FR-mls-medium.pt`, and `nl_NL-mls-medium.pt`.
- `TRAINED_LANGUAGES` is derived from the base language portion of the model filename's locale prefix. For example, `de_DE-mls-medium.pt` produces `["de"]`. This records the synthesis model language; it does not validate pronunciation or multilingual training coverage.
- Hugging Face authentication is optional. An `HF_TOKEN` may provide higher rate limits and faster or more reliable downloads; the notebook also operates without one. Store tokens in Colab secrets or environment variables, not in the notebook.
- Full-mode AudioSet input is streamed from the current balanced Parquet configuration. The configured sample count controls how many rows are converted.
- FFmpeg and libsndfile are installed through the Colab system package manager.
- Google Drive access requires the interactive authorization prompt shown by Colab.

## Repeatability notes

- The smoke-noise generator and clip splits use explicit seeds. Piper generation, augmentation, training sampling, TensorFlow operations, and resumed data generation are not fully deterministic.
- A resumed run restores model, optimizer, and step state. It does not restore Python, NumPy, TensorFlow, or dataset iterator random states.
- Regenerated speech and augmented features can differ between Colab sessions.
- Cached files in `/content/micro-wake-word-work` are reused within one runtime. They are not retained when Colab replaces the runtime.
- Performance records are comparable only when the run settings, repository commit, accelerator, Colab tier, and input counts are recorded together.
- `PROBABILITY_CUTOFF`, `SLIDING_WINDOW_SIZE`, and `TENSOR_ARENA_SIZE` are deployment settings in the generated manifest. Confirm them with model/device testing before distribution.
- `FEATURE_STEP_SIZE` must match feature generation and all MicroWakeWord/VAD models loaded together. The v2 schema accepts 0 through 30 ms; this workflow uses the ecosystem-standard 10 ms setting.

## Export compatibility and troubleshooting

- The Python 3.12 Colab environment currently installs TensorFlow 2.21. The export path is therefore `Python 3.12 -> TensorFlow 2.21 -> Keras ExportArchive -> TFLiteConverter`.
- A model produced during validation was structurally equivalent to known-good MicroWakeWord models: INT8 input `[1, 3, 40]`, UINT8 output `[1, 1]`, quantized streaming state variables, and the expected streaming operator set. Its deployment failure was resolved by adding the required `model` and `minimum_esphome_version` manifest fields.
- TensorFlow 2.21 assigned the unnamed `ExportArchive` endpoint generic signature names (`inputs` and `output_0`), while known-good TensorFlow 2.15 exports used names such as `input_audio` and `dense`. This did not prevent the corrected manifest from working, but it should be recorded when troubleshooting loaders that bind tensors by name rather than index.
- The signature-name behavior is directly relevant to this Colab notebook because it pins TensorFlow 2.21. It is also potentially relevant to the greater [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word) project: its current `convert_model_saved` implementation creates an unnamed `tf.TensorSpec` endpoint, so export names may vary with TensorFlow/Keras versions. This is an upstream compatibility consideration, not evidence that the upstream exporter is currently broken on its supported dependency set.
- A syntactically valid TFLite file is not sufficient deployment validation. For future failures, inspect tensor shapes, dtypes, quantization parameters, streaming resource tensors, operator support, signature names, the manifest schema, and target-device tensor-arena allocation separately.
- MicroWakeWord and OpenWakeWord classifier files use different preprocessing and tensor contracts. A MicroWakeWord model must not be diagnosed by loading it as an OpenWakeWord classifier.

## ESPHome v2 naming and publication conventions

- Keep `wake_word` human-readable; ESPHome passes this value to detection automations. Do not replace it with `RUN_NAME`.
- Keep the manifest `model` value equal to the exact case-sensitive TFLite filename. A relative filename beside the JSON is the most portable form.
- Lowercase snake_case filenames with a `_v<MODEL_REVISION>` suffix avoid punctuation differences across third-party loaders. ESPHome itself also accepts letters, numbers, periods, underscores, and hyphens for official shorthand names.
- Use short base-language codes such as `en`, `de`, or `fr` in `trained_languages`. These describe the primary languages/pronunciations in the training samples, not every language in which the phrase may be understood.
- `minimum_esphome_version` records the earliest compatible ESPHome release. Version 2 support began with ESPHome 2024.7; do not raise this field merely because training used a newer environment.
- `probability_cutoff` and `sliding_window_size` are deployment defaults and can be overridden by ESPHome YAML. Published values should come from model evaluation rather than filename or revision conventions.

## Performance records

The final summary records Python/platform information, repository commit, TensorFlow and PyTorch versions, CUDA availability, run parameters, elapsed time, stage events, and resource samples. Resource sampling defaults to 60 seconds and records system RAM availability, workspace disk use, and NVIDIA GPU utilization and memory when available.

For a comparison run, record:

- `PERFORMANCE_RUN_LABEL` and free-form `PERFORMANCE_NOTES`;
- Colab plan/tier and whether a high-RAM runtime was selected;
- CPU or GPU mode and the assigned GPU model;
- `SMOKE_TEST`, batch size, training steps, synthetic sample count, and AudioSet sample count;
- whether the run was new or resumed and the starting checkpoint step;
- wall-clock times for setup, sample generation, feature generation, training, validation, conversion, and total execution;
- peak system RAM, peak GPU memory, typical GPU utilization, final workspace disk use, interruptions, quota messages, retries, and failed downloads.

Screenshots of Colab resource graphs can supplement `run_summary.json`; note the displayed time range and the training stage active during that range.

## Observed free-tier Colab performance

Seven full-mode runs were measured on free-tier Colab between 11 and 13 August 2026. Each used Python 3.12, TensorFlow 2.21, two CPU cores, 10,000 training steps, batch size 128, 1,000 synthetic samples, and 2,000 AudioSet samples. Four runs executed on CPU and three executed on an assigned NVIDIA T4 with CUDA available.

| Effective hardware | Runs | Mean total time | Observed range | Mean training span |
| --- | ---: | ---: | ---: | ---: |
| CPU | 4 | 121.7 minutes | 111.7-135.1 minutes | 98.5 minutes |
| NVIDIA T4 | 3 | 85.3 minutes | 84.0-86.1 minutes | 66.7 minutes |

For this configuration, the T4 reduced mean total runtime by approximately 30%, from about two hours to about 85 minutes. Three same-target CPU/T4 comparisons showed total-time reductions of 25.7%, 32.7%, and 36.3%. These are operational observations rather than controlled hardware benchmarks: Colab allocation, network inputs, repository revision, and service load can affect results.

The three T4 runs recorded a median sampled GPU utilization of 2%, peak sampled utilization of 7-8%, and peak GPU memory use of approximately 1.4 GB. Resource sampling occurred once per minute and can miss short utilization spikes. The consistently low utilization and memory use indicate that this workflow is not predominantly GPU-compute-bound; preprocessing, validation, Python execution, storage, and checkpoint activity contribute materially to total duration. A more powerful GPU should not be expected to provide proportional acceleration without pipeline changes.

Across the measured runs:

- additional temporary disk use was approximately 35 GB in six comparable runs;
- peak system RAM use was approximately 4.5-5.9 GB;
- a standard 12.7 GB Colab runtime had sufficient memory headroom;
- high-RAM mode was not required;
- batch size 128 did not create observed RAM or GPU-memory pressure;
- preparation required approximately 15-21 minutes before training began;
- the 100-step checkpoint interval limited lost model progress, although regenerated runtime-only datasets and features still add recovery time after runtime replacement.

Plan for at least 40 GB of free temporary disk before a full run; 45 GB or more provides headroom for downloads, extraction, and dependency changes. CPU-only execution is practical when GPU quota is unavailable. For this measured workload, a future user should expect approximately two hours on CPU or 85 minutes on a T4, without treating either value as a completion guarantee.

`PERFORMANCE_RUN_LABEL` records user intent and must not be used alone to classify benchmark hardware. Use `cuda_available`, the installed PyTorch build, and recorded GPU metrics to determine the effective runtime. Some early summaries labelled `free-tier-gpu` were CPU executions because CUDA was unavailable.

These measurements describe notebook execution performance only. They do not establish model quality, false-accept rate, false-reject rate, pronunciation coverage, probability-cutoff suitability, microphone robustness, or target-device latency. Those properties require a fixed independent evaluation set and real deployment testing.
