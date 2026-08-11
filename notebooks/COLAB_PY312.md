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
- `RUN_ACTION = "new"` requires a unique run name. Use another unique name for a fresh run of the same target or for a different target.
- `RUN_ACTION = "resume"` requires the previous run name, matching target word, metadata, and TensorFlow checkpoint.
- Full checkpoints contain model weights, Adam optimizer state, and the completed training step. The three most recent checkpoints are retained.
- Generated speech, downloaded datasets, converted audio, and RaggedMmap features remain in the Colab runtime and are regenerated after runtime replacement.
- The exported model is copied to `<RUN_NAME>.tflite` in the persistent run directory. Browser download is used only when that copy fails.
- Execution events and periodic resource samples are appended to JSONL files. The final cell writes a session-specific `run_summary_<SESSION_ID>.json` and refreshes `run_summary.json` with the latest completed session.

## External inputs and environment assumptions

- The runtime must use Python 3.12. GPU training is expected for the full configuration, but Colab hardware type, memory, storage, quotas, and session duration are not fixed.
- Piper source and model files, Hugging Face datasets, MIT room responses, FMA audio, and pre-generated negative features are network inputs. File locations and repository formats can change independently of this repository.
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
