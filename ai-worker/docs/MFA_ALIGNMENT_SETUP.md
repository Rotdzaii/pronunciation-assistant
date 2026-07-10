# MFA Alignment Setup

## Scope and compatibility

The worker runs the MFA 3.x `mfa align` workflow and parses long TextGrid output. The target model pair in the existing worker configuration is:

```text
Dictionary: english_us_mfa
Acoustic model: english_mfa
```

This repository does not include MFA, acoustic models, dictionaries, audio, TextGrids, or checkpoints. The current development environment used for the Phase 4A unit tests did not have an `mfa` executable, so an end-to-end MFA version must be recorded when it is run locally:

```powershell
mfa version
```

MFA 3.3 and 3.4 documentation uses the same `mfa align` workflow. MFA 4 changes are not validated by this worker yet.

## Local install

Install MFA with Conda, then download the required models:

```powershell
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
mfa --help
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa
mfa model inspect acoustic english_mfa
```

Set `ai-worker/.env` after the command succeeds:

```dotenv
ALIGNMENT_MODE=mfa
ALLOW_ALIGNMENT_FALLBACK=true
MFA_COMMAND=mfa
MFA_CONDA_ENV=aligner
MFA_DICTIONARY_PATH=english_us_mfa
MFA_ACOUSTIC_MODEL_PATH=english_mfa
MFA_ALIGNMENT_TIMEOUT_SECONDS=300
MFA_KEEP_DEBUG_ARTIFACTS=false
```

`MFA_DICTIONARY_PATH` and `MFA_ACOUSTIC_MODEL_PATH` may be MFA model names or local files. For a local dictionary file, the worker checks OOV words before invoking MFA. For an MFA registry name, it preserves the MFA-reported OOV error because the vocabulary is owned by MFA.

On Windows, either activate the Conda environment before starting the worker or set `MFA_CONDA_ENV`. Existing WSL mode remains supported with `MFA_WSL_DISTRO`, `MFA_WSL_USER`, and `MFA_WSL_BINARY`; its dictionary and acoustic-model paths must be valid inside WSL.

## Docker validation

The repository's current shared Docker image installs FFmpeg and worker dependencies but does not install MFA. Its compose file deliberately sets `ALIGNMENT_MODE=fallback`, so it must not be presented as MFA validation.

Use the official MFA container for a separate alignment readiness check. Prepare `/data/corpus/example.wav` and `/data/corpus/example.lab`, then run:

```powershell
docker image pull mmcauliffe/montreal-forced-aligner:latest
docker run --rm -it -v "${PWD}\mfa-data:/data" mmcauliffe/montreal-forced-aligner:latest bash
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa
mfa align /data/corpus english_us_mfa english_mfa /data/aligned --clean --single_speaker
```

For a worker container deployment, bake the same MFA version and downloaded models into the worker image or mount a persistent MFA model root. Do not use a transient container model cache as a production dependency. Record both `mfa version` and model metadata in the deployment validation report.

## Worker behavior

The runner creates a new temporary corpus per job. It does not overwrite source audio. Audio is decoded through the shared preprocessor and written as mono, 16 kHz, signed 16-bit PCM WAV. Silence trim, normalization, and truncation are disabled for MFA so resulting boundaries remain compatible with the original audio used by `cnn_attention_context`.

The runner returns structured categories including `mfa_not_installed`, `dictionary_missing`, `acoustic_model_missing`, `oov`, `audio_invalid`, `audio_silent`, `mfa_timeout`, `mfa_nonzero_exit`, `textgrid_missing`, and `textgrid_invalid`.

Set `MFA_KEEP_DEBUG_ARTIFACTS=true` only during local debugging. It retains a per-job temporary corpus and MFA artifacts on failure; paths are removed from webhook payloads. Leave it false in normal worker operation.
