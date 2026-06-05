# MFA Local Validation

## Purpose

This workflow validates whether a local Montreal Forced Aligner setup can align one local audio file and transcript, generate a TextGrid, and parse that TextGrid into the existing AI Worker alignment contract.

It is a local validation step only. It does not install MFA, download models, train models, or change worker runtime behavior.

## Prerequisites

- Local MFA command available as `mfa`, another command path, or through a conda environment.
- MFA pronunciation dictionary model name or local file path.
- MFA acoustic model name or local file path.
- Local audio file provided by the developer.
- Transcript matching the audio prompt.
- AI Worker virtual environment available at `ai-worker/.venv`.

MFA is used because it is a common forced-alignment tool that can align known transcript text to audio and emit TextGrid word/phone intervals. Those intervals are needed before reliable phone-level scoring such as GOP/CaGOP.

## Required Local Setup

Provide paths through CLI arguments or environment variables:

```powershell
$env:MFA_DICTIONARY_PATH = "C:\path\to\dictionary.dict"
$env:MFA_ACOUSTIC_MODEL_PATH = "C:\path\to\acoustic-model.zip"
```

Dictionary and acoustic model arguments may also be MFA model names, for example `english_us_mfa` and `english_mfa`.

Audio should be a local WAV file. Prefer 16 kHz mono WAV if the local MFA setup requires that format. The validation script does not convert audio unless an existing local setup has already prepared it.

On Windows, directly invoking `mfa.exe` can fail when DLL paths are not resolved by the shell, including `libsndfile.dll` / `soundfile` loading failures. Prefer checking MFA through conda:

```powershell
conda run -n mfa mfa version
```

## Dry Run

Use dry-run first to check configuration without running MFA:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\validate_mfa_local_alignment.py --dry-run --transcript Architecture
```

Dry-run prints MFA availability, configured dictionary/model status, and setup instructions when something is missing.

## Real Validation

Recommended Windows command when MFA is installed in a conda environment:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\validate_mfa_local_alignment.py `
  --conda-env mfa `
  --audio-path "C:\path\to\local_sample.wav" `
  --transcript "Architecture" `
  --dictionary-path english_us_mfa `
  --acoustic-model-path english_mfa
```

You can also run real validation with a direct MFA command and local model files:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\validate_mfa_local_alignment.py `
  --audio-path "C:\path\to\local_sample.wav" `
  --transcript "Architecture" `
  --dictionary-path "C:\path\to\dictionary.dict" `
  --acoustic-model-path "C:\path\to\acoustic-model.zip"
```

Optional arguments:

- `--conda-env mfa` to run MFA as `conda run -n mfa mfa`.
- `--mfa-command "C:\path\to\mfa.exe"` to override the direct MFA command, or the command used after `conda run -n <env>`.
- `--output-dir "C:\path\to\scratch"` to choose where the temporary validation folder is created.
- `--keep-temp` to keep generated local files for inspection.
- `--no-parse-textgrid` to skip parser validation.

## Expected Output

A successful run should print:

- MFA command availability.
- MFA exit code.
- local TextGrid path.
- parsed `alignment_status`.
- parsed `alignment_method`.
- word segment count.
- phone segment count.
- sample word and phone timestamp intervals.

Alignment is timing evidence only. It is not pronunciation correctness and should not be reported as a GOP/CaGOP score.

## TextGrid Parsing

The script uses `ai-worker/app/alignment/textgrid_parser.py`. The parser supports common MFA/Praat interval tier names:

- `words` or `word`
- `phones` or `phone`

Parsed results are mapped into the existing alignment contract with `alignment_method=mfa`, word segments, phone segments, timestamps, and metadata marking forced alignment.

## Cleanup Behavior

By default, the script deletes its temporary MFA corpus and output directory after the run. Use `--keep-temp` only for local debugging.

Never commit generated validation artifacts, including:

- audio files
- `.wav`, `.webm`, `.m4a`
- `.TextGrid`
- temporary MFA folders
- dictionaries or acoustic models if they are local runtime artifacts

## Troubleshooting

`MFA command not found`

Install MFA locally outside this repository, pass the command path using `--mfa-command`, or use `--conda-env mfa` when MFA is installed in a conda environment.

`Direct mfa.exe fails on Windows with DLL loading errors`

Use `conda run -n mfa mfa version` to verify MFA and pass `--conda-env mfa` to the validation script. This lets conda set the environment DLL paths before MFA starts.

`Dictionary model/path is not configured` or `Acoustic model/path is not configured`

Set `MFA_DICTIONARY_PATH` / `MFA_ACOUSTIC_MODEL_PATH` or pass CLI arguments.

`Audio path is required for real alignment`

Provide a local audio path. Do not commit the audio file.

`Audio format unsupported`

Convert the sample outside this script to a format supported by the local MFA setup, preferably WAV 16 kHz mono.

`Transcript mismatch`

Use the exact prompt sentence or target word spoken in the audio. A word-only transcript may align less robustly than a full prompt.

`No TextGrid generated`

Check MFA stdout/stderr, dictionary coverage, acoustic model compatibility, audio duration, and transcript normalization.

## Safety Notes

- Do not commit audio files.
- Do not commit generated TextGrid files.
- Do not commit temporary MFA folders.
- Do not commit checkpoints, `.env` files, secrets, or signed URLs.
- Alignment is not pronunciation correctness.
- Fallback alignment remains available for development and degraded operation, but it is approximate and not real forced alignment.
