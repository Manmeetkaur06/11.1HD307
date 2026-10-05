# Upload this project to GitHub

1. Extract `SIT307_11_1HD_GitHub_Ready.zip` on your computer.
2. Open the extracted `SIT307_11.1HD` folder. Its root contains `README.md`, `run_all.py`,
   `requirements.txt`, `data/`, `src/`, `experiments/` and `notebooks/`.
3. Create an empty GitHub repository, for example `SIT307-11.1HD`, or open the intended repository.
4. Choose **Add file → Upload files**. Drag the **contents** of `SIT307_11.1HD` into the upload area,
   so `README.md` and `run_all.py` appear at the repository root. Include the `.gitignore` file;
   on macOS, press **Command + Shift + .** in Finder to show hidden files if needed.
5. Enter a commit message such as `Add SIT307 11.1HD project`, then commit the uploaded files.

This folder contains fewer than 100 files, each smaller than 25 MiB, so it fits a single browser-upload batch.
Upload `results_snapshot.zip` as a file; keep it compressed when uploading. The outer delivery ZIP should be
extracted so the Python code and notebooks are directly visible in the repository.

For an assessor to access the repository, use the visibility and access settings required for the submission.

## Check the saved results after downloading or cloning

From the repository root, using Python 3.12 with the required packages installed:

```bash
python -m pip install -r requirements.txt
python restore_saved_results.py
python verify_results.py
```

The restore command unpacks the result CSVs and 237 prediction files. It checks the archive's SHA-256 and
preserves any different existing outputs. The verification notebook restores the snapshot automatically
when the result files are absent.

## Run all experiments again

```bash
python run_all.py
python verify_results.py
```

Training does not require the saved-results archive. Run instructions, dataset provenance and measured
findings are in `README.md`. Google Colab users can open `notebooks/SIT307_HD_Colab.ipynb`.

GitHub upload instructions: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
