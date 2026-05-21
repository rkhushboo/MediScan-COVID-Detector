# MediScan COVID Detector

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-ready-brightgreen)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)
![Keras](https://img.shields.io/badge/keras-compatible-red)

## Project Overview

MediScan COVID Detector is a professional Streamlit dashboard for CNN-based COVID-19 detection using chest X-ray images. The app loads a pre-trained TensorFlow/Keras model and performs inference to classify images into:

- `Covid`
- `Normal`
- `Viral Pneumonia`

This repository is designed for deployment, inference, and reporting rather than retraining.

## Repository Structure

```text
Covid_19/
│   app.py
│   README.md
│   requirements.txt
│   .gitignore
│   .gitattributes
│   basic_cnn_tuned.h5
│   best_tuned_model.keras
│   vgg16_aug_tuned.h5
│   Richa_K_Batch_13_ANN_CNN_Mini_Project.ipynb
│
└───datasets/
    └───Covid19-dataset/
        ├───train/
        └───test/

myenv/             # local Python virtual environment (do not push)
```

## What to Push to GitHub

### Recommended
- `app.py`
- `README.md`
- `requirements.txt`
- `Richa_K_Batch_13_ANN_CNN_Mini_Project.ipynb`
- `.gitignore`
- `.gitattributes`
- small project files and utilities

### Optional but safe
- Saved model files (`.h5`, `.keras`) only if you configure Git LFS.

### Not recommended
- `myenv/`
- `datasets/` (especially if large)
- `__pycache__/`
- `.ipynb_checkpoints/`

## Setup Instructions

1. Open the project folder in VS Code.
2. Open the integrated terminal: `Terminal > New Terminal`.
3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Run the app locally:

```powershell
streamlit run app.py
```

## GitHub Push Guide (Windows)

### 1. Initialize Git locally

```powershell
cd "C:\Users\richi\DATA SCIENCE - IIT GUWAHATI\DEPLOYMENT\Covid_19"
git init
```

### 2. Add the existing GitHub remote

```powershell
git remote add origin https://github.com/rkhushboo/MediScan-COVID-Detector.git
```

### 3. Check current git status

```powershell
git status
```

### 4. Add files to staging

```powershell
git add .
```

### 5. Commit changes

```powershell
git commit -m "Initial Streamlit COVID detector app with model inference support"
```

### 6. Push to GitHub

```powershell
git branch -M main
git push -u origin main
```

## Git LFS for Large Model Files

GitHub blocks files larger than 100 MB. Your model files are currently:

- `basic_cnn_tuned.h5`
- `best_tuned_model.keras`
- `vgg16_aug_tuned.h5`

If you want to include them, install Git LFS:

```powershell
git lfs install
git lfs track "*.h5"
git lfs track "*.keras"
git add .gitattributes
git add *.h5 *.keras
git commit -m "Track model files with Git LFS"
```

Then push normally.

## Editing README.md Later

1. Open `README.md` in VS Code.
2. Edit content directly.
3. Save file.
4. Commit and push:

```powershell
git add README.md
git commit -m "Update README"
git push
```

## Best Practices for Future Updates

To update the project later:

```powershell
git status
git add <changed_files>
git commit -m "Describe what changed"
git push
```

## Common Git Errors and Fixes

- `git: command not found`:
  - Install Git for Windows from https://git-scm.com/download/win
  - Restart VS Code terminal.

- `Authentication failed`:
  - Use GitHub CLI or Git credential helper.
  - If prompted, sign in with GitHub username/password or personal access token.

- `repository not found`:
  - Verify the remote URL exactly.
  - Ensure the GitHub repo exists and you have permission.

- `large file issue`:
  - Use Git LFS for `.h5` / `.keras` files.
  - Otherwise remove those files from git and store externally.

- `push rejected`:
  - Run `git pull origin main --rebase` first, then resolve conflicts.

## Recommended GitHub Topics

- `streamlit`
- `tensorflow`
- `keras`
- `covid-19`
- `medical-imaging`
- `cnn`
- `healthcare`
- `deep-learning`

## Deployment Notes

For a professional GitHub repo:

- Keep `myenv/` and `datasets/` out of git.
- Store large models with Git LFS or external cloud storage.
- Keep README clean, with badges and quick run instructions.
- Add a screenshot or demo GIF if possible.
