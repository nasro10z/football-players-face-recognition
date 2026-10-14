# Football Player Face Recognition

A first deep-learning project that recognizes seven football players from a face photo. It combines MTCNN face detection, FaceNet embeddings, a trained Keras classifier, and a cosine-distance check that can reject faces outside the known classes.

The included Streamlit app lets you upload an image and view the predicted player, classifier confidence, and top predictions. It reports **Unknown person** when the face is too far from the stored player centroids, and explains when no face is detected.

## Players

- Cristiano Ronaldo
- Kylian Mbappé
- Lamine Yamal
- Lionel Messi
- Neymar
- Riyad Mahrez
- Zinedine Zidane

## Run the app

Use Python 3.10 or 3.11. From the project directory, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install and launch:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app runs locally in your browser. PyTorch uses CUDA when a compatible GPU setup is available; otherwise inference runs on CPU. PyTorch GPU installation requirements vary by machine, so consult the [official PyTorch install selector](https://pytorch.org/get-started/locally/) if you need CUDA support.

On first launch, FaceNet and MTCNN may download pretrained weights into the local PyTorch cache, so an internet connection is required once.

## Command-line prediction

Put test images in `test_images/`, then run:

```bash
python predict.py
```

## Project files

- `streamlit_app.py`: browser-based image upload and prediction interface.
- `predict.py`: face detection, embedding extraction, unknown-person rejection, and classification.
- `football_player_classifier.keras`, `classes.npy`, `player_centroids.npy`: trained inference artifacts required by the app.
- `train.ipynb`: model training notebook.
- `pipeline.py`: image collection and face-cropping pipeline.
- `requirements.txt`: Python dependencies for the app and project tools.

The app processes uploads in memory and does not save them. Training datasets and test images are excluded from Git by default. The pipeline records image attribution metadata where available; review the rights and license of all images and datasets before redistributing them. Model predictions are experimental and should not be treated as identity verification.

## Publish on GitHub

Suggested repository name: `football-player-face-recognition`.

Before publishing, review the repository contents and make sure you have permission to distribute any included training data, test images, and model artifacts. This repository intentionally has no license file yet; choose a license that matches how you want others to use your code before presenting it as open source. After creating an empty repository on GitHub, connect and push this project with:

```bash
git add .
git commit -m "Add football player face recognition app"
git branch -M main
git remote add origin https://github.com/<your-username>/football-player-face-recognition.git
git push -u origin main
```
