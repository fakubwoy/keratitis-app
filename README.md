# Keratitis AI — Web App

Flask backend + vanilla HTML/CSS/JS frontend for corneal ulcer classification.

## Project structure

```
keratitis-app/
├── app.py                  # Flask backend + inference
├── templates/
│   └── index.html          # Frontend (single file)
├── requirements.txt
├── Procfile
├── railway.toml
└── keratitis_model_final.pth   ← you add this
```

## Local test

```bash
pip install -r requirements.txt
# put your keratitis_model_final.pth in this folder
python app.py
# open http://localhost:5000
```

## Deploy to Railway

1. Download `keratitis_model_final.pth` from your Kaggle output
2. Put it in this folder
3. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "init"
   git remote add origin https://github.com/YOUR_USERNAME/keratitis-ai.git
   git push -u origin main
   ```
4. Go to railway.app → New Project → Deploy from GitHub repo
5. Select your repo — Railway auto-detects and deploys
6. Set environment variable if needed: `MODEL_PATH=keratitis_model_final.pth`
7. Done — Railway gives you a public URL

## Class names

Edit `CLASS_NAMES` in `app.py` if your model uses different class names.
Default: `['bacterial', 'fungal', 'viral']`

Check what yours actually are by running in Kaggle:
```python
print(CLASS_NAMES)
```
