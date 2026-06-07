---
title: MNIST Digit Recognition API
emoji: 🔢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# MNIST Digit Recognition API

REST API de reconnaissance de chiffres manuscrits, construite avec **FastAPI** et déployée sur **HuggingFace Spaces** via Docker.

Le modèle est un **EfficientNetV2S** (timm) fine-tuné sur le dataset MNIST (60 000 images d'entraînement, 10 000 de test) avec une architecture de tête de classification custom. Il classifie les chiffres de **0 à 9** avec une précision >99 % sur le jeu de test.

---

## Architecture du modèle

Le modèle est défini et entraîné dans le notebook `mnist.ipynb` à la racine du dépôt.

**Backbone** : `tf_efficientnetv2_s.in21k_ft_in1k` (pré-entraîné ImageNet-21k, fine-tuné sur ImageNet-1k), chargé via `timm` avec la tête native supprimée (`num_classes=0`).

**Tête de classification** :

```
LayerNorm(num_features)
→ Linear(num_features → 512)
→ GELU
→ Dropout(0.3)
→ Linear(512 → 10)
```

**Entrainement** :

| Paramètre | Valeur |
|---|---|
| Dataset | MNIST (PNG, 28x28, niveaux de gris) |
| Input size | 32x32 (resize à l'entrainement et en inférence) |
| Batch size | 128 |
| Epochs | 40 (10 freeze + 30 unfreeze avec warmup) |
| Optimizer | AdamW avec Layer-wise LR Decay (decay=0.8) |
| Scheduler | LinearLR (warmup 3 epochs) + CosineAnnealingLR |
| Loss | CrossEntropyLoss (label smoothing=0.02) |
| Augmentations | Affine, GaussianBlur/GaussNoise, ThinPlateSpline |
| Mixed precision | AMP (autocast + GradScaler) |

**Preprocessing en inférence** (identique au `val_transform` du notebook) :

```
Resize(32, 32)
→ Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
→ ToTensorV2
```

---

## Structure du projet

```
.
├── app/
│   ├── __init__.py
│   ├── main.py           # Point d'entree FastAPI (lifespan, middleware, routes)
│   ├── config.py         # Parametres (API key, modele, preprocessing, classes)
│   ├── model_arch.py     # Architecture MNISTClassifier (identique au notebook)
│   ├── model_loader.py   # Telechargement et chargement du modele depuis Kaggle
│   ├── preprocessing.py  # Transform val_transform replique exacte du notebook
│   ├── inference.py      # Logique d'inference (softmax, top3, probabilities)
│   ├── routes.py         # Endpoints /predict, /predict/base64, /health
│   ├── schemas.py        # Modeles Pydantic (requetes et reponses)
│   └── security.py       # Dependance FastAPI pour la verification de l'API Key
├── mnist.ipynb           # Notebook d'entrainement complet
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Endpoints

### `GET /api/v1/health`

Verification de l'etat de l'API. Aucune authentification requise.

**Reponse** :

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /api/v1/predict`

Predit le chiffre manuscrit depuis un fichier image uploade.

**Authentification** : header `X-API-Key` requis.

**Corps** : `multipart/form-data` avec un champ `file`.

**Formats acceptes** : JPEG, PNG, BMP, TIFF, WebP — taille max 10 MB.

**Exemple cURL** :

```bash
curl -X POST "https://<votre-space>.hf.space/api/v1/predict" \
  -H "X-API-Key: your-api-key" \
  -F "file=@/chemin/vers/chiffre.png"
```

**Reponse** :

```json
{
  "prediction": "7",
  "confidence": 0.998712,
  "probabilities": {
    "0": 0.000012,
    "1": 0.000088,
    "2": 0.000031,
    "3": 0.000019,
    "4": 0.000044,
    "5": 0.000010,
    "6": 0.000009,
    "7": 0.998712,
    "8": 0.000055,
    "9": 0.000020
  },
  "top3": [
    {"label": "7", "confidence": 0.998712},
    {"label": "1", "confidence": 0.000088},
    {"label": "8", "confidence": 0.000055}
  ]
}
```

---

### `POST /api/v1/predict/base64`

Predit le chiffre manuscrit depuis une image encodee en Base64.

**Authentification** : header `X-API-Key` requis.

**Corps JSON** :

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
}
```

Accepte le raw Base64 ou le format data URI `data:image/...;base64,`.

**Exemple cURL** :

```bash
curl -X POST "https://<votre-space>.hf.space/api/v1/predict/base64" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgo..."}'
```

**Reponse** : identique a `/predict`.

---

## Codes d'erreur

| Code HTTP | Cause |
|---|---|
| `401 Unauthorized` | Header `X-API-Key` absent |
| `403 Forbidden` | API Key incorrecte |
| `413 Request Entity Too Large` | Image > 10 MB |
| `415 Unsupported Media Type` | Format d'image non supporte |
| `422 Unprocessable Entity` | Image illisible ou corrompue |
| `429 Too Many Requests` | Rate limit depasse (60 req/minute) |
| `500 Internal Server Error` | Erreur interne du serveur |

---

## Deploiement local

### Pre-requis

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/<votre-compte>/MNIST-Classification-API.git
cd MNIST-Classification-API

python -m venv .venv
# Windows :
.venv\Scripts\activate
# Linux/macOS :
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Editer .env et remplir API_KEY
```

Le fichier `.env` :

```env
API_KEY=votre-cle-api-secrete
```

Pour generer une cle securisee :

```bash
openssl rand -hex 32
```

### Lancement

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La documentation interactive est disponible sur `http://localhost:8000/docs`.

---

## Deploiement Docker

### Build et run

```bash
docker build -t mnist-api .

docker run -p 7860:7860 \
  -e API_KEY=votre-cle-api-secrete \
  mnist-api
```

---

## Deploiement sur HuggingFace Spaces

1. Creer un nouveau Space sur [huggingface.co/spaces](https://huggingface.co/spaces) avec le SDK **Docker**.

2. Pousser le code :

```bash
git remote add space https://huggingface.co/spaces/<votre-username>/<votre-space>
git push space main
```

3. Ajouter les secrets dans les parametres du Space :

   - `API_KEY` : votre cle API secrete

4. Au demarrage, l'API telecharge automatiquement le modele depuis Kaggle via `kagglehub`.

> **Note** : si le modele est sur Kaggle, il faut egalement configurer les credentials Kaggle (`KAGGLE_USERNAME` et `KAGGLE_KEY`) dans les secrets du Space.

---

## Modele Kaggle

Le checkpoint du modele est heberge sur Kaggle Models :

```
parfaitbotchi1/mnist/pyTorch/default/1
```

Fichier du modele : `best_model_1.pth`

Le telechargement est automatique au premier demarrage de l'API via `kagglehub.model_download()`.

---

## Variables d'environnement

| Variable | Defaut | Description |
|---|---|---|
| `API_KEY` | `change-me-in-production` | Cle API pour authentifier les requetes |
| `KAGGLE_MODEL_HANDLE` | `parfaitbotchi1/mnist/pyTorch/default/1` | Handle du modele sur Kaggle |

---

## Licence

MIT
