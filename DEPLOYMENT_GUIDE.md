# Guide de Deploiement et Documentation API
## MNIST Digit Recognition API — EfficientNetV2S

---

## 1. Structure du projet

```
MNIST-Classification-API/
├── Dockerfile
├── requirements.txt
├── README.md
├── DEPLOYMENT_GUIDE.md
├── .env.example
├── .gitignore
└── app/
    ├── __init__.py
    ├── main.py           # Point d'entree FastAPI (lifespan, middleware)
    ├── config.py         # Parametres (pydantic-settings)
    ├── security.py       # Verification API Key via header X-API-Key
    ├── model_arch.py     # Architecture MNISTClassifier (identique au notebook)
    ├── model_loader.py   # Telechargement Kaggle + cache memoire
    ├── preprocessing.py  # val_transform replique exacte du notebook
    ├── inference.py      # Moteur d'inference (softmax, top3)
    ├── routes.py         # Endpoints FastAPI
    └── schemas.py        # Modeles Pydantic I/O
```

---

## 2. Deploiement sur HuggingFace Spaces

### Etape 1 — Modele sur Kaggle

Le modele est heberge sur Kaggle Models :

```
parfaitbotchi1/mnist/pyTorch/default/1
```

Fichier : `best_model_1.pth`

L'API telecharge ce fichier automatiquement au demarrage via `kagglehub`.
Aucune action manuelle n'est requise pour le modele.

### Etape 2 — Creer le Space HuggingFace

Via la CLI :

```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli repo create mnist-digit-api --type space --space-sdk docker
```

Ou via https://huggingface.co/new-space :
- SDK : **Docker**
- Visibility : Public ou Private selon ton usage

### Etape 3 — Pousser le code

```bash
# Cloner le Space vide cree
git clone https://huggingface.co/spaces/TON_USERNAME/mnist-digit-api
cd mnist-digit-api

# Copier les fichiers du projet
cp -r /chemin/vers/MNIST-Classification-API/. .

# Pousser
git add .
git commit -m "feat: initial deployment"
git push
```

### Etape 4 — Configurer les secrets

Sur HuggingFace : `Ton Space -> Settings -> Variables and secrets -> New secret`

| Name | Value | Obligatoire |
|---|---|---|
| `API_KEY` | Cle generee avec `openssl rand -hex 32` | Oui |
| `KAGGLE_USERNAME` | Ton nom d'utilisateur Kaggle | Oui |
| `KAGGLE_KEY` | Ton token API Kaggle | Oui |

Pour obtenir ton token Kaggle : https://www.kaggle.com/settings -> API -> Create New Token.
Cela telecharge un fichier `kaggle.json` contenant `username` et `key`.

> Le Space ne demarrera pas correctement sans les credentials Kaggle, car le modele
> est telecharge depuis Kaggle au premier lancement.

---

## 3. Documentation API pour les developpeurs

### URL de base

```
https://TON_USERNAME-mnist-digit-api.hf.space/api/v1
```

### Authentification

Toutes les requetes de prediction necessitent le header :
```
X-API-Key: <votre-cle-api>
```

---

### GET `/api/v1/health`

Verifie que l'API et le modele sont operationnels. **Pas d'auth requise.**

**Reponse 200 :**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### POST `/api/v1/predict`

Predit le chiffre manuscrit (0-9) depuis un fichier image uploade.

**Headers :**
```
X-API-Key: <votre-cle>
Content-Type: multipart/form-data
```

**Body :** `form-data`

| Champ | Type | Description |
|---|---|---|
| `file` | File | Image du chiffre (JPEG/PNG/BMP/TIFF/WebP, max 10 MB) |

Les images en niveaux de gris et en RGB sont toutes deux acceptees.

**Reponse 200 :**
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

**Exemple cURL :**
```bash
curl -X POST "https://TON_USERNAME-mnist-digit-api.hf.space/api/v1/predict" \
  -H "X-API-Key: votre-cle-api" \
  -F "file=@/chemin/vers/chiffre.png"
```

**Exemple JavaScript (fetch) :**
```javascript
const formData = new FormData();
formData.append('file', imageFile); // imageFile = objet File du <input>

const response = await fetch(
  'https://TON_USERNAME-mnist-digit-api.hf.space/api/v1/predict',
  {
    method: 'POST',
    headers: { 'X-API-Key': 'votre-cle-api' },
    body: formData,
  }
);
const result = await response.json();
console.log(result.prediction);   // "7"
console.log(result.confidence);   // 0.998712
console.log(result.top3);         // [{label: "7", confidence: 0.998712}, ...]
```

**Exemple Python (requests) :**
```python
import requests

with open("chiffre.png", "rb") as f:
    response = requests.post(
        "https://TON_USERNAME-mnist-digit-api.hf.space/api/v1/predict",
        headers={"X-API-Key": "votre-cle-api"},
        files={"file": ("chiffre.png", f, "image/png")},
    )

result = response.json()
print(result["prediction"])   # "7"
print(result["confidence"])   # 0.998712
```

---

### POST `/api/v1/predict/base64`

Predit le chiffre manuscrit depuis une image encodee en Base64.

**Headers :**
```
X-API-Key: <votre-cle>
Content-Type: application/json
```

**Body JSON :**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
}
```

Accepte le raw Base64 ou le format data URI `data:image/...;base64,`.

**Reponse 200 :** identique a `/predict`.

**Exemple JavaScript :**
```javascript
// Convertir un File en base64
function fileToBase64(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result); // inclut le prefixe data:...
    reader.readAsDataURL(file);
  });
}

const base64 = await fileToBase64(imageFile);

const response = await fetch(
  'https://TON_USERNAME-mnist-digit-api.hf.space/api/v1/predict/base64',
  {
    method: 'POST',
    headers: {
      'X-API-Key': 'votre-cle-api',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: base64 }),
  }
);
const result = await response.json();
console.log(result.prediction); // "7"
```

**Exemple Python (requests) :**
```python
import base64, requests

with open("chiffre.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://TON_USERNAME-mnist-digit-api.hf.space/api/v1/predict/base64",
    headers={
        "X-API-Key": "votre-cle-api",
        "Content-Type": "application/json",
    },
    json={"image": b64},
)
print(response.json()["prediction"])  # "7"
```

---

## 4. Codes d'erreur

| Code HTTP | Signification |
|---|---|
| `200` | Succes |
| `401` | Header `X-API-Key` manquant |
| `403` | API Key invalide |
| `413` | Image trop volumineuse (> 10 MB) |
| `415` | Format d'image non supporte |
| `422` | Image corrompue ou impossible a decoder |
| `429` | Trop de requetes (limite : 60/minute par IP) |
| `500` | Erreur interne du serveur |

**Format d'erreur :**
```json
{
  "detail": "Description de l'erreur",
  "code": "INTERNAL_ERROR"
}
```

---

## 5. Classes du modele

| Classe | Chiffre represente |
|---|---|
| `"0"` | Zero |
| `"1"` | Un |
| `"2"` | Deux |
| `"3"` | Trois |
| `"4"` | Quatre |
| `"5"` | Cinq |
| `"6"` | Six |
| `"7"` | Sept |
| `"8"` | Huit |
| `"9"` | Neuf |

---

## 6. Deploiement Docker local

```bash
# Build
docker build -t mnist-api .

# Run
docker run -p 7860:7860 \
  -e API_KEY=votre-cle-api \
  -e KAGGLE_USERNAME=ton-username-kaggle \
  -e KAGGLE_KEY=ton-token-kaggle \
  mnist-api
```

L'API est disponible sur `http://localhost:7860`.
Documentation interactive : `http://localhost:7860/docs`

---

## 7. Notes techniques

- **Modele** : EfficientNetV2S (`tf_efficientnetv2_s.in21k_ft_in1k`) fine-tune sur MNIST.
- **Preprocessing** : Resize 32x32, normalisation MNIST `mean=0.1307` / `std=0.3081` (canaux RGB identiques).
- **Device** : CPU (HF Spaces tier gratuit). Latence estimee : 300ms - 1s par image.
- **Rate limit** : 60 requetes/minute par IP.
- **Documentation interactive Swagger** : `https://TON_USERNAME-mnist-digit-api.hf.space/docs`
- **Documentation ReDoc** : `https://TON_USERNAME-mnist-digit-api.hf.space/redoc`
