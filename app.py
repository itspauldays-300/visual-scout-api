from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import tempfile
import requests
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DEMO FACE POOL (Unsplash)
# -------------------------

FACE_POOL = [
"https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1517841905240-472988babdf9?w=600&h=800&fit=crop",
"https://images.unsplash.com/photo-1520813792240-56fc4a3765a7?w=600&h=800&fit=crop"
]

CACHE_DIR = "faces"
os.makedirs(CACHE_DIR, exist_ok=True)


# -------------------------
# DOWNLOAD FACE
# -------------------------

def download_face(url, index):

    path = f"{CACHE_DIR}/face_{index}.jpg"

    if not os.path.exists(path):

        r = requests.get(url)

        with open(path, "wb") as f:
            f.write(r.content)

    return path


# -------------------------
# SIMILARITY
# -------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------
# SEARCH
# -------------------------

def search_face(query_img):

    query_embedding = DeepFace.represent(
        img_path=query_img,
        model_name="SFace",
        enforce_detection=False
    )[0]["embedding"]

    results = []

    for i, url in enumerate(FACE_POOL):

        face_path = download_face(url, i)

        emb = DeepFace.represent(
            img_path=face_path,
            model_name="SFace",
            enforce_detection=False
        )[0]["embedding"]

        score = cosine_similarity(query_embedding, emb)

        percent = round(score * 100, 2)

        results.append({
            "image": url,
            "score": percent
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:6]


# -------------------------
# API
# -------------------------

@app.post("/api/search")

async def api_search(image: UploadFile = File(...)):

    temp_path = os.path.join(tempfile.gettempdir(), image.filename)

    with open(temp_path, "wb") as f:
        f.write(await image.read())

    results = search_face(temp_path)

    return {"results": results}


# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(app, host="0.0.0.0", port=port)
