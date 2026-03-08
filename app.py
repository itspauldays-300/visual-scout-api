from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import cv2
import requests
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo face pool (same style images your app already uses)
FACE_POOL = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=800&fit=crop",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600&h=800&fit=crop",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600&h=800&fit=crop",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&h=800&fit=crop",
    "https://images.unsplash.com/photo-1520813792240-56fc4a3765a7?w=600&h=800&fit=crop",
    "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=600&h=800&fit=crop"
]


def url_to_image(url):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def get_embedding(img):
    embedding = DeepFace.represent(
        img_path=img,
        model_name="SFace",
        enforce_detection=False
    )[0]["embedding"]
    return embedding


@app.post("/api/search")
async def api_search(image: UploadFile = File(...)):

    try:

        import traceback

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(await image.read())
        temp.close()

        query_embedding = get_embedding(temp.name)

        results = []

        for url in FACE_POOL:

            face_img = url_to_image(url)

            face_embedding = get_embedding(face_img)

            similarity = cosine_similarity(
                [query

        return results

    except Exception as e:

        return {"error": str(e)}
