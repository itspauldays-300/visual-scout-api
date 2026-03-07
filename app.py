import gradio as gr
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import requests
import uvicorn

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

CACHE_DIR = "face_cache"

embeddings = []
image_paths = []


# -------------------------
# DOWNLOAD FACES
# -------------------------

def download_faces():

    os.makedirs(CACHE_DIR, exist_ok=True)

    paths = []

    for i, url in enumerate(FACE_POOL):

        file_path = f"{CACHE_DIR}/face_{i}.jpg"

        if not os.path.exists(file_path):

            r = requests.get(url)

            with open(file_path, "wb") as f:
                f.write(r.content)

        paths.append(file_path)

    return paths


# -------------------------
# BUILD DATABASE
# -------------------------

def build_database():

    global embeddings, image_paths

    embeddings = []
    image_paths = []

    faces = download_faces()

    for path in faces:

        try:

            emb = DeepFace.represent(
                img_path=path,
                model_name="SFace",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(emb)
            image_paths.append(path)

        except:
            pass

    print("Faces loaded:", len(image_paths))


# -------------------------
# SIMILARITY
# -------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------
# SEARCH FUNCTION
# -------------------------

def search_face(query_img):

    query_embedding = DeepFace.represent(
        img_path=query_img,
        model_name="SFace",
        enforce_detection=False
    )[0]["embedding"]

    scores = []

    for i, emb in enumerate(embeddings):

        similarity = cosine_similarity(query_embedding, emb)

        scores.append((similarity, image_paths[i]))

    scores.sort(reverse=True)

    results = []

    for score, path in scores[:6]:

        percent = round(score * 100, 2)

        results.append({
            "image": path,
            "score": percent
        })

    return results


# Build database at startup
build_database()


# -------------------------
# GRADIO UI
# -------------------------

with gr.Blocks() as demo:

    gr.Markdown("# Visual Scout Face Search")

    input_img = gr.Image(type="filepath")
    btn = gr.Button("Search")

    gallery = gr.Gallery()
    results_box = gr.Textbox()

    def ui_search(img):

        results = search_face(img)

        images = [r["image"] for r in results]
        labels = "\n".join([f'{r["score"]}% match' for r in results])

        return images, labels

    btn.click(
        ui_search,
        inputs=input_img,
        outputs=[gallery, results_box]
    )


# -------------------------
# FASTAPI SERVER
# -------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = gr.mount_gradio_app(app, demo, path="/")


# -------------------------
# API ENDPOINT
# -------------------------

@app.post("/api/search")
async def api_search(image: UploadFile = File(...)):

    temp_path = os.path.join(tempfile.gettempdir(), image.filename)

    with open(temp_path, "wb") as f:
        f.write(await image.read())

    results = search_face(temp_path)

    return {"results": results}


# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(app, host="0.0.0.0", port=port)
