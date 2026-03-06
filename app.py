import gradio as gr
import os
import numpy as np
from deepface import DeepFace

DB_PATH = "faces"

embeddings = []
image_paths = []

def build_database():
    global embeddings, image_paths

    embeddings = []
    image_paths = []

    for file in os.listdir(DB_PATH):
        path = os.path.join(DB_PATH, file)

        try:
            embedding = DeepFace.represent(
                img_path=path,
                model_name="Facenet512",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(embedding)
            image_paths.append(path)

        except:
            pass

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_face(query_img):

    query_embedding = DeepFace.represent(
        img_path=query_img,
        model_name="Facenet512",
        enforce_detection=False
    )[0]["embedding"]

    scores = []

    for i, emb in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, emb)
        scores.append((similarity, image_paths[i]))

    scores.sort(reverse=True)

    gallery = []
    labels = []

    for score, path in scores[:10]:
        percent = round(score * 100, 2)
        gallery.append(path)
        labels.append(f"{percent}% match")

    return gallery, "\n".join(labels)

build_database()

with gr.Blocks() as demo:
    gr.Markdown("# Visual Scout Face Search")

    input_img = gr.Image(type="filepath")
    btn = gr.Button("Search")

    gallery = gr.Gallery()
    results = gr.Textbox()

    btn.click(
        search_face,
        inputs=input_img,
        outputs=[gallery, results]
    )

demo.launch()

# visual scout update