import gradio as gr
import os
import numpy as np
from deepface import DeepFace

DB_PATH = "faces"

embeddings = []
image_paths = []

def build_database():
    # build_database()
    global embeddings, imag
    
    embeddings = []
    image_paths = []

    for file in os.listdir(DB_PATH):
        path = os.path.join(DB_PATH, file)

        try:
            embedding = DeepFace.represent(
                img_path=path,
                model_name="SFace",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(embedding)
            image_paths.append(path)

        except:
            pass

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_face(query_img):

    if len(embeddings) == 0:
        build_database()

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
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    results = search_face(temp_path)   # your existing function

    return jsonify({
        "matches": results[:5]
    })
demo.launch(server_name="0.0.0.0", server_port=10000)

# visual scout update
