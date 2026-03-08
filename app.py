from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo face pool with attributes
FACE_POOL = [
    {"image":"https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600","gender":"female","hair":"blonde"},
    {"image":"https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600","gender":"female","hair":"brown"},
    {"image":"https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600","gender":"female","hair":"black"},
    {"image":"https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=600","gender":"female","hair":"red"},
    {"image":"https://images.unsplash.com/photo-1520813792240-56fc4a3765a7?w=600","gender":"male","hair":"brown"},
    {"image":"https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600","gender":"male","hair":"black"}
]

@app.get("/")
def root():
    return {"status": "Visual Scout API running"}

@app.post("/api/search")
async def api_search(image: UploadFile = File(...)):

    # Read uploaded image
    await image.read()

    # Simulated detected attributes
    detected_gender = "female"
    detected_hair = random.choice(["blonde","brown","black","red"])

    results = []

    for face in FACE_POOL:

        score = 0

        # Hair color weight (60%)
        if face["hair"] == detected_hair:
            score += 60

        # Gender weight (10%)
        if face["gender"] == detected_gender:
            score += 10

        # Face similarity placeholder (30%)
        score += random.randint(0,30)

        results.append({
            "image": face["image"],
            "score": score
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "results": results,
        "matches": results
    }

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
