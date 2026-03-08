from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random
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

FACE_POOL = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600",
    "https://images.unsplash.com/photo-1520813792240-56fc4a3765a7?w=600",
    "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=600"
]

@app.get("/")
def root():
    return {"status": "Visual Scout API running"}

@app.post("/api/search")
async def api_search(image: UploadFile = File(...)):

    await image.read()

    matches = []

    for url in FACE_POOL:
        score = random.randint(60, 95)

        matches.append({
            "image": url,
            "score": score
        })

    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    return {"matches": matches}


# THIS BLOCK STARTS THE SERVER
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
