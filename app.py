from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo faces (these will always return)
FACE_POOL = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600"
]

@app.get("/")
def root():
    return {"status": "Visual Scout API running"}

@app.post("/api/search")
async def api_search(image: UploadFile = File(...)):

    # read uploaded image just to confirm upload works
    contents = await image.read()

    results = []

    for i, url in enumerate(FACE_POOL):

        results.append({
            "image": url,
            "score": 80 - (i * 5)
        })

    return results
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run("app:app", host="0.0.0.0", port=port)
