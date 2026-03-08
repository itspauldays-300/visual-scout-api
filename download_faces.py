"""
download_faces.py

Downloads large numbers of portrait-style faces from Unsplash to build
a dataset for face-similarity search (e.g., with DeepFace + FAISS).

INSTRUCTIONS
1. Get a free Unsplash API key: https://unsplash.com/developers
2. Replace YOUR_UNSPLASH_ACCESS_KEY below.
3. Run: python download_faces.py
4. Images will be saved into ./faces/

Optional:
- Change TOTAL_PAGES to control dataset size.
"""

import os
import requests
import time

ACCESS_KEY = "zwu9S2B2-4UPlMlFXLnatuelCiu5KRrUtqURjVrU5n4"

SEARCH_URL = "https://api.unsplash.com/search/photos"

QUERY = "portrait face person"
PER_PAGE = 30
TOTAL_PAGES = 40  # 40 pages * 30 images ≈ 1200 images

SAVE_FOLDER = "faces"

os.makedirs(SAVE_FOLDER, exist_ok=True)

headers = {
    "Authorization": f"Client-ID {ACCESS_KEY}"
}

count = 0

for page in range(1, TOTAL_PAGES + 1):

    params = {
        "query": QUERY,
        "page": page,
        "per_page": PER_PAGE,
        "orientation": "portrait"
    }

    response = requests.get(SEARCH_URL, headers=headers, params=params)

    if response.status_code != 200:
        print("API error:", response.text)
        break

    data = response.json()

    for photo in data["results"]:

        img_url = photo["urls"]["regular"]

        try:
            img_data = requests.get(img_url).content

            filename = os.path.join(SAVE_FOLDER, f"face_{count}.jpg")

            with open(filename, "wb") as f:
                f.write(img_data)

            print("Saved", filename)

            count += 1

        except Exception as e:
            print("Skipped image:", e)

    time.sleep(1)

print("\nDownload complete.")
print("Total images:", count)
