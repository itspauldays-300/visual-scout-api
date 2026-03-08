import requests
import os

ACCESS_KEY = "zwu9S2B2-4UPlMlFXLnatuelCiu5KRrUtqURjVrU5n4"

url = "https://api.unsplash.com/search/photos"

params = {
    "query": "portrait face person",
    "per_page": 30,
    "page": 1,
    "orientation": "portrait"
}

headers = {
    "Authorization": f"Client-ID {ACCESS_KEY}"
}

os.makedirs("faces", exist_ok=True)

count = 0

for page in range(1, 40):

    params["page"] = page

    r = requests.get(url, headers=headers, params=params)
    data = r.json()

    for photo in data["results"]:
        img_url = photo["urls"]["regular"]

        img = requests.get(img_url).content

        with open(f"faces/face_{count}.jpg", "wb") as f:
            f.write(img)

        count += 1

print("Downloaded", count, "faces")
