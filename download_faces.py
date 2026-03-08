def search_faces(query_img):

    FACE_POOL = [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=800&fit=crop",
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600&h=800&fit=crop",
        "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=600&h=800&fit=crop",
        "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&h=800&fit=crop",
        "https://images.unsplash.com/photo-1520813792240-56fc4a3765a7?w=600&h=800&fit=crop",
        "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?w=600&h=800&fit=crop"
    ]

    query_embedding = DeepFace.represent(
        img_path=query_img,
        model_name="SFace",
        enforce_detection=False
    )[0]["embedding"]

    results = []

    for url in FACE_POOL:

        emb = DeepFace.represent(
            img_path=url,
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

    return results
