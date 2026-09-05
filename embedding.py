from deepface import DeepFace
import json
import os

IMAGE_PATH = "registered_faces/Sattam.jpg"
OUTPUT_PATH = "registered_faces/Sattam_embedding.json"

print("1) Starting...")

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found")
    raise SystemExit

print("2) Image found")
print("3) Creating FaceNet512 embedding...")

result = DeepFace.represent(
    img_path=IMAGE_PATH,
    model_name="Facenet512",
    detector_backend="skip",
    enforce_detection=False
)

embedding = result[0]["embedding"]

with open(OUTPUT_PATH, "w") as f:
    json.dump(embedding, f)

print("4) Embedding created successfully")
print("Saved to:", OUTPUT_PATH)
print("Embedding length:", len(embedding))