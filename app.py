from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import requests

# Import your utility functions
from utils.pdf_reader import load_pdfs
from utils.retriever import build_index
from gemini_funny_bot import get_funny_response
from image_generator import generate_image

app = Flask(__name__)
CORS(app, origins=["https://funny-mario.vercel.app"])

# Load PDFs and build index at startup
backend_dir = os.path.dirname(os.path.abspath(__file__))
pdf_paths = [
    os.path.join(backend_dir, "Alice_In_Wonderland.pdf"),
    os.path.join(backend_dir, "Gullivers_Travels.pdf"),
    os.path.join(backend_dir, "The_Arabian_Nights.pdf")
]
docs = load_pdfs(pdf_paths)
index, texts = build_index(docs)

@app.route('/api/pdf-index', methods=['GET'])
def pdf_index():
    # For demo, just return the texts (embeddings can be added if needed)
    return jsonify({
        "embeddings": [],  # You can add real embeddings if you want
        "texts": texts
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    # Search similar content
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([question])
    D, I = index.search(np.array(query_emb), k=1)
    if D[0][0] > 1.0:
        response = "I don't know, but I love cheese sandwiches! 🥪"
    else:
        text = texts[I[0][0]]
        response = get_funny_response(text)
    return jsonify({"answer": response})

@app.route('/generate-image', methods=['POST'])
def generate_image_api():
    data = request.get_json()
    prompt = data.get('prompt', '')
    img_path = generate_image(prompt[:100])
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        return jsonify({"image_base64": img_base64})
    else:
        return jsonify({"error": "Failed to generate image."}), 500

@app.route('/')
def home():
    return "Backend is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050) 