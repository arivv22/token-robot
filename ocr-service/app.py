from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image, ImageFilter
import numpy as np
import re

app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang='en')


# -----------------------------
# 🔧 CLEANING OCR TEXT
# -----------------------------
def clean_text(text):
    text = text.replace('O', '0')
    text = text.replace('I', '1')
    text = text.replace('B', '8')
    text = text.replace('S', '5')
    return text


# -----------------------------
# 🔧 IMAGE PREPROCESSING
# -----------------------------
def preprocess(img):
    # grayscale
    img = img.convert("L")

    # sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # convert back to numpy
    return np.array(img)


# -----------------------------
# 🎯 EXTRACT TOKEN LOGIC
# -----------------------------
def extract_token(text):
    text = clean_text(text)

    groups = re.findall(r'[\d\s]{10,}', text)

    best = ""

    for g in groups:
        digits = re.findall(r'\d', g)
        joined = ''.join(digits)

        if len(joined) > len(best):
            best = joined

    if len(best) >= 20:
        return best[:20]

    return best

# -----------------------------
# 🔍 OCR EXECUTION (MULTI CROP)
# -----------------------------
def run_ocr(img_np):
    results = []

    h, w = img_np.shape[:2]

    crop1 = img_np[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]

    crop2 = img_np[int(h*0.35):int(h*0.7), int(w*0.4):int(w*0.95)]

    for crop in [img_np, crop1, crop2]:
        result = ocr.ocr(crop)

        text_all = ""
        for line in result:
            for word in line:
                text_all += word[1][0] + " "

        results.append(text_all)

    return results


# -----------------------------
# 🚀 MAIN ENDPOINT
# -----------------------------
@app.route('/ocr', methods=['POST'])
def do_ocr():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    # preprocessing
    img_np = preprocess(img)

    # multi OCR
    texts = run_ocr(img_np)

    best_token = ""
    best_text = ""

    # pilih hasil terbaik
    for t in texts:
        
        #filter text 
        token = extract_token(t)

        if len(token) > len(best_token):
            best_token = token
            best_text = t

    valid = len(best_token) == 20
    needs_confirmation = len(best_token) in [18, 19, 21]

    return jsonify({
        "raw_text": best_text,
        "token": best_token,
        "length": len(best_token),
        "valid": valid,
        "needs_confirmation": needs_confirmation
    })


# -----------------------------
# ▶️ RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)