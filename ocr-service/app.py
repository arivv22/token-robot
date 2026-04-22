from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_digits(text):
    import re
    return ''.join(re.findall(r'\d', text))

@app.route('/ocr', methods=['POST'])
def do_ocr():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    result = ocr.ocr(img_np)

    text_all = ""
    for line in result:
        for word in line:
            text_all += word[1][0] + " "

    token = extract_digits(text_all)

    return jsonify({
        "token": token,
        "valid": len(token) == 20
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)