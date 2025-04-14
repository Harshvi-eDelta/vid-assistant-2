from flask import Flask, render_template, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400

    img = request.files['image']
    # print(img)
    if img.filename == '':
        return "No selected file", 400

    img_path = os.path.join('/static/uploads', img.filename)
    # img.save(img_path)
    # print(f"File saved at: {img_path}")

    try:
        img.save(img_path)
        print(f" File saved at: {img_path}")
    except Exception as e:
        print(f" Failed to save image: {e}")

    abs_path = os.path.abspath(img_path)
    print(f"🧭 Absolute save path: {abs_path}")


    return render_template('index.html', uploaded_image=f"/{img_path}")

if __name__ == '__main__':
    app.run(debug=True)