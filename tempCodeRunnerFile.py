from flask import Flask, render_template, request, url_for
import os
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from datetime import datetime
from gtts import gTTS  
from playsound import playsound  
from translate import translate_caption  # Import the translation function

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
AUDIO_FOLDER = "static/audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

gen_kwargs = {"max_length": 16, "num_beams": 4}

def predict_caption(image_path):
    try:
        with Image.open(image_path).convert("RGB") as image:
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            output_ids = model.generate(pixel_values, **gen_kwargs)

        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_audio(caption):
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".mp3"
    audio_path = os.path.join(AUDIO_FOLDER, filename)

    tts = gTTS(text=caption, lang="en", slow=False)
    tts.save(audio_path)

    return url_for("static", filename=f"audio/{filename}")

@app.route("/", methods=["GET", "POST"])
def index():
    caption = ""
    image_url = None  
    audio_url = None  
    translated_caption = ""
    selected_language = "en"

    if request.method == "POST":
        file = request.files.get("image")
        selected_language = request.form.get("language", "en")

        if file and file.filename:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            caption = predict_caption(image_path)
            image_url = url_for("static", filename=f"uploads/{file.filename}")

            if caption:
                audio_url = generate_audio(caption)

                if selected_language != "en":
                    translated_caption = translate_caption(caption, selected_language)

    return render_template(
        "index.html", caption=caption, translated_caption=translated_caption, 
        image_url=image_url, audio_url=audio_url, selected_language=selected_language
    )

if __name__ == "__main__":
    app.run(threaded=True, debug=False)
