import os
from flask import Flask, render_template, request
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

app = Flask(__name__)

# Model and tokenizer initialization
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_path):
    try:
        # Open the image file
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        
        # Extract features and generate prediction
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0]  # Return the first prediction (if there's only one image)
    
    except Exception as e:
        return f"Error: {e}"

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None  # Default value for caption
    if request.method == "POST":
        # Handle file upload
        image_file = request.files["image"]
        
        # Save the uploaded file temporarily to process it
        temp_image_path = os.path.join(r"C:\Users\Hanisha\OneDrive\Documents\image caption\images", image_file.filename)
        image_file.save(temp_image_path)
        
        # Generate caption for the uploaded image
        caption = predict_step(temp_image_path)  # Pass the full image path to the function
    
    return render_template("frontened.html", caption=caption)


if __name__ == "__main__":
    app.run(debug=True)
