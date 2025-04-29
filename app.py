from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModel
)
import torch.nn.functional as F

app = FastAPI()

# Load models
image_processor = AutoImageProcessor.from_pretrained("falconsai/nsfw_image_detection")
nsfw_model = AutoModelForImageClassification.from_pretrained("falconsai/nsfw_image_detection")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Helpers
def check_nsfw(image: Image.Image):
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = nsfw_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()

def generate_caption(image: Image.Image):
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_match(text1, text2, threshold=0.6):
    encoded_input = tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = text_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = torch.dot(sentence_embeddings[0], sentence_embeddings[1]).item()
    return similarity >= threshold, similarity

# API route
@app.post("/validate")
async def validate(image: UploadFile, description: str = Form(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    nsfw_score = check_nsfw(img)
    if nsfw_score > 0.9:
        return JSONResponse(content={"status": "rejected", "reason": "NSFW", "score": nsfw_score})

    caption = generate_caption(img)
    matches, similarity = semantic_match(caption, description)

    if matches:
        return {"status": "accepted", "caption": caption, "similarity": similarity}
    else:
        return {"status": "flagged", "caption": caption, "similarity": similarity}
