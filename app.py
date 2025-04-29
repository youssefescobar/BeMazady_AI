from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from PIL import Image
import io
import json
import base64

image_processor = AutoImageProcessor.from_pretrained("falconsai/nsfw_image_detection")
nsfw_model = AutoModelForImageClassification.from_pretrained("falconsai/nsfw_image_detection")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_match(text1, text2, threshold=0.6):
    encoded_input = tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = torch.dot(sentence_embeddings[0], sentence_embeddings[1]).item()
    return similarity >= threshold, similarity

def vercel_handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST allowed"})
        }

    try:
        body = request.body
        data = json.loads(body)

        image_data = base64.b64decode(data["image_base64"])
        description = data["description"]

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # NSFW detection
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = nsfw_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            nsfw_score = probs[0][1].item()

        if nsfw_score > 0.9:
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "Rejected", "reason": "NSFW content", "nsfw_score": nsfw_score})
            }

        # Caption generation
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        # Semantic matching
        matches, score = semantic_match(description, caption)

        status = "Accepted" if matches else "Flagged"

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": status,
                "nsfw_score": nsfw_score,
                "caption": caption,
                "match_score": score
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
