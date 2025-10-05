import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class BLIPCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)

    def generate_caption(self, image_path, prompt=None, max_length=20, decoding="greedy"):
        image = Image.open(image_path).convert("RGB")

        # describing prompt
        if prompt and prompt.strip():
            final_prompt = f"Describe this image in detail, focusing on: {prompt.strip()}."
        else:
            final_prompt = None  # unconditional caption

        inputs = self.processor(
            images=image,
            text=final_prompt,
            return_tensors="pt"
        ).to(self.device)

        # decoding strategies
        if decoding == "beam":
            out = self.model.generate(**inputs, max_length=max_length, num_beams=5)
        elif decoding == "nucleus":
            out = self.model.generate(**inputs, max_length=max_length, do_sample=True, top_p=0.9)
        else:  # greedy
            out = self.model.generate(**inputs, max_length=max_length)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
