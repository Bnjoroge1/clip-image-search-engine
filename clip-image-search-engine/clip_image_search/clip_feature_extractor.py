import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPFeatureExtractor:
    def __init__(self):
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def get_text_features(self, text):
        max_chunk_length = 77  # Maximum sequence length supported by the model

        # Chunk the input text to fit within the model's maximum sequence length
        text_chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

        # Process each text chunk separately and concatenate the results
        concatenated_text_features = []
        for chunk in text_chunks:
            inputs = self.processor(text=chunk, return_tensors="pt")
            inputs = inputs.to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.tolist()
            concatenated_text_features.extend(text_features)

        return concatenated_text_features

    @torch.no_grad()
    def get_image_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.tolist()
        return image_features