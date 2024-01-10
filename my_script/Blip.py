from typing import Any
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIP:
    def __init__(self) -> None:
        vision = '/mnt/nfs/file_server/public/mingjiahui/models/Salesforce--blip-image-captioning-base'     # "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(vision)
        self.model = BlipForConditionalGeneration.from_pretrained(vision).to("cuda")
    
    def __call__(self, image) -> Any:
        assert isinstance(image, Image.Image)
        # unconditional image captioning
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        prompt = self.processor.decode(out[0], skip_special_tokens=True)
        
        return prompt