

from typing import List
from PIL.Image import Image
from ingestion.embeddings.base import BaseEmbedder
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig


class SiglipEmbedder(BaseEmbedder):

    is_siglip2: bool
    is_naflex: bool
    has_get_image_features: bool
    has_get_text_features: bool

    def _load_models(self):
        quant_cfg = None

        if self.cfg.use_bnb_4bit:
            try:
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
            except Exception as e:
                print(f"ERROR: BitsAndBytes quantization failed to initialize: {e}")
                print("\nThis is likely due to Python version incompatibility.")
                raise RuntimeError(
                    "BitsAndBytes quantization is required but failed to initialize. "
                    "Please use Python 3.11-3.13 or disable quantization by setting use_bnb_4bit=False"
                ) from e

        try:
            if quant_cfg:
                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True,
                    quantization_config=quant_cfg,
                    device_map="auto",
                )
            else:
                dtype = torch.float16 if self.cfg.fp16_fallback else torch.float32

                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True,
                    torch_dtype=(torch.float16 if dtype ==
                                 torch.float16 else None),
                    device_map="auto" if self.device.type == "cuda" else None,
                )
                print(f"Loaded model: {self.image_model_name}")

        except Exception as e:
            print(str(e))
            print(
                "Primary model load failed with the above reason, trying SigLIP SO400M fallback")
            try:
                self.image_model_name = "google/siglip-so400m-patch14-384"
                self.text_model_name = "google/siglip-so400m-patch14-384"
                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name)
                self.model = AutoModel.from_pretrained(self.image_model_name)
                print(f"Loaded fallback model: {self.image_model_name}")
            except Exception as e2:
                print(
                    f"SigLIP SO400M fallback failed, using base SigLIP: {e2}")
                # Final fallback to base SigLIP
                self.image_model_name = "google/siglip-base-patch16-224"
                self.text_model_name = "google/siglip-base-patch16-224"
                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name)
                self.model = AutoModel.from_pretrained(self.image_model_name)
                print(f"Loaded final fallback model: {self.image_model_name}")
        self.has_get_image_features = hasattr(self.model, "get_image_features")
        self.has_get_text_features = hasattr(self.model, "get_text_features")

        self.is_siglip2 = "siglip2" in self.image_model_name.lower()
        self.is_naflex = "naflex" in self.image_model_name.lower()

        if self.is_siglip2:
            variant = "NaFlex" if self.is_naflex else "FixRes"
            print(f"Model type: SigLIP2 ({variant})")
        else:
            print(f"Model type: SigLIP v1")

    def encode_images(self, images: List[Image]) -> np.ndarray:
        if self.is_siglip2:
            if self.is_naflex:
                # NaFlex variant: use max_num_patches parameter
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    max_num_patches=256  
                )
            else:
                # FixRes variant: use standard parameters
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding="max_length"
                )
        else:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding="max_length"
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.has_get_image_features:
                feats = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)

                if hasattr(out, 'image_embeds'):
                    feats = out.image_embeds
                elif hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Unknown model out format")

        feats = feats.cpu().numpy()
        return feats

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        # SigLIP2 was trained with lowercased text
        if self.is_siglip2:
            texts = [text.lower() for text in texts]

        if self.is_siglip2:
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding="max_length",
                max_length=64
            )
        else:
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding="max_length"
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.has_get_text_features:
                feats = self.model.get_text_features(**inputs)
            else:
                out = self.model(**inputs)
                if hasattr(out, 'text_embeds'):
                    feats = out.text_embeds
                elif hasattr(out, 'pooler_output'):
                    feats = out.pooler_output
                elif hasattr(out, 'last_hidden_state'):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Unexpected model output format")

        feats = feats.cpu().numpy()
        return feats
