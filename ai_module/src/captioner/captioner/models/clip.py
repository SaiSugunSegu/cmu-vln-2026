"""CLIP / SigLIP image-text embedding backends used for crop-to-query matching."""
import open_clip
import torch
import torchvision.transforms.v2 as tt
from transformers import SiglipModel, SiglipProcessor


class BaseCLIP:

    def __init__(
            self,
            device = 'cuda:0'
            ):
        self.device = device

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """L2-normalised image embedding, shape (1, D)."""
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        """L2-normalised text embedding, shape (1, D)."""
        raise NotImplementedError

    @torch.no_grad()
    def get_similarity_score(
        self, 
        feature1: torch.Tensor, 
        feature2: torch.Tensor) -> float:
        logits = (feature1 @ feature2.T) * self.model.logit_scale.exp()
        if self.model.logit_bias is not None:
            logits += self.model.logit_bias
        logits = float(logits.cpu())
        return logits


class OpenCLIP(BaseCLIP):
    def __init__(
            self,
            model_id='hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
            device='cuda:0'
            ):
        super().__init__(device)
        self.model_id = model_id
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            self.model_id, 
            device=self.device, 
            precision='bf16'
            )
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

        self.preprocess = tt.Compose(
            [lambda x: x.transpose(-1, 0)]
            + self.preprocess.transforms[:-3]
            + [tt.ToDtype(torch.bfloat16, scale=True)]
            + self.preprocess.transforms[-1:])
    
    @torch.no_grad()
    def encode_image(self, image):
        img_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
        image_feature = self.model.encode_image(img_preprocessed)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        return image_feature
    
    @torch.no_grad()
    def encode_text(self, text):
        text_tokens = self.tokenizer([text]).to(self.device)
        text_feature = self.model.encode_text(text_tokens)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        return text_feature


class SigLIPHF(BaseCLIP):
    def __init__(
            self,
            model_id='google/siglip-so400m-patch14-384',
            device='cuda:0'
            ):
        super().__init__(device)
        self.model_id = model_id

        self.model = SiglipModel.from_pretrained(
            model_id,
            device_map=self.device,
            # torch_dtype=torch.bfloat16
            # attn_implementation="flash_attention_2"
            )
        self.processor = SiglipProcessor.from_pretrained(model_id)
    
    @torch.no_grad()
    def encode_image(self, image):
        image_processed = self.processor(images=image, return_tensors='pt', padding='max_length').to(self.device)
        features = self.model.get_image_features(**image_processed)
        features /= features.norm(dim=-1, keepdim=True)
        return features
    
    @torch.no_grad()
    def encode_text(self, text):
        text_processed = self.processor(text=text, return_tensors='pt', padding='max_length').to(self.device)
        features = self.model.get_text_features(**text_processed)
        features /= features.norm(dim=-1, keepdim=True)
        return features


def main(argv=None):
    """Score one local image against candidate labels — for tuning clip_threshold.

    Reads from disk rather than a URL: the point is to score the crops this pipeline
    actually produces, so a demo that fetched a COCO image over HTTP would be tuning
    against the wrong distribution.

      python -m captioner.models.clip /data/crops/3_sofa/crop.png sofa "red sofa"
    """
    import argparse

    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument('image', help='Path to a local RGB image.')
    parser.add_argument('labels', nargs='+', help='Candidate labels to score.')
    parser.add_argument('--model_type', default='clip', choices=['clip', 'siglip'])
    args = parser.parse_args(argv)

    clip_model = SigLIPHF() if args.model_type == 'siglip' else OpenCLIP()

    # HWC uint8, matching what Captioner.get_crop() hands encode_image() in
    # production (captioning_backend.py:152) so scores here are comparable.
    image = torch.from_numpy(np.asarray(Image.open(args.image).convert('RGB')))

    image_feature = clip_model.encode_image(image)
    similarities = [
        clip_model.get_similarity_score(image_feature, clip_model.encode_text(label))
        for label in args.labels
    ]

    for label, score in zip(args.labels, similarities):
        print(f'{score:8.3f}  {label}')
    print('Softmax:', torch.softmax(torch.tensor(similarities), dim=0).tolist())


if __name__ == "__main__":
    main()
