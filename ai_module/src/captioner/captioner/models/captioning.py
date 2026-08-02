from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    PaliGemmaProcessor, 
    PaliGemmaForConditionalGeneration,
    GemmaTokenizerFast,
    SiglipVisionModel,
    SiglipImageProcessor,
    SiglipModel,
    SiglipTextModel,
    SiglipTokenizer,
    StopStringCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)
import torch
from PIL import Image
import requests
from typing import List, Optional
import cv2
import imageio.v3 as iio
import json
import os
import re
from time import time
try:
    from vllm import LLM, SamplingParams, RequestOutput
except ImportError as e:
    print("VLLM not installed, ignoring dependency.")


def _hub_offline() -> bool:
    '''True when we must not touch the network (robot / offline deploy).'''
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CAPTIONER_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def _from_pretrained_kwargs() -> dict:
    '''
    Hugging Face from_pretrained("org/name") still contacts the Hub by default
    to resolve the latest revision, even when weights are cached. On a robot with
    no internet that hangs or fails — force local-only loads.
    '''
    kwargs = {}
    if _hub_offline():
        kwargs["local_files_only"] = True
    return kwargs


class CaptioningModel:
    def __init__(self):
        pass

    def generate_captions(self, images: list[torch.Tensor] | torch.Tensor) -> list[str]:
        pass

    @staticmethod
    def split_into_batches(lst, batch_size: int):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]


class PaliGemmaHFBackend(CaptioningModel):
    def __init__(
            self,
            model_id = "google/paligemma2-3b-ft-docci-448",
            quantization: Optional[str] = "int4",
            batch_size: Optional[int] = 16
            ):

        self.model_id = model_id
        self.batch_size = batch_size
        
        if quantization == "int8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization is None:
            bnb_config = None
        else:
            raise ValueError('Invalid quantization. Must either be int8, int4, or None.')


        self.processor = PaliGemmaProcessor.from_pretrained(
            model_id, **_from_pretrained_kwargs())
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
            **_from_pretrained_kwargs(),
            )

        self.stopping_criteria = StoppingCriteriaList([StopStringCriteria(self.processor.tokenizer, '.')])

        self.prompt = "<image>caption en "

    def generate_captions(self, images):

        captions = []

        for batch in self.split_into_batches(images, self.batch_size): 

            start_time = time()  

            inputs = self.processor(
                batch, 
                [self.prompt]*len(batch), 
                self.prompt,
                return_tensors="pt").to("cuda")


            output = self.model.generate(
                **inputs, 
                # max_new_tokens=200
                stopping_criteria=self.stopping_criteria
                )

            caption_batch = self.processor.batch_decode(output, skip_special_tokens=True)
            caption_batch = [caption[len("caption en \n"):] for caption in caption_batch]

            captions.extend(caption_batch)


        return captions


class QwenVLHFBackend(CaptioningModel):
    '''
    Shared HF generate path for the Qwen-VL family. Qwen2.5-VL and Qwen3-VL differ
    only in checkpoint and model class, so keeping one implementation means the two
    backends are directly comparable (same prompt, pixel budget, and decode).
    '''

    default_model_id: str = None

    def __init__(
            self,
            model_id = None,
            quantization: Optional[str] = "int4",
            batch_size: Optional[int] = 16,
            dtype = torch.bfloat16,
            max_new_tokens: int = 200,
            # Cap vision tokens for crop captioning; Qwen defaults can be much higher.
            min_pixels: Optional[int] = 4 * 28 * 28,
            max_pixels: Optional[int] = 640 * 28 * 28,
            ):

        self.model_id = model_id if model_id is not None else self.default_model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.last_run_stats = {}

        if quantization not in ("int8", "int4", None):
            raise ValueError('Invalid quantization. Must either be int8, int4, or None.')

        # Checkpoints such as *-AWQ / *-GPTQ carry their own quantization config;
        # layering BitsAndBytes on top of them fails to load.
        self.prequantized_checkpoint = any(
            tag in self.model_id.lower() for tag in ("awq", "gptq"))

        if self.prequantized_checkpoint:
            bnb_config = None
            self.quantization = "checkpoint"
            # AWQ CUDA kernels are fp16-only; transformers otherwise warns and casts.
            if dtype == torch.bfloat16:
                dtype = torch.float16
        elif quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.quantization = quantization
        elif quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            self.quantization = quantization
        else:
            bnb_config = None
            self.quantization = None

        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **_from_pretrained_kwargs(),
            )

        # Note: int4 BitsAndBytes re-quantizes the full checkpoint on every process
        # start (slow). Prefer a pre-quantized local checkpoint for robot deploy.
        model_kwargs = {
            "device_map": "auto",
            "dtype": dtype,
            **_from_pretrained_kwargs(),
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        load_start = time()
        self.model = self.model_class().from_pretrained(
            self.model_id,
            **model_kwargs,
            )
        self.load_seconds = time() - load_start

        self.prompt = "Describe the {obj} in this image, using properties like color, material, shape, affordances, and other meaningful attributes. Provide the response in this format: “The <object name> is <color>, <material>, <shape>."
        # Ask for a bare integer so challenge numerical answers need minimal parsing.
        self.vqa_prompt = (
            "Answer the question about this image with a single integer only. "
            "Do not include units, words, or explanation.\nQuestion: {question}"
        )
        # Pass-through text for extract / attribute prompts (no integer constraint).
        self.freeform_vqa_prompt = "{question}"

    @staticmethod
    def model_class():
        raise NotImplementedError

    @staticmethod
    def to_pil(image) -> Image.Image:
        '''
        Crops arrive as HWC uint8 RGB tensors on the GPU. Converting explicitly avoids
        relying on the image processor to infer the channel dimension and device.
        '''
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            image = image.detach().contiguous().cpu().numpy()
        return Image.fromarray(image)

    @staticmethod
    def extract_integer(text: str) -> Optional[int]:
        '''Parse the first integer from a VLM reply (handles "4", "There are 4 pillows").'''
        if text is None:
            return None
        match = re.search(r"-?\d+", text.replace(",", ""))
        if match is None:
            return None
        return int(match.group(0))

    def build_prompts(self, pil_images, names: list[str]) -> list[str]:
        messages = [[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.prompt.format(obj=name)},
                ],
            }
        ] for image, name in zip(pil_images, names)]

        return [self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        ) for message in messages]

    def build_vqa_prompts(
            self,
            pil_images,
            questions: list[str],
            *,
            freeform: bool = False,
            ) -> list[str]:
        template = self.freeform_vqa_prompt if freeform else self.vqa_prompt
        messages = [[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": template.format(question=question),
                    },
                ],
            }
        ] for image, question in zip(pil_images, questions)]

        return [self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        ) for message in messages]

    def answer_questions(
            self,
            images,
            questions: list[str],
            max_new_tokens: Optional[int] = None,
            *,
            freeform: bool = False,
            ) -> list[str]:
        '''Visual question answering for one image/question pair each.

        freeform=False (default): wrap with the integer-only challenge prompt.
        freeform=True: send the question text as-is (extract / attribute captions).
        '''
        if len(images) != len(questions):
            raise ValueError(
                f"images ({len(images)}) and questions ({len(questions)}) length mismatch")

        pil_images = [self.to_pil(image) for image in images]
        prompts = self.build_vqa_prompts(pil_images, questions, freeform=freeform)
        token_budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        answers = []
        for batch in self.split_into_batches(list(zip(pil_images, prompts)), self.batch_size):
            image_batch = [b[0] for b in batch]
            prompt_batch = [b[1] for b in batch]
            answer_batch, _, _ = self.run_batch(image_batch, prompt_batch, token_budget)
            answers.extend(answer_batch)
        return answers

    def answer_numerical(
            self,
            images,
            questions: list[str],
            max_new_tokens: int = 16,
            ) -> list[Optional[int]]:
        '''VQA that returns parsed integers (None when the model reply has no digit).'''
        answers = self.answer_questions(
            images, questions, max_new_tokens=max_new_tokens)
        return [self.extract_integer(a) for a in answers]

    def run_batch(self, image_batch, prompt_batch, max_new_tokens: int):
        inputs = self.processor(
            images=image_batch,
            text=prompt_batch,
            padding=True,
            return_tensors="pt").to("cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            )

        # Left padding means every sequence in the batch shares the same prompt
        # length, so trimming by input length is safe and avoids parsing the
        # chat template out of the decoded string.
        generated = output[:, inputs.input_ids.shape[1]:]

        caption_batch = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        return [caption.strip() for caption in caption_batch], inputs, generated

    def warmup(self, images, names: list[str], max_new_tokens: int = 8):
        '''
        The first generate call pays for CUDA kernel selection and cuBLAS workspace
        allocation, which would otherwise be charged to the first timed batch.
        '''
        if not images:
            return
        pil_images = [self.to_pil(images[0])]
        prompts = self.build_prompts(pil_images, names[:1])
        self.run_batch(pil_images, prompts, max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def generate_captions(self, images, names: list[str]):

        pil_images = [self.to_pil(image) for image in images]
        prompts = self.build_prompts(pil_images, names)

        captions = []
        batch_stats = []

        pad_token_id = self.processor.tokenizer.pad_token_id

        for batch in self.split_into_batches(list(zip(pil_images, prompts)), self.batch_size):

            image_batch = [b[0] for b in batch]
            prompt_batch = [b[1] for b in batch]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time()

            caption_batch, inputs, generated = self.run_batch(
                image_batch, prompt_batch, self.max_new_tokens)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time() - start_time

            if pad_token_id is not None:
                new_tokens = int((generated != pad_token_id).sum())
            else:
                new_tokens = int(generated.numel())

            batch_stats.append({
                "images": len(image_batch),
                "prompt_tokens": int(inputs.input_ids.shape[1]),
                "generated_tokens": new_tokens,
                "seconds": elapsed,
            })

            captions.extend(caption_batch)

        total_seconds = sum(b["seconds"] for b in batch_stats)
        total_tokens = sum(b["generated_tokens"] for b in batch_stats)

        self.last_run_stats = {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "dtype": str(self.dtype),
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "load_seconds": self.load_seconds,
            "images": len(pil_images),
            "batches": len(batch_stats),
            "generation_seconds": total_seconds,
            "seconds_per_image": total_seconds / max(len(pil_images), 1),
            "generated_tokens": total_tokens,
            "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024 ** 3
                if torch.cuda.is_available() else None),
            "per_batch": batch_stats,
        }

        return captions


class QwenHFBackend(QwenVLHFBackend):

    # The official AWQ checkpoint needs autoawq, which the AI module image does not
    # ship; the base checkpoint plus BitsAndBytes matches the Qwen3-VL backend.
    default_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    @staticmethod
    def model_class():
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


class Qwen3VLHFBackend(QwenVLHFBackend):

    default_model_id = "Qwen/Qwen3-VL-4B-Instruct"

    @staticmethod
    def model_class():
        # Imported here rather than at module level: on transformers < 4.57 the import
        # fails, and semantic_mapper treats any ImportError from the captioner package
        # as "no captioner installed", which would silently disable captioning.
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Qwen3-VL requires transformers>=4.57. Upgrade transformers or select "
                "another captioning backend."
            ) from e
        return Qwen3VLForConditionalGeneration


QWEN_BACKENDS = {
    "qwen3vl": Qwen3VLHFBackend,
    "qwen2_5vl": QwenHFBackend,
}


def load_qwen_backend(
        captioning_model: str,
        quantization: Optional[str] = "int4",
        model_id: Optional[str] = None,
        batch_size: int = 1,
        max_new_tokens: int = 32,
        max_pixels: int = 1280 * 28 * 28,
        ):
    '''Construct a Qwen-VL HF backend by name (shared by CLI / ROS nodes).'''
    if captioning_model not in QWEN_BACKENDS:
        raise ValueError(
            f"Unknown captioning_model={captioning_model!r}. "
            f"Choose from {sorted(QWEN_BACKENDS)}")
    if quantization in ("", "none", "None"):
        quantization = None
    return QWEN_BACKENDS[captioning_model](
        model_id=model_id,
        quantization=quantization,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
    )


class PaliGemmaVLLMBackend(CaptioningModel):
    def __init__(
            self,
            model_id="google/paligemma2-3b-ft-docci-448"
            ):
        self.model_id = model_id
        # model_id = "google/paligemma2-3b-pt-448"

        self.llm = LLM(
            model=model_id, 
            dtype=torch.bfloat16,
            cpu_offload_gb=16)


        self.prompt = "<image>caption en "

        self.sampling_params = self.llm.get_default_sampling_params()
        self.sampling_params.stop = '.'



    def generate_captions(self, images):

    
        inputs = [{
            "prompt": self.prompt,
            "multi_modal_data": {
                "image": image
            }
        } for image in images]

        outputs: List[RequestOutput] = self.llm.generate(inputs, sampling_params=self.sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)



def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Caption a directory of saved crops, such as a SemanticDictSaver '
                    'output directory of <object_id>_<name>/crop.png folders.')
    parser.add_argument('crops_path')
    parser.add_argument('--captioning_model', default='qwen3vl', choices=['qwen3vl', 'qwen2_5vl'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument(
        '--quantization',
        default='int4',
        choices=['int4', 'int8', 'none'],
        help='Weight quantization. "none" loads full precision (bf16).')
    parser.add_argument(
        '--model_id',
        default=None,
        help='Override the backend default checkpoint.')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument(
        '--no_warmup',
        action='store_true',
        help='Skip the untimed warmup batch. Warmup keeps CUDA kernel selection out '
             'of the first timed batch, which matters when comparing backends.')
    parser.add_argument(
        '--output_dir',
        default=None,
        help='If set, write captions under output_dir/<id>_<name>/caption.txt '
             '(mirrors the crops_path folder layout).')
    parser.add_argument(
        '--also_write_caption_file',
        default=None,
        help='Optional filename written inside each original crop folder, e.g. caption_qwen3.txt')
    args = parser.parse_args(argv)

    # List to store images
    images = []
    names = []
    subdirs = []

    # Iterate over all subdirectories
    for subdir in sorted(os.listdir(args.crops_path)):  # Sorting ensures order
        subdir_path = os.path.join(args.crops_path, subdir)

        if not os.path.isdir(subdir_path):
            continue

        crop_name = subdir.split('_', 1)[-1]

        # crop.png is written by SemanticDictSaver, rgb.png by the offline crop datasets
        for file_name in ("crop.png", "rgb.png"):
            img_path = os.path.join(subdir_path, file_name)

            if os.path.exists(img_path):
                img = iio.imread(img_path)
                if img is not None:
                    images.append(img)
                    names.append(crop_name)
                    subdirs.append(subdir)
                break

    if not images:
        raise SystemExit(f'No crops found under {args.crops_path}')

    print(f'Loaded {len(images)} crops from {args.crops_path}')

    captioning_model = load_qwen_backend(
        args.captioning_model,
        quantization=args.quantization,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f'Loaded {captioning_model.model_id} in {captioning_model.load_seconds:.1f}s')

    if not args.no_warmup:
        captioning_model.warmup(images, names)

    captions = captioning_model.generate_captions(images, names)

    stats = dict(captioning_model.last_run_stats)
    stats['backend'] = args.captioning_model
    stats['crops_path'] = args.crops_path
    stats['warmup'] = not args.no_warmup

    print(
        f'\n{args.captioning_model} ({stats["model_id"]}, quantization={stats["quantization"]}): '
        f'{stats["generation_seconds"]:.1f}s for {stats["images"]} crops '
        f'({stats["seconds_per_image"]:.2f}s/crop, '
        f'{stats["tokens_per_second"]:.1f} tok/s, '
        f'{stats["generated_tokens"]} tokens generated, '
        f'peak GPU {stats["peak_gpu_memory_gb"]:.1f} GB)\n')

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'timing.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    for subdir, name, caption in zip(subdirs, names, captions):
        print(f'{subdir}: {caption}')

        if args.output_dir is not None:
            out_folder = os.path.join(args.output_dir, subdir)
            os.makedirs(out_folder, exist_ok=True)
            with open(os.path.join(out_folder, 'caption.txt'), 'w', encoding='utf-8') as f:
                f.write(caption)

        if args.also_write_caption_file is not None:
            crop_folder = os.path.join(args.crops_path, subdir)
            with open(os.path.join(crop_folder, args.also_write_caption_file), 'w', encoding='utf-8') as f:
                f.write(caption)

    if args.output_dir is not None:
        print(f'Saved {len(captions)} captions to {args.output_dir}')


if __name__ == "__main__":
    main()