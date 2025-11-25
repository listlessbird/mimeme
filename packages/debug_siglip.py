#!/usr/bin/env python3
"""
Debug script to test SigLIP2 with BitsAndBytes quantization on a single image.
"""
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import sys

def main():
    print("="*60)
    print("SigLIP2 + BitsAndBytes Debug Script")
    print("="*60)

    # Configuration
    model_name = "google/siglip2-base-patch16-naflex"

    print(f"\n1. Loading model: {model_name}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    # Test with and without quantization
    use_quantization = len(sys.argv) > 1 and sys.argv[1] == "--quantize"

    if use_quantization:
        # Create BitsAndBytes config
        print("\n2. Creating BitsAndBytes config...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"   ✓ BnB config created")
    else:
        print("\n2. Skipping quantization (testing without BnB)")
        bnb_config = None

    # Load processor
    print("\n3. Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✓ Processor loaded")

    # Load model
    if use_quantization:
        print("\n4. Loading model WITH quantization...")
        model_kwargs = {
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "device_map": "auto",
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
    else:
        print("\n4. Loading model WITHOUT quantization...")
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "device_map": "auto",
        }

    try:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        print(f"   ✓ Model loaded successfully")
        print(f"   Model device: {model.device}")
        print(f"   Model dtype: {model.dtype}")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        sys.exit(1)

    # Create a simple test image
    print("\n5. Creating test image (256x256 RGB)...")
    test_image = Image.new('RGB', (256, 256), color='red')
    print(f"   ✓ Test image created: {test_image.size}, mode={test_image.mode}")

    # Process image
    print("\n6. Processing image with processor...")
    inputs = processor(
        images=test_image,
        return_tensors="pt",
        padding="max_length",
        max_num_patches=256,
    )
    print(f"   ✓ Inputs created")
    print(f"   Input keys: {list(inputs.keys())}")
    for key, val in inputs.items():
        if hasattr(val, 'shape'):
            print(f"   - {key}: shape={val.shape}, dtype={val.dtype}")

    # Move inputs to device and convert to bfloat16
    print("\n7. Moving inputs to model device and converting dtype...")
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to(model.device)
        for k, v in inputs.items()
    }
    print(f"   ✓ Inputs moved to {model.device}")
    for key, val in inputs.items():
        if hasattr(val, 'dtype'):
            print(f"   - {key}: dtype={val.dtype}")

    # Run inference
    print("\n8. Running inference...")
    try:
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        print(f"   ✓ Inference successful!")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output dtype: {outputs.dtype}")
    except Exception as e:
        print(f"   ✗ Inference failed!")
        print(f"   Error: {e}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()
