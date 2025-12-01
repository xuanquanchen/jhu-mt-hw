# -*- coding: utf-8 -*-
"""
Quick test script to verify adapter model works
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from model_adapter import BertChineseEmbSlimCNNlstmBertAdapter

# Test parameters
SEQ_LEN = 200
OUTPUT_SIZE = 4  # O, ，, 。, ？
DROPOUT = 0.1
BATCH_SIZE = 2

print("Testing Adapter Model...")
print("=" * 60)

# Test with adapter
print("\n1. Testing model WITH adapter:")
model_adapter = BertChineseEmbSlimCNNlstmBertAdapter(
    SEQ_LEN, OUTPUT_SIZE, DROPOUT, None,
    use_adapter=True,
    adapter_size=384,
    use_attention_fusion=True
)

# Count parameters
total_params = sum(p.numel() for p in model_adapter.parameters())
trainable_params = sum(p.numel() for p in model_adapter.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {frozen_params:,}")
print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")

# Test forward pass
test_input = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
output = model_adapter(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected output shape: [{BATCH_SIZE * SEQ_LEN}, {OUTPUT_SIZE}]")
assert output.shape == (BATCH_SIZE * SEQ_LEN, OUTPUT_SIZE), "Output shape mismatch!"
print("✓ Forward pass successful!")

# Test without adapter
print("\n2. Testing model WITHOUT adapter:")
model_no_adapter = BertChineseEmbSlimCNNlstmBertAdapter(
    SEQ_LEN, OUTPUT_SIZE, DROPOUT, None,
    use_adapter=False,
    use_attention_fusion=False
)

total_params_no = sum(p.numel() for p in model_no_adapter.parameters())
trainable_params_no = sum(p.numel() for p in model_no_adapter.parameters() if p.requires_grad)

print(f"Total parameters: {total_params_no:,}")
print(f"Trainable parameters: {trainable_params_no:,}")
print(f"Trainable ratio: {trainable_params_no/total_params_no*100:.2f}%")

output_no = model_no_adapter(test_input)
assert output_no.shape == (BATCH_SIZE * SEQ_LEN, OUTPUT_SIZE), "Output shape mismatch!"
print("✓ Forward pass successful!")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print(f"\nParameter reduction with adapter:")
print(f"  Trainable params: {trainable_params:,} vs {trainable_params_no:,}")
print(f"  Reduction: {(1 - trainable_params/trainable_params_no)*100:.1f}%")

