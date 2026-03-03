'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-02 03:05:22
 # @ Description:
 '''

import os
import time
import torch
import transformers
from vllm import LLM, SamplingParams

# CONFIGURATION
# 1. Select a Transformer (Qwen2.5-1.5B or 0.5B to match your paper)
# 2. Select an SSM (Mamba-1.4B or 2.8B) 
# Note: vLLM support for Mamba is newer, ensure you grab a supported version.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
# MODEL_NAME = "state-spaces/mamba-790m-hf" # Run separately
# MODEL_NAME = "tiiuae/Falcon-H1-0.5B-Base"

# Set GPU memory utilization to max to give it a fair chance (like your PyTorch runs)
gpu_memory_utilization = 0.9

def gen_random_prompt(seq_len: int = 16) -> str:
    """Generate a prompt with approximately seq_len tokens"""
    prompt_ = "random "
    prompt = ""
    for i in range(seq_len - 1):
        prompt += prompt_
    return prompt

def test_sequence_length(seq_len):
    print(f"Testing sequence length: {seq_len}...")
    
    try:
        # Calculate max_model_len: need space for input + at least 1 output token
        max_model_len = seq_len + 16  # Add buffer for generation
        
        # For sequences > 32768, enable chunked prefill to work around kernel limitations
        enable_chunked_prefill = seq_len > 32768
        if enable_chunked_prefill:
            print(f"Enabling chunked prefill for large sequence length")
        
        # Initialize vLLM engine
        # We re-init for every length to ensure clean memory state (slow but accurate)
        llm = LLM(model=MODEL_NAME, 
                  gpu_memory_utilization=gpu_memory_utilization,
                  trust_remote_code=True,
                  tensor_parallel_size=1,
                  max_model_len=max_model_len,
                  enforce_eager=True,  # Use eager mode for large sequences to avoid CUDA graph limits
                  enable_chunked_prefill=enable_chunked_prefill,  # Enable for large sequences
                  max_num_batched_tokens=max_model_len if enable_chunked_prefill else None)  # Control batch size
        
        # Initialize tokenizer to verify token count
        is_mamba_model = 'mamba' in MODEL_NAME.lower()
        tokenizer_config = "EleutherAI/gpt-neox-20b" if is_mamba_model else MODEL_NAME
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_config)
        
        # Generate prompt using the same approach as run_mem_footprint_vllm.py
        prompt_str = gen_random_prompt(seq_len)
        
        # Verify token count
        actual_tokens = len(tokenizer.encode(prompt_str))
        print(f"Generated prompt with {actual_tokens} tokens (target: {seq_len})")
                  
        sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
        
        outputs = llm.generate([prompt_str], sampling_params)
        
        # Check actual input length from outputs
        actual_input_len = len(outputs[0].prompt_token_ids)
        print(f"SUCCESS: Processed {actual_input_len} tokens.")
        
        # Clean up to free memory for next run
        import gc
        del llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return True, actual_input_len
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"FAIL: OOM at {seq_len}")
            return False, seq_len
        else:
            print(f"FAIL: Other error at {seq_len}: {e}")
            return False, seq_len
    except Exception as e:
        print(f"CRITICAL FAIL: {e}")
        return False, seq_len

# SEQUENCE SWEEP
# Step up in chunks (e.g., 8k, 16k, 32k, 48k, 64k, ...)
# Stop when you hit False
sequences = [8192, 16384, 32768, 49152, 57344, 59000, 65536, 81920, 98304, 131072]
# sequences = [163840, 180224]

results = {}

for seq in sequences:
    success, actual_len = test_sequence_length(seq)
    time.sleep(5)  # Brief pause to allow memory to settle
    results[seq] = "PASS" if success else "OOM"
    if not success:
        break

print("\nFINAL RESULTS:")
print(results)