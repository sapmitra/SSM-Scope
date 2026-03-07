'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-02 03:04:23
 # @ Description: Measures GPU memory footprint for the prefill and decode
 #                 phases across transformer and SSM-based language models.
 #                 Outputs results to CSV for downstream analysis.
 '''

import argparse
import os
import sys
import gc
import torch
import csv
from datetime import datetime

from models.profile_runner import LMProfile, MambaProfile

EXPORT = True

def save_to_csv(data, filename="memory_footprints.csv"):
    """Save memory footprint data to a CSV file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'model_config', 'seq_len', 'batch_size', 
                     'model_size_mb', 'activation_memory_mb', 'kv_cache_mb', 'reserved_memory_mb', 'total_memory_mb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    print(f"Data saved to {filepath}")

def model_prefill(model_name: str, model_config: str, seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', csv_filename: str = "memory_footprints.csv"):
    # Record baseline memory before loading any model
    torch.cuda.empty_cache()
    gc.collect()
    baseline_memory = torch.cuda.memory_allocated()
    # baseline_memory = 0
    print(f"Baseline GPU memory: {baseline_memory / (1024**2):.2f} MB")
    
    # Determine which profile class to use based on model name/config
    is_mamba_model = 'mamba' in model_name.lower() or 'mamba' in model_config.lower()
    ProfileClass = MambaProfile if is_mamba_model else LMProfile
    # ProfileClass = LMProfile    
    if is_mamba_model:
        # For Mamba models - only need to run once as there's no KV cache
        print("=== Memory footprint for Mamba model (no KV cache) ===")
        model = ProfileClass(model_name, model_config, device)
        memory_details = model.eval_memory_prefill(seq_len, batch_size, EXPORT, use_kv_cache=False)
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Process memory details
        print(f"Memory details: {memory_details}")
        
        # For Mamba models, KV cache is 0
        kv_cache_size = 0
        activation_memory = memory_details['activation_memory_mb']
        model_size = memory_details['model_size_mb']
        
        total_memory = model_size + activation_memory
        print(f"Model size: {model_size:.2f} MB")
        print(f"Activation memory: {activation_memory:.2f} MB")
        print(f"KV cache size: {kv_cache_size:.2f} MB (not applicable for Mamba models)")
        print(f"Total memory footprint: {total_memory:.2f} MB")
        
        # Save data to CSV
        data = {
            'model_name': model_name,
            'model_config': model_config,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'model_size_mb': round(model_size, 2),
            'activation_memory_mb': round(activation_memory, 2),
            'kv_cache_mb': 0,  # No KV cache for Mamba
            'reserved_memory_mb': round(memory_details['peak_memory_reserved_mb'], 2),
            'total_memory_mb': round(total_memory, 2)
        }
        save_to_csv(data, filename=csv_filename)
    else:
        # Original flow for transformer models with KV cache
        # Memory footprint without KV cache
        print("=== Memory footprint WITHOUT KV cache ===")
        model = ProfileClass(model_name, model_config, device)
        memory_details_nokv = model.eval_memory_prefill(seq_len, batch_size, EXPORT, use_kv_cache=False)

        # More aggressive cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations are complete
        
               
        # Memory footprint with KV cache enabled
        print("=== Memory footprint WITH KV cache ===")
        model = ProfileClass(model_name, model_config, device)
        memory_details_withkv = model.eval_memory_prefill(seq_len, batch_size, EXPORT, use_kv_cache=True)

        print(f"Memory details: {memory_details_nokv}")
        print(f"Memory details: {memory_details_withkv}")


        del model

        kv_cache_size = memory_details_withkv['activation_memory_mb'] - memory_details_nokv['activation_memory_mb']
        print(f"KV cache size: {kv_cache_size:.2f} MB")
        only_activation_memory = memory_details_withkv['activation_memory_mb'] - kv_cache_size
        print(f"Activation memory without KV cache: {only_activation_memory:.2f} MB")
        print(f"model size: {memory_details_withkv['model_size_mb']:.2f} MB")

        total_memory = memory_details_withkv['model_size_mb'] + only_activation_memory + kv_cache_size
        print(f"Total memory footprint: {total_memory:.2f} MB")
        
        # Save data to CSV
        data = {
            'model_name': model_name,
            'model_config': model_config,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'model_size_mb': round(memory_details_withkv['model_size_mb'], 2),
            'activation_memory_mb': round(only_activation_memory, 2),
            'kv_cache_mb': round(kv_cache_size, 2),
            'reserved_memory_mb': round(memory_details_withkv['peak_memory_reserved_mb'], 2),
            'total_memory_mb': round(total_memory, 2)
        }
        save_to_csv(data, filename=csv_filename)
    

def save_decode_to_csv(data, filename="memory_decode_footprints.csv"):
    """Save decode phase memory footprint data to a CSV file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'model_config', 'input_seq_len', 'output_seq_len', 'total_seq_len',
                     'batch_size', 'model_size_mb', 'generation_memory_mb', 'kv_cache_growth_mb', 
                     'reserved_memory_mb', 'total_memory_mb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    print(f"Decode data saved to {filepath}")


def model_decode(model_name: str, model_config: str, input_seq_len: int = 64, output_seq_len: int = 32, batch_size: int = 1, device: str = 'cuda'):
    """Measure memory footprint during the decoding phase."""
    print(f"=== Memory footprint for DECODING phase ===")
    print(f"Model: {model_name}")
    print(f"Input sequence length: {input_seq_len}")
    print(f"Output sequence length: {output_seq_len}")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Determine which profile class to use
    is_mamba_model = 'mamba' in model_name.lower() or 'mamba' in model_config.lower()
    ProfileClass = MambaProfile if is_mamba_model else LMProfile
    
    # Load model and measure decode memory
    model = ProfileClass(model_name, model_config, device)
    memory_details = model.eval_memory_decode(input_seq_len, batch_size, output_seq_len, EXPORT)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Decode memory details: {memory_details}")
    
    # Calculate KV cache growth (approximate)
    generation_memory = memory_details['generation_memory_mb']
    model_size = memory_details['model_size_mb']
    
    # For decode phase, most memory growth is from KV cache
    kv_cache_growth = generation_memory
    total_memory = model_size + generation_memory
    
    print(f"Model size: {model_size:.2f} MB")
    print(f"Generation memory (including KV cache growth): {generation_memory:.2f} MB")
    print(f"Estimated KV cache growth: {kv_cache_growth:.2f} MB")
    print(f"Total memory footprint: {total_memory:.2f} MB")
    
    # Save data to CSV
    data = {
        'model_name': model_name,
        'model_config': model_config,
        'input_seq_len': input_seq_len,
        'output_seq_len': output_seq_len,
        'total_seq_len': input_seq_len + output_seq_len,
        'batch_size': batch_size,
        'model_size_mb': round(model_size, 2),
        'generation_memory_mb': round(generation_memory, 2),
        'kv_cache_growth_mb': round(kv_cache_growth, 2),
        'reserved_memory_mb': round(memory_details['peak_memory_reserved_mb'], 2),
        'total_memory_mb': round(total_memory, 2)
    }
    save_decode_to_csv(data)


def run_mem_footprint():
    # Test models with specific sequence lengths
    # First set: 256 to 16384 in powers of 2
    # initial_seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]
    
    # # # Second set: from 8192 to 65536 in increments of 8192
    # # extended_seq_lengths = list(range(8192, 65536 + 1, 8192))
    # extended_seq_lengths = list(range(8192, 150000 + 1, 8192))
    # extended_seq_lengths = list(range(8192, 65536 + 1, 8192))
    # another_extended_seq_lengths = list(range(65536, 163840 + 1, 16384))

    
    # # # Combine all unique sequence lengths
    # all_seq_lengths = sorted(list(set(initial_seq_lengths + extended_seq_lengths)))
    # all_seq_lengths = sorted(list(set(initial_seq_lengths + extended_seq_lengths + another_extended_seq_lengths)))
    

    # print(f"Testing with sequence lengths: {all_seq_lengths}")
    
    # for seq_len in all_seq_lengths:
    #     print(f"\n{'='*60}")
    #     print(f"Testing model with sequence length: {seq_len}")
    #     print(f"{'='*60}")
    #     model_prefill(
    #         # model_name='qwen25-instruct',
    #         model_name='qwen25-1.5b-instruct',
    #         # model_name='qwen25-instruct-int4',
    #         # model_config='Qwen/Qwen2.5-0.5B-Instruct',
    #         model_config='Qwen/Qwen2.5-1.5B-Instruct',
    #         # model_config='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
    #         seq_len=seq_len,
    #         batch_size=1,
    #         device='cuda'
    #     )
        # model_prefill(
        #     # model_name='mamba-790m',
        #     model_name='mamba-1.4b',
        #     model_config='state-spaces/mamba-1.4b',
        #     seq_len=seq_len,
        #     batch_size=1,
        #     device='cuda'
        # )
        # model_prefill(
        #     model_name='zamba2',
        #     model_config='Zyphra/Zamba2-1.2B-Instruct-v2',
        #     seq_len=seq_len,
        #     batch_size=1,
        #     device='cuda'
        # )
        # model_prefill(
        #     model_name='llama3_2',
        #     model_config='meta-llama/Llama-3.2-1B-Instruct',
        #     seq_len=seq_len,
        #     batch_size=1,
        #     device='cuda'
        # )
        # model_prefill(
        #     model_name='phi3',
        #     model_config='microsoft/Phi-3-mini-128k-instruct',
        #     seq_len=seq_len,
        #     batch_size=1,
        #     device='cuda'
        # )
        # model_prefill(
        #     model_name='mamba2-780m',
        #     model_config='state-spaces/mamba2-780m',
        #     seq_len=seq_len,
        #     batch_size=1,
        #     device='cuda'
        # )
        # model_prefill(
        #     model_name='falcon-h1-0.5b',
        #     model_config='tiiuae/Falcon-H1-0.5B-Base',
        #     seq_len=seq_len,
        #     batch_size=1,
        #     device='cuda'
        # )
    # Uncomment to test with different models
    # model_prefill(model_name='qwen25-instruct', model_config='Qwen/Qwen2.5-0.5B-Instruct', seq_len=8192, batch_size=1, device='cuda')
    # model_prefill(model_name='qwen25-instruct-int4', model_config='Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', seq_len=57344, batch_size=1, device='cuda')
    # model_prefill(model_name='phi3', model_config='microsoft/Phi-3-mini-128k-instruct', seq_len=256, batch_size=1, device='cuda')
    # model_prefill(model_name='llama3_2', model_config='meta-llama/Llama-3.2-1B-Instruct', seq_len=131072, batch_size=1, device='cuda')
    # model_prefill(model_name='opt-350m', model_config='facebook/opt-350m', seq_len=2047, batch_size=1, device='cuda')
    # model_prefill(model_name='mamba-790m', model_config='state-spaces/mamba-790m', seq_len=180224, batch_size=1, device='cuda')
    # model_prefill(model_name='quamba-790m-w4a8', model_config='ut-enyac/quamba-790m-w4a8', seq_len=1024, batch_size=1, device='cuda')
    # model_prefill(model_name='mamba-1.4b', model_config='state-spaces/mamba-1.4b', seq_len=220000, batch_size=1, device='cuda')
    model_prefill(model_name='mamba2-780m', model_config='state-spaces/mamba2-780m', seq_len=8192, batch_size=1, device='cuda')
    # model_prefill(model_name='quamba2-780m-w4a8', model_config='ut-enyac/quamba2-780m-w4a8', seq_len=8192, batch_size=1, device='cuda')
    # model_prefill(model_name='zamba2', model_config='Zyphra/Zamba2-1.2B-Instruct-v2', seq_len=256, batch_size=1, device='cuda')
    # model_prefill(model_name='hymba', model_config='nvidia/Hymba-1.5B-Base', seq_len=1024, batch_size=1, device='cuda')
    # model_prefill(model_name='mamba-790m-hf', model_config='state-spaces/mamba-790m-hf', seq_len=65536, batch_size=1, device='cuda')
    # model_prefill(model_name='falcon-h1-0.5b', model_config='tiiuae/Falcon-H1-0.5B-Base', seq_len=180224, batch_size=1, device='cuda')

    # Test decode phase
    # TODO: This only works for low input sequence lengths when activation memory is not too high and does not exceed the KV cache size
    # model_decode(model_name='qwen25-instruct', model_config='Qwen/Qwen2.5-0.5B-Instruct', input_seq_len=64, output_seq_len=200000, batch_size=1, device='cuda')
    # model_decode(model_name='phi3', model_config='microsoft/Phi-3-mini-128k-instruct', input_seq_len=64, output_seq_len=8192, batch_size=1, device='cuda')

    # Test KV cache scaling
    # analyze_kv_cache_scaling(model_name='qwen25-instruct', model_config='Qwen/Qwen2.5-0.5B-Instruct', input_seq_len=64, output_seq_lengths=[1, 8, 16, 32, 64], batch_size=1, device='cuda')
    
    # Uncomment to run comprehensive analysis
    # run_comprehensive_memory_analysis()

if __name__ == "__main__":
    run_mem_footprint()
    pass