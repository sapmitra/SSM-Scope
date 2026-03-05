'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-01 20:26:10
 # @ Description: Model loader and profiling entry points.
 #                 Defines LMProfile (HuggingFace transformer models) and
 #                 MambaProfile (mamba_ssm models), plus per-model convenience
 #                 functions invokable via the CLI (main()).
 '''

import time
import torch
import transformers 
from profiling.eval import profile_model, profile_model_generate, profile_model_shape, profile_generate_shape, profile_model_dynamo, profile_model_dynamo_generate, profile_model_mamba, profile_model_mamba_generate, profile_model_energy, profile_model_mamba_energy
import argparse 
torch.manual_seed(1969)

# Lazy import for mamba_ssm - only available in SSM-specific venv
MambaLMHeadModel = None

def get_mamba_lm_head_model():
    global MambaLMHeadModel
    if MambaLMHeadModel is None:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as _MambaLMHeadModel
        MambaLMHeadModel = _MambaLMHeadModel
    return MambaLMHeadModel

out_dir = "./ssm-characterization-out"
# out_dir = "./hybrid-out"

custom_ops = [ 'newgeluactivation', 'llamarmsnorm',   'segformerdwconv', 'detrfrozenbatchnorm2d', 'wqlinearmmfunction', 'frozenbatchnorm2d', 'mistralrmsnorm','mixtralrmsnorm', 'zamba2rmsnormgated', 'zamba2rmsnorm','qwen2rmsnorm', 'mambarmsnorm', 'hymbarmsnorm'] #'rmsnorm', 'layernormfn', 'layernormlinearfn','llamarotaryembedding', 'wqlinear_gemm',  'qwen2rotaryembedding'
NUM_RUNS=10 #25
EXPORT=True #False

def gen_random_prompt (seq_len: int = 16, batch_size: int = 1): 
    prompt_ = "random "
    prompt = ""
    for i in range (seq_len - 1): 
        prompt += prompt_ 
    if batch_size > 1: 
        prompt_list = []
        for i in range (batch_size): 
            prompt_list.append(prompt)
        return prompt_list
    return prompt

def memory_usage_prefill(model_name, model, inputs, device, use_kv_cache=False):
    # Clear cache to get accurate measurements
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # Calculate model weights memory
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    # Record memory before inference
    if device.startswith('cuda'):
        mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"Memory before inference: {mem_before:.2f} MB")
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"Memory reserved before inference: {mem_reserved:.2f} MB")
    else:
        mem_before = 0  # Cannot track CPU memory in this way
    
    # Run inference
    with torch.no_grad():
        if ("mamba" in model_name):
            outputs = model(input_ids=inputs['input_ids'],)
        else:
            if use_kv_cache:
                outputs = model(**inputs, use_cache=True)
            else:
                outputs = model(**inputs, use_cache=False)

    # Measure peak memory and current memory after inference
    if device.startswith('cuda'):
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        current_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        
        mem_reserved_after = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"Memory reserved after inference: {mem_reserved_after:.2f} MB")
        
        # activation_mem = peak_mem - mem_before
        activation_mem = peak_mem - model_size_mb
    else:
        peak_mem = 0
        current_mem = 0
        activation_mem = 0
    
    
    return {
        "model_name": model_name,
        "device": device,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": peak_mem,
        "activation_memory_mb": activation_mem,
        "current_memory_mb": current_mem,
        "peak_memory_reserved_mb": mem_reserved_after,
    }

def memory_usage_decode(model_name, model, inputs, device, output_seq_len=1, input_seq_len=None):
        """Measure memory usage during the decoding phase with KV cache growth."""
        # Clear cache to get accurate measurements
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        
        # Calculate model weights memory
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        # num_layers = model.config.num_hidden_layers  # Number of transformer layers
        # num_heads = model.config.num_attention_heads  # Number of attention heads
        # num_key_value_heads = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else num_heads  # For models with separate KV heads
        # head_dim = model.config.hidden_size // num_heads  # Dimension per head
        # print(f"Model size: {model_size_mb:.2f} MB, Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}, KV heads: {num_key_value_heads}")

        # past_kv = tuple(
        #     (
        #         torch.randn(1, num_key_value_heads, max_seq_len, head_dim).to(inputs.input_ids.dtype).to(inputs.input_ids.device),  # Random key tensor
        #         torch.randn(1, num_key_value_heads, max_seq_len, head_dim).to(inputs.input_ids.dtype).to(inputs.input_ids.device)   # Random value tensor
        #     )
        #     for _ in range(num_layers)
        # )

        # Record memory before generation
        if device.startswith('cuda'):
            mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)
            print(f"Memory before generation: {mem_before:.2f} MB")
        else:
            mem_before = 0
        
        # Run generation to simulate decoding phase
        with torch.no_grad():
            if ("mamba" in model_name.lower()):
                # For Mamba models, generate tokens to measure memory scaling
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=output_seq_len,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else 50256
                )
            else:
                # For transformer models with KV cache
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=output_seq_len,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else 50256
                )

                # outputs = model(input_ids=inputs.input_ids[:,:1], past_key_values=past_kv, use_cache=True)
                
        # Measure peak memory after generation
        if device.startswith('cuda'):
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            current_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_reserved_after = torch.cuda.memory_reserved(device) / (1024 ** 2)
            
            # TODO: This only works for low input sequence lengths when activation memory is not too high and does not exceed the KV cache size
            total_generation_mem = peak_mem - model_size_mb
        else:
            peak_mem = 0
            current_mem = 0
            total_generation_mem = 0
            mem_reserved_after = 0
        
        return {
            "model_name": model_name,
            "device": device,
            "input_seq_len": input_seq_len or inputs['input_ids'].shape[1],
            "output_seq_len": output_seq_len,
            # "max_seq_len": max_seq_len,
            "total_seq_len": (input_seq_len or inputs['input_ids'].shape[1]) + output_seq_len,
            "model_size_mb": model_size_mb,
            "peak_memory_mb": peak_mem,
            "generation_memory_mb": total_generation_mem,
            "current_memory_mb": current_mem,
            "peak_memory_reserved_mb": mem_reserved_after,
        }



class LMProfile: 
    def __init__(self, model_name: str = 'gpt2' ,model_config: str = 'gpt2', device: str = 'cuda'): 

        self.model_config = model_config 
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'
        

        if device == "cuda":
            if ("8bit" in model_name):
                self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config).eval()
                
            else:    
                if ("Mixtral" in model_config):
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, device_map = "auto", torch_dtype = "auto").eval()
                elif ("mamba" in model_config):
                    # loads fp32, converts to fp16, keeps both weights in memory, but fp16 is used for inference probably
                    # self.model = transformers.MambaForCausalLM.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
                    # loads original weights, fp32 here, no double copy stored
                    # self.model = transformers.MambaForCausalLM.from_pretrained(self.model_config, torch_dtype="auto", device_map="cuda:0")
                    # loads in fp16
                    self.model = transformers.MambaForCausalLM.from_pretrained(self.model_config, torch_dtype=torch.float16, device_map="cuda:0")
                elif ("Hymba" in model_config):
                    # self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, trust_remote_code=True).to(device).to(torch.float16).eval()
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, trust_remote_code=True, torch_dtype="auto", device_map="cuda:0")
                elif ("Qwen" in model_config):
                    #for quantized and non quantized models
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, torch_dtype="auto", device_map="cuda:0")
                    # for non quantized models
                    # self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
                elif ("Phi" in model_config):
                    # self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True).eval()
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, torch_dtype="auto", device_map="cuda:0", trust_remote_code=False).eval()
                elif("Llama" in model_config):
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, torch_dtype="auto", device_map="cuda:0").eval()
                elif ("Zamba" in model_config):
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, torch_dtype="auto", device_map="cuda:0").eval()
                elif ("Falcon" in model_config):
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
                elif ("Nemotron" in model_config):
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, trust_remote_code=True).eval()
                    self.model = self.model.cuda().to(torch.bfloat16)
                else:
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config,).to(device).to(torch.float16).eval()
        else: 
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, ).to(device).eval()
        print (self.model.dtype)
        print(self.model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def eval_(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
        # profile_model_energy(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)
    
    def eval_shape(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir_shapes, export)
    
    def eval_gen_(self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_generate(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir, export)
    
    def eval_gen_shape (self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        profile_generate_shape(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir_shapes, export)
    
    def eval_dynamo(self, seq_len: int = 16, batch_size: int = 1, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_dynamo (self.model_name, self.model, inputs, custom_ops, num_runs, self.device, True, out_dir_dynamo, export)
    
    def eval_dynamo_gen (self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None):
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        torch._dynamo.reset()
        self.model = torch.compile(self.model, backend= "inductor")
        profile_model_dynamo_generate(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir_dynamo, export)

    def eval_memory_prefill(self, seq_len: int = 16, batch_size: int = 1, export: bool = EXPORT, use_kv_cache: bool = False, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        return memory_usage_prefill(self.model_name, self.model, inputs, self.device, use_kv_cache=use_kv_cache)

    

    def eval_memory_decode(self, seq_len: int = 16, batch_size: int = 1, output_seq_len: int = 1, export: bool = EXPORT): 
        """Evaluate memory usage during decoding phase with specified output length."""
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}_decode_{output_seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        return memory_usage_decode(self.model_name, self.model, inputs, self.device, output_seq_len=output_seq_len, input_seq_len=seq_len)


class MambaProfile:
    def __init__(self, model_name: str = 'mamba', model_config: str = 'mamba', device: str = 'cuda'):
        self.model_config = model_config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = f'{model_name}'

        MambaLMHeadModel = get_mamba_lm_head_model()
        if device == "cuda":
            self.model = MambaLMHeadModel.from_pretrained(self.model_config).to(device).to(torch.float16).eval()
        else:
            self.model = MambaLMHeadModel.from_pretrained(self.model_config).to(device).eval()
        print(self.model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def eval_(self, seq_len: int = 16, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None):
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        self.model.eval()
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        # memory_usage(self.model_name, self.model, inputs, self.device)
        profile_model_mamba(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export, self.tokenizer)
        # profile_model_mamba_energy(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)

    def eval_memory_prefill(self, seq_len: int = 16, batch_size: int = 1, export: bool = EXPORT, use_kv_cache: bool = False, inputs = None): 
        self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        return memory_usage_prefill(self.model_name, self.model, inputs, self.device, use_kv_cache=use_kv_cache)
    
    def eval_gen_(self, seq_len: int = 16, max_num_tokens: int = 16, num_runs : int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None): 
        self.model_name = f'gen_{self.model_name}_{self.device}_{max_num_tokens}_{seq_len}'
        prompt = gen_random_prompt(seq_len)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        profile_model_mamba_generate(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, max_num_tokens, False, out_dir, export)

# for evaluating Zamba2 please enable seperate virtual env with transformers>=4.48.0
# class Zamba2Profile:
#     def __init__(self, model_name: str = 'zamba2', model_config: str = 'zamba2', device: str = 'cuda'):
#         self.model_config = model_config
#         self.device = device if torch.cuda.is_available() else 'cpu'
#         self.model_name = f'{model_name}'

#         if device == "cuda":
#             # consumes less memory for some reason
#             # self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config, device_map="cuda", torch_dtype=torch.bfloat16).to(device).eval()
#             self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config).to(device).eval()
#         else:
#             self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_config).to(device).eval()
#         print(self.model)
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_config)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

#     def eval_(self, seq_len: int = 16, batch_size: int = 1, num_runs: int = NUM_RUNS, export: bool = EXPORT, custom_ops = custom_ops, inputs = None):
#         self.model_name = f'{self.model_name}_{self.device}_{batch_size}_{seq_len}'
#         prompt = gen_random_prompt(seq_len)
#         inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
#         # outputs = self.model.generate(**inputs, max_new_tokens=100)
#         # print(self.tokenizer.decode(outputs[0]))
#         profile_model(self.model_name, self.model, inputs, custom_ops, num_runs, self.device, False, out_dir, export)


def mistral_MoE(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', weights: str = None): 
    model = LMProfile(model_name = 'mistral_MoE', model_config = weights or "mistralai/Mixtral-8x7B-v0.1", device = device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def mamba(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = MambaProfile('mamba-130m', weights or 'state-spaces/mamba-130m', device)
    # model = MambaProfile('mamba-370m', weights or 'state-spaces/mamba-370m', device)
    # model = MambaProfile('mamba-790m', weights or 'state-spaces/mamba-790m', device)
    # model = MambaProfile('mamba-1.4b', weights or 'state-spaces/mamba-1.4b', device)
    # model = MambaProfile('mamba-2.8b', weights or 'state-spaces/mamba-2.8b', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def mamba2(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    # model = MambaProfile('mamba2-130m', weights or 'state-spaces/mamba2-130m', device)
    # model = MambaProfile('mamba2-370m', weights or 'state-spaces/mamba2-370m', device)
    model = MambaProfile('mamba2-780m', weights or 'state-spaces/mamba2-780m', device)
    # model = MambaProfile('mamba2-1.3b', weights or 'state-spaces/mamba2-1.3b', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def mamba2_generate(seq_len: int = 8, max_num_tokens: int = 256, device: str = 'cuda', weights: str = None):
    path_weights = weights or "state-spaces/mamba2-780m"
    # path_weights = "state-spaces/mamba2-1.3b"
    model = MambaProfile('mamba2', path_weights, device)
    model.eval_gen_(seq_len, max_num_tokens, NUM_RUNS, EXPORT, custom_ops)
    del model

def mamba_hf(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('mamba-130m-hf', weights or 'state-spaces/mamba-130m-hf', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def zamba2(seq_len: int = 8192, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    # model = Zamba2Profile('zamba2', weights or 'Zyphra/Zamba2-1.2B', device)
    # model = LMProfile('zamba2', weights or 'Zyphra/Zamba2-1.2B', device)
    model = LMProfile('zamba2', weights or 'Zyphra/Zamba2-1.2B-Instruct-v2', device)
    # model = LMProfile('zamba2', weights or 'Zyphra/Zamba2-2.7B-Instruct-v2', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def hymba(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    # model = LMProfile('hymba', weights or 'nvidia/Hymba-1.5B-Base', device)
    model = LMProfile('hymba', weights or 'nvidia/Hymba-1.5B-Instruct', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def qwen25_instruct(seq_len: int = 8, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('qwen25-instruct', weights or 'Qwen/Qwen2.5-0.5B-Instruct', device)
    # model = LMProfile('qwen25-instruct-int4', weights or 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', device)
    # model = LMProfile('qwen25-1.5b-instruct', weights or 'Qwen/Qwen2.5-1.5B-Instruct', device)
    # model = LMProfile('qwen25-1.5b-instruct-int4', weights or 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4', device)
    
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    # model.eval_memory(seq_len, batch_size, EXPORT)
    # model.eval_memory_generation(seq_len, batch_size, EXPORT, num_new_tokens=1)
    # model.eval_shape(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def qwen25_instruct_generate(seq_len: int = 8, max_num_tokens: int = 256, device: str = 'cuda', weights: str = None):
    path_weights = weights or "Qwen/Qwen2.5-0.5B-Instruct"
    # path_weights = weights or "Qwen/Qwen2.5-1.5B-Instruct"
    model = LMProfile('qwen25-instruct', path_weights, device)
    model.eval_gen_(seq_len, max_num_tokens, NUM_RUNS, EXPORT, custom_ops)
    del model


def tinyllama(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('tinyllama', weights or "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device)
    # model = LMProfile('tinyllama', "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def gpt_neo(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('gpt-neo-125m', weights or 'EleutherAI/gpt-neo-125m', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def phi3(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('phi3', weights or 'microsoft/Phi-3-mini-128k-instruct', device)
    # model = LMProfile('phi3', weights or 'leliuga/Phi-3-mini-128k-instruct-bnb-4bit', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def llama3_2(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('llama3_2', weights or 'meta-llama/Llama-3.2-1B-Instruct', device)
    # model = LMProfile('llama3_2', weights or 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def falcon_h1(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('falcon-h1-0.5b', weights or 'tiiuae/Falcon-H1-0.5B-Base', device)
    # model = LMProfile('falcon-h1-0.5b', weights or 'tiiuae/Falcon-H1-0.5B-Instruct', device)
    # model = LMProfile('falcon-h1', weights or 'tiiuae/Falcon-H1-1.5B-Instruct', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def falcon_h1_generate(seq_len: int = 8, max_num_tokens: int = 1, device: str = 'cuda', weights: str = None):
    path_weights = weights or "tiiuae/Falcon-H1-0.5B-Base"
    # path_weights = weights or "tiiuae/Falcon-H1-1.5B-Instruct"
    model = LMProfile('falcon-h1', path_weights, device)
    model.eval_gen_(seq_len, max_num_tokens, NUM_RUNS, EXPORT, custom_ops)
    del model

def nemotron_flash(seq_len: int = 64, batch_size: int = 1, device: str = 'cuda', weights: str = None):
    model = LMProfile('nemotron-flash-1b', weights or 'nvidia/Nemotron-Flash-1B', device)
    model.eval_(seq_len, batch_size, NUM_RUNS, EXPORT, custom_ops)
    del model

def nemotron_flash_generate(seq_len: int = 8, max_num_tokens: int = 1, device: str = 'cuda', path_weights: str ="nvidia/Nemotron-Flash-1B"):
    model = LMProfile('nemotron-flash-1b', path_weights, device)
    model.eval_gen_(seq_len, max_num_tokens, NUM_RUNS, EXPORT, custom_ops)
    del model

profiling_functions = {
    'qwen25-instruct': qwen25_instruct,
    'tinyllama': tinyllama,
    'gpt-neo-125m': gpt_neo,
    'phi3': phi3,
    'llama3_2': llama3_2,
    'mistral':mistral_MoE,
    'mamba': mamba,
    'mamba2': mamba2,
    'mamba_hf': mamba_hf,
    'zamba2': zamba2,
    'hymba': hymba,
    'falcon-h1': falcon_h1,
    'nemotron-flash': nemotron_flash,
}

def parse_arguments(): 
    parser = argparse.ArgumentParser(description ='Torch Profiling')
    
    parser.add_argument ("--model_name", dest="model_name",
                        required = True,  type = str, help = "Name of Model to profile", choices = ['llama3', 'qwen25-instruct', 'tinyllama', 'gpt-neo-125m', 'phi3', 'llama3_2', 'mistral', 'mamba', 'mamba2', 'mamba_hf', 'zamba2', 'hymba', 'falcon-h1', 'nemotron-flash'])
    
    parser.add_argument ("--model_weights", dest="weights",
                        required = False,  type = str, help = "Path to local weights")
    
    parser.add_argument ("--batch_size", dest="batch_size",
                        required = True,  type = int, help = "batch_size")
    
    parser.add_argument ("--seq_len", dest="seq_len",
                        required = False,  type = int, help = "Input Sequence Length for Language Models")
    
    parser.add_argument ("--device", dest="device",
                        required = True,  type = str, help = "cpu or cuda")
    
    parser.add_argument ("--out_dir", dest="out_dir",
                        required = False,  type = str, help = "Directory to store output csv files")
    
    args = parser.parse_args()
    return args

def main ():
    args = parse_arguments()
    
    model_name = args.model_name
    weight_path = args.weights if args.weights is not None else None
    batch_size = args.batch_size
    seq_len = args.seq_len if args.seq_len is not None else 1
    device = args.device

    global out_dir
    out_dir = args.out_dir if args.out_dir is not None else out_dir

    print(f'Profiling {model_name} on {device}')
    st = time.perf_counter()
    profiling_functions[model_name](seq_len, batch_size, device, weights=weight_path)
    et = time.perf_counter()
    print (f"Finished Profiling {model_name} on {device} in {et-st:.2f} seconds")



def debug(): 
    # Run mamba with increasing sequence lengths
    # # seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # # seq_lengths = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152, 57344]
    # # seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 40960, 49152, 57344]
    # # seq_lengths = [1024, 2048, 4096, 8192, 16384, 24576, 32768]
    # seq_lengths = [8192, 16384, 24576, 32768]
    # # seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]
    
    # # print(f"\n=== Starting profiling run - timing data will be saved to ttft_logs/iteration_times.csv ===\n")
    # print(f"\n=== Starting profiling run - timing data will be saved to tpot_logs/iteration_times.csv ===\n")

    
    # for seq_len in seq_lengths:
    # #     print(f"\n=== Running mamba with sequence length: {seq_len} ===\n")
    # #     # mamba(seq_len=seq_len, batch_size=1, device='cuda')
    # #     # mamba2(seq_len=seq_len, batch_size=1, device='cuda')
    # #     # zamba2(seq_len=seq_len, batch_size=1, device='cuda')
    # #     hymba(seq_len=seq_len, batch_size=1, device='cuda')
    #     # print(f"\n=== Running model with sequence length: {seq_len} ===\n")
    #     # falcon_h1(seq_len=seq_len, batch_size=1, device='cuda')
    #     # qwen25_instruct(seq_len=seq_len, batch_size=1, device='cuda')
    #     # mamba2(seq_len=seq_len, batch_size=1, device='cuda')
    #     # qwen25_instruct_generate(seq_len=seq_len, max_num_tokens=256, device='cuda')
    #     # falcon_h1_generate(seq_len=seq_len, max_num_tokens=256, device='cuda')
    #     mamba2_generate(seq_len=seq_len, max_num_tokens=256, device='cuda')
    #     time.sleep(5)  # Pause between runs to stabilize system resources
    
    # Commented out previous code
    # llama2_dynamo_generate(seq_len=2048)
    # swin_base_dynamo()
    # llama2_dynamo(seq_len = 512)
    # swin_tiny()
    # gpt2()
    # gpt2(seq_len = 1024)
    qwen25_instruct(seq_len=8192)
    # qwen25_instruct_generate(seq_len=1024, max_num_tokens=256)
    # falcon_h1(seq_len=24576)
    # falcon_h1_generate(seq_len=30000, max_num_tokens=256)
    # nemotron_flash(seq_len=1024)
    # tinyllama(seq_len = 1024, device = 'cuda')
    # opt(seq_len = 64)
    # gpt_neo(seq_len = 1024)
    # phi3(seq_len = 64)
    # llama3_2(seq_len = 1024)
    # mamba(seq_len=32768, batch_size=1, device='cuda')
    # mamba2(seq_len=16384, batch_size=1, device='cuda')
    # mamba2_generate(seq_len=1024, max_num_tokens=32768)
    # mamba_hf(seq_len= 8192)
    # zamba2(seq_len=8192, batch_size=1, device='cuda')
    # hymba(seq_len=8192)
    # llama2_shape_shape(seq_len = 1, )
    #llama2_generate(seq_len = 2048, max_num_tokens=8192)
    # llama2_shape_generate(seq_len = 2048, max_num_tokens=1)
    # llama2 (64, 1, 'cuda')
    # llama2 (2048, 1, 'cuda')
    #llama3_8bit(2048,1,'cuda')
    #model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B-INT8").eval()
    #print(model)
    #segformer_b5()
    # detr_shape()
    # detr_shape(batch_size=2)
    # detr_shape(batch_size=4)
    # detr_shape(batch_size=8)

    # segformer_shape(batch_size=1)
    # segformer_shape(batch_size=2)
    # segformer_shape(batch_size=4)
    # segformer_shape(batch_size=8)

    # llama3(seq_len = 64, device = 'cuda')
    # llama3(seq_len = 128, device = 'cpu')
    # llama3(seq_len = 128, device = 'cuda')
    # llama3(seq_len = 512, device = 'cpu')
    # llama3(seq_len = 512, device = 'cuda')
    # llama3(seq_len = 1024, device = 'cpu')
    # llama3(seq_len = 1024, device = 'cuda')

    # bert(seq_len = 64, device = 'cuda')
    # bert(seq_len = 128, device = 'cpu')
    # bert(seq_len = 128, device = 'cuda')

    # bert_large(seq_len = 128, device = 'cpu')
    # bert_large(seq_len = 128, device = 'cuda')

    # hf_vit_base_16(device = 'cpu')
    # hf_vit_base_16(device='cuda')
    

    # hf_vit_huge(device = 'cpu')
    # hf_vit_huge(device='cuda')

    # mistral_MoE(seq_len=2048, device = 'cuda')

    #swin_small_shape()
    







    pass 

if __name__ == "__main__": 
    main()
    # debug()
    #energy()
    
    pass
