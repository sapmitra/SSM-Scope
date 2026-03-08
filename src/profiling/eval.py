'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-02 03:00:50
 # @ Description: Core PyTorch profiler engine.  Records operator-level
 #                 CPU/CUDA timing (TTFT, TPOT), energy, and tensor shapes
 #                 for LM and SSM models using torch.profiler.
 '''

import torch 
import transformers 
import os 
import subprocess
import gc
import signal
import argparse 
import datasets
#import torchvision 
from torch.utils.data import Subset
from torch.utils.data import DataLoader 
import random
import time
from typing import List
import csv
#from torchvision import datasets
import pandas as pd
from profiling.power_logger import parse_energy_from_log
torch.manual_seed(1969)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

gemm_ops = ["aten::mm", "aten::matmul", "aten::bmm", "aten::linear", 
       "aten::addmm", "aten::addbmm", "aten::baddbmm", "aten::mv","aten::dot",       
       "aten::ger", "aten::matmul.out", "aten::scaled_dot_product_attention", 
       "aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::conv_tbc", "aten::conv_transpose1d", "aten::conv_transpose2d", "aten::conv_transpose3d", "aten::slow_conv3d", "aten::slow_conv_dilated2d", "aten::slow_conv_dilated3d", "aten::slow_conv_transpose2d", "aten::slow_conv_transpose3d", "aten::thnn_conv2d", "aten::thnn_conv_depthwise2d","aten::scaled_dot_product_attention", "aten::linear",'wqlinearmmfunction',
       "conv1d", 'aten::_native_multi_head_attention', 'segformerdwconv', 'aten::einsum', 'aten::roll','aten::_scaled_dot_product_efficient_attention',
          ]

ssm_scan_ops = ["mambainnerfn", "mambasplitconv1dscancombinedfn", "mambachunkscancombinedfn", 'selectivescanfn', 'causalconv1dfn',]

# Ops made of multiple aten operators to be captured 
ops = ["conv1d", "wqlinear_gemm", "llamarmsnorm"]


def debug_test_aggregate (prof, op, filename): 

    result_rows = [] 
    for op in ops: 
        pass


def test_aggregate(prof, ops, filename): 
    st = time.perf_counter()
    reshape = 0
    linear = 0 
    matmul = 0
    # Prepare the result list
    result_rows = []
    op_dict = {} 


    # Process each operation
    for op in ops:
        op_rows = []
        for e in prof.profiler.function_events:
            if e.name == "Inference_prof" : #'torch._C._autograd.DeviceType'
                #print (e.device_type)
                #print (type(e.device_type))
                if (e.device_type == torch._C._autograd.DeviceType.CPU):
                    op_rows.append({
                        'name': e.name.lower(),
                        'cpu_time (us)': e.cpu_time,
                        'cuda_time (us)': e.device_time,
                        'total_time (us)': e.cpu_time + e.cuda_time,
                        'count':1
                    })
            elif e.cpu_parent and "_prof" in e.cpu_parent.name and e.name == op:
                op_rows.append({
                    'name': e.name.lower(),
                    'cpu_time (us)': e.cpu_time,
                    'cuda_time (us)': e.cuda_time,
                    'total_time (us)': e.cpu_time + e.cuda_time, 
                    'count':1
                })
                if (e.name=="aten::reshape"):
                    reshape += 1 
                elif (e.name=="aten::linear" ): 
                    linear+=1
                elif (e.name=="aten::matmul"):
                    matmul+=1

        
        # Aggregate the operation results
        df = pd.DataFrame(op_rows)
        cpu = df['cpu_time (us)'].sum() if not df.empty else 0
        cuda = df['cuda_time (us)'].sum() if not df.empty else 0
        count = df['count'].sum() if not df.empty else 0
        result_rows.append({
            'name': op.lower(),
            'cpu_time (us)': cpu,
            'cuda_time (us)': cuda,
            'total_time (us)': cpu + cuda, 
            'count': count
        })
    
    # Create the final DataFrame
    df_ = pd.DataFrame(result_rows)
    
    # Save to CSV
    df_.to_csv(filename)
    
    et = time.perf_counter()
    print(f"Time to Summarize Files {et - st} s")
    return

def aggreagate (prof, ops, filename): 
    print (f"Aggregating profiles:")
    st = time.perf_counter()
    columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'count']
    df_= pd.DataFrame(columns=columns)

    for op in ops:
        df = pd.DataFrame(columns=columns)
        skip_first = False
        cpu = 0
        cuda = 0 
        count = 0
        for e in prof.profiler.function_events: 
            
            if not(e.cpu_parent is None):
                if (e.cpu_parent.name == "Inference_prof" or e.name=="Inference_prof" or e.cpu_parent.name =="aten::multinomial") and (e.name == op) and (e.name!="aten::multinomial"):## surgery multinomial 
                    # new_entry = {'name':e.name.lower(), 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
                    # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True) 
                    cpu += e.cpu_time
                    cuda += e.device_time
                    count += 1
                if ("8bit" in e.cpu_parent.name):
                    op_ = op.replace("_q8bit","") 
                    if (e.name ==op_): 
                        # new_entry = {'name':f"{e.name.lower()}_q8bit", 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
                        # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True)
                        cpu += e.cpu_time
                        cuda += e.cuda_time
                        count += 1

        # cpu = df['cpu_time (us)'].sum()
        # cuda = df['cuda_time (us)'].sum()
        # count = df['count'].sum()
        # del df 
        new_entry_ = {'name':op.lower(), 'cpu_time (us)':[cpu], 'cuda_time (us)':[cuda], 'total_time (us)':[cpu+cuda], 'count':[count] }
        df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    df_.to_csv(filename)
    et = time.perf_counter() 
    print (f"Finished Aggregating Profiles: {et - st} s")
    return     

def generate_report (filename): 
    print ("Generating CSV Files")
    st = time.perf_counter()
    df = pd.read_csv(filename)
    df = df.drop(df.columns[0], axis=1)
   

    inference = df[df['name'].str.contains('inference_')]

    gemm_ops_ = []
    non_gemm_ops_ = []
    ssm_scan_ops_ = []
    uniq = df['name'].unique().tolist()
    for i in uniq: 
        if (i != "inference_prof"):
            if ("_prof" in i):
                for j in gemm_ops: 
                    if (i.replace('_prof',"") in j):
                        gemm_ops_.append(i)
                        break 
            else: 
                non_gemm_ops_.append(i)
        
    gemm_ops_ = gemm_ops_ + gemm_ops
    # print (gemm_ops_)
    # print ("GEMM OPS DONE")
    ssm_scan_ops_ = ssm_scan_ops_ + ssm_scan_ops

    gemm = df[df ["name"].isin(gemm_ops_)]
    ssm_scan = df[df ["name"].isin(ssm_scan_ops_)]
    non_gemm = df[~df["name"].isin(gemm_ops_) & ~df["name"].isin(ssm_scan_ops_)]
    non_gemm = non_gemm[~ non_gemm['name'].str.contains('profiler|inference|cuda')]
   
    new_gemm_entry = {'name':"GEMM", 'cpu_time (us)':[gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[gemm['cuda_time (us)'].sum()], 'total_time (us)':[gemm['total_time (us)'].sum()]} 
    
    new_non_gemm_entry = {'name':"NonGEMM", 'cpu_time (us)':[non_gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[non_gemm['cuda_time (us)'].sum()], 'total_time (us)':[non_gemm['total_time (us)'].sum()]} 

    new_ssm_scan_entry = {'name':"SSM_Scan", 'cpu_time (us)':[ssm_scan['cpu_time (us)'].sum()], 'cuda_time (us)':[ssm_scan['cuda_time (us)'].sum()], 'total_time (us)':[ssm_scan['total_time (us)'].sum()]}

    new_inference_entry = {'name':"Inference", 'cpu_time (us)':[inference['cpu_time (us)'].sum()], 'cuda_time (us)':[inference['cuda_time (us)'].sum()], 'total_time (us)':[inference['total_time (us)'].sum()]} 

    df = pd.concat((df, pd.DataFrame(new_gemm_entry)))
    df = pd.concat((df, pd.DataFrame(new_non_gemm_entry)))
    df = pd.concat((df, pd.DataFrame(new_ssm_scan_entry)))
    
 
    gemm = pd.concat((gemm, pd.DataFrame(new_gemm_entry)))
    gemm = pd.concat((gemm, pd.DataFrame(new_inference_entry)))

    ssm_scan = pd.concat((ssm_scan, pd.DataFrame(new_ssm_scan_entry)))
    ssm_scan = pd.concat((ssm_scan, pd.DataFrame(new_inference_entry)))

    non_gemm = pd.concat((non_gemm, pd.DataFrame(new_non_gemm_entry)))
    non_gemm = pd.concat((non_gemm, pd.DataFrame(new_inference_entry)))

    gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/gemm.csv")
    ssm_scan.to_csv(f"{filename.rsplit('/', 1)[0]}/ssm_scan.csv")
    non_gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/non_gemm.csv")
    df.to_csv(f'{filename}')
    et = time.perf_counter() 
    print (f"Finished generating Report {et - st} s")

def _analyze_prof (prof, filename, custom): 
    print ("Analyzing Profile logs")
    #print (prof.key_averages())
    avg = prof.key_averages() 
    ops = []
    for a in avg: 
        if ("Step" in a.key): 
            ops.append(a.key)
        if ("Inference_" in a.key): 
            ops.append(a.key)
            
        
        # if custom:
        #     if not (a.cpu_parent is None): 
        #         if ("_prof" in a.cpu_parent.name):
        #             if not (a.key in ops): 
        #                 ops.append(a.key)
        #         else: 
        #             if ("_prof" in a.key) and not (a.key in ops): 
        #                 ops.append(a.key)
                    
                    
        multinomial = True
        if not (a.cpu_parent is None):
            if ("Inference_" in a.key ): 
                for child in a.cpu_children: 
                    
                    if not (child.name in ops):
                        #ops.append(child.name)
                        if ("8bit" in child.name):
                            ops.append(child.name) 
                            for child8 in child.cpu_children:
                                child8_ = f"{child8.name}_q8bit"
                                if not (child8_ in ops):
                                    ops.append(child8_)
                        else:
                            if (child.key == "aten::multinomial") and multinomial: 
                                for child_ in child.cpu_children:
                                    if not(child_ in ops):
                                        ops.append(child_.name)
                                multinomial = False 
                            else:
                                if (child.key != "aten::multinomial"):
                                    ops.append(child.name)
                inf =  a
                break
    if (custom): 
        aggregate_custom(prof, ops, filename)
        #test_aggregate(prof, ops, filename)
        generate_report(filename)

    else: 
        print (ops)
        
        aggreagate (prof, ops, filename)
        #test_aggregate(prof, ops, filename)
        
        generate_report(filename)
    return


########### Shape Recording ####################
def aggreagate_shape (prof, ops_to_be_recorded, filename): 
    
    print (f"Aggregating profiles:")
    st = time.perf_counter()
    columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'shape', 'count']
    df_= pd.DataFrame(columns=columns)
    avg = prof.key_averages(group_by_input_shape=True)
    print (ops_to_be_recorded)
    for op in avg: 
        if op.key in ops_to_be_recorded:
            shape = str(op.input_shapes)
            count = op.count
            mem_cuda = op.device_memory_usage 
            mem_cpu = op.cpu_memory_usage 
            new_entry_ = {'name':op.key.lower(), 'count':[count], 'shape':shape, 'mem_cpu':mem_cpu, 'mem_cuda':mem_cuda }
            df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    df_.to_csv(filename)
    et = time.perf_counter() 
    print (f"Finished Aggregating Profiles: {et - st} s")
    return
    
    # print (f"Aggregating profiles:")
    # st = time.perf_counter()
    # columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'shape', 'count']
    # df_= pd.DataFrame(columns=columns)

    # for op in ops:
    #     df = pd.DataFrame(columns=columns)
    #     skip_first = False
    #     cpu = 0
    #     cuda = 0 
    #     shape = []
    #     count = 0
    #     for e in prof.profiler.function_events: 
            
    #         if not(e.cpu_parent is None):
    #             if (e.cpu_parent.name == "Inference_prof" or e.name=="Inference_prof") and (e.name == op): 
    #                 shape.append(e.ipnut_shapes)
    #                 count += 1
    #             if ("8bit" in e.cpu_parent.name):
    #                 op_ = op.replace("_q8bit","") 
    #                 if (e.name ==op_): 
    #                     # new_entry = {'name':f"{e.name.lower()}_q8bit", 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
    #                     # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True)
    #                     cpu += e.cpu_time
    #                     cuda += e.cuda_time
    #                     count += 1

    #     # cpu = df['cpu_time (us)'].sum()
    #     # cuda = df['cuda_time (us)'].sum()
    #     # count = df['count'].sum()
    #     # del df 
    #     new_entry_ = {'name':op.lower(), 'count':[count], 'shape':shape }
    #     df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    # df_.to_csv(filename)
    # et = time.perf_counter() 
    # print (f"Finished Aggregating Profiles: {et - st} s")
    # return     

def generate_report_shape (filename): 
    print ("Generating CSV Files")
    st = time.perf_counter()
    df = pd.read_csv(filename)
    df = df.drop(df.columns[0], axis=1)
   

    inference = df[df['name'].str.contains('Inference_')]

    gemm_ops_ = []
    non_gemm_ops_ = []
    uniq = df['name'].unique().tolist()
    for i in uniq: 
        if (i != "inference_prof"):
            if ("_prof" in i):
                for j in gemm_ops: 
                    if (i.replace('_prof',"") in j):
                        gemm_ops_.append(i)
                        break 
            else: 
                non_gemm_ops_.append(i)
        
    gemm_ops_ = gemm_ops_ + gemm_ops
    # print (gemm_ops_)
    # print ("GEMM OPS DONE")

    gemm = df[df ["name"].isin(gemm_ops_)]
    non_gemm = df[~df["name"].isin(gemm_ops_)]
    non_gemm = non_gemm[~ non_gemm['name'].str.contains('profiler|inference')]
   
    #new_gemm_entry = {'name':"GEMM", 'cpu_time (us)':[gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[gemm['cuda_time (us)'].sum()], 'total_time (us)':[gemm['total_time (us)'].sum()]} 
    
    #new_non_gemm_entry = {'name':"NonGEMM", 'cpu_time (us)':[non_gemm['cpu_time (us)'].sum()], 'cuda_time (us)':[non_gemm['cuda_time (us)'].sum()], 'total_time (us)':[non_gemm['total_time (us)'].sum()]} 
    
    #new_inference_entry = {'name':"Inference", 'cpu_time (us)':[inference['cpu_time (us)'].sum()], 'cuda_time (us)':[inference['cuda_time (us)'].sum()], 'total_time (us)':[inference['total_time (us)'].sum()]} 

    #df = pd.concat((df, pd.DataFrame(new_gemm_entry)))
    #df = pd.concat((df, pd.DataFrame(new_non_gemm_entry)))
    
 
    #gemm = pd.concat((gemm, pd.DataFrame(new_gemm_entry)))
    #gemm = pd.concat((gemm, pd.DataFrame(new_inference_entry)))
    
    #non_gemm = pd.concat((non_gemm, pd.DataFrame(new_non_gemm_entry)))
    #non_gemm = pd.concat((non_gemm, pd.DataFrame(new_inference_entry)))

    gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/gemm.csv")
    non_gemm.to_csv(f"{filename.rsplit('/', 1)[0]}/non_gemm.csv")
    df.to_csv(f'{filename}')
    et = time.perf_counter() 
    print (f"Finished generating Report {et - st} s")

def _analyze_prof_shape (prof, filename, ops_to_be_recorded): 
    print ("Analyzing Profile logs")
    #print (prof.key_averages())
    avg = prof.key_averages(group_by_input_shape=True) 
    with open("shape_stat.txt", "w") as f:
        print(avg, file=f)
    
    ops = []
    for a in avg: 
        if ("Step" in a.key): 
            ops.append(a.key)
        if ("Inference_" in a.key): 
            ops.append(a.key)
            
        
        # if custom:
        #     if not (a.cpu_parent is None): 
        #         if ("_prof" in a.cpu_parent.name):
        #             if not (a.key in ops): 
        #                 ops.append(a.key)
        #         else: 
        #             if ("_prof" in a.key) and not (a.key in ops): 
        #                 ops.append(a.key)
                    
                    
        multinomial = True
        if not (a.cpu_parent is None):
            if ("Inference_" in a.key ): 
                for child in a.cpu_children: 
                    
                    if not (child.name in ops):
                        #ops.append(child.name)
                        if ("8bit" in child.name):
                            ops.append(child.name) 
                            for child8 in child.cpu_children:
                                child8_ = f"{child8.name}_q8bit"
                                if not (child8_ in ops):
                                    ops.append(child8_)
                        
                        else:
                            if (child.key == "aten::multinomial") and multinomial: 
                                for child_ in child.cpu_children:
                                    if not(child_ in ops):
                                        ops.append(child_.name)
                                multinomial = False 
                            else:
                                if (child.key != "aten::multinomial"):
                                    ops.append(child.name)
                inf =  a
                break

    print (ops)
    
    aggreagate_shape (prof, ops, filename)
    #test_aggregate(prof, ops, filename)
    
    generate_report_shape(filename)
    return

########## End Shape Recording ################

######### DYNAMO ###################

def _analyze_prof_dynamo (prof, filename, custom): 
    print ("Analyzing Profile logs")
    #print (prof.key_averages())
    avg = prof.key_averages() 
    ops = []
    for a in avg: 
        # if ("Step" in a.key): 
        #     ops.append(a.key)
        # if ("Inference_" in a.key): 
        #     ops.append(a.key)
            
        
        # if custom:
        #     if not (a.cpu_parent is None): 
        #         if ("_prof" in a.cpu_parent.name):
        #             if not (a.key in ops): 
        #                 ops.append(a.key)
        #         else: 
        #             if ("_prof" in a.key) and not (a.key in ops): 
        #                 ops.append(a.key)
                    
                    
        multinomial = True
        if not (a.cpu_parent is None):
            if ("Inference_prof_7" in a.key): 
                for child in a.cpu_children: 
                    if ("Torch-Compiled Region" in child.key):
                        for child_ in child.cpu_children: 
                                if ("CompiledFunction" in child_.key):
                                    for child_op in child_.cpu_children:
                                        if not (child_op.name in ops):
                                            ops.append(child_op.name)
                                    break                      
                inf =  a
                break
    if (custom): 
        aggregate_custom(prof, ops, filename)
        #test_aggregate(prof, ops, filename)
        generate_report(filename)

    else: 
        print (ops)
        
        aggreagate_dynamo (prof, ops, filename)
        #test_aggregate(prof, ops, filename)
        
        generate_report(filename)
    return


def aggreagate_dynamo (prof, ops, filename): 
    print (f"Aggregating profiles:")
    st = time.perf_counter()
    columns = ['name', 'cpu_time (us)', 'cuda_time (us)', 'total_time (us)', 'count']
    df_= pd.DataFrame(columns=columns)

    for op in ops:
        df = pd.DataFrame(columns=columns)
        skip_first = False
        cpu = 0
        cuda = 0 
        count = 0
        for e in prof.profiler.function_events: 
            
            if not(e.cpu_parent is None):
                if (e.cpu_parent.name == "CompiledFunction") and (e.name == op):# and (not (e.cpu_parent.cpu_parent.name in [f'Inference_prof_{i}' for i in range (5)])):## surgery multinomial 
                    # new_entry = {'name':e.name.lower(), 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
                    # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True) 
                    cpu += e.cpu_time
                    cuda += e.device_time
                    count += 1
                if ("8bit" in e.cpu_parent.name):
                    op_ = op.replace("_q8bit","") 
                    if (e.name ==op_): 
                        # new_entry = {'name':f"{e.name.lower()}_q8bit", 'cpu_time (us)':[e.cpu_time], 'cuda_time (us)':[e.cuda_time], 'total_time (us)':[e.cpu_time + e.cuda_time], 'count':[1]}
                        # df = pd.concat((df, pd.DataFrame(new_entry)), ignore_index = True)
                        cpu += e.cpu_time
                        cuda += e.cuda_time
                        count += 1

        # cpu = df['cpu_time (us)'].sum()
        # cuda = df['cuda_time (us)'].sum()
        # count = df['count'].sum()
        # del df 
        new_entry_ = {'name':op.lower(), 'cpu_time (us)':[cpu], 'cuda_time (us)':[cuda], 'total_time (us)':[cpu+cuda], 'count':[count] }
        df_= pd.concat((df_, pd.DataFrame(new_entry_)), ignore_index = True)
    df_.to_csv(filename)
    et = time.perf_counter() 
    print (f"Finished Aggregating Profiles: {et - st} s")
    return     



###############################################


def replace_forward(module, ops = None): 
    _old = getattr(module, "forward")
    
    def new_forward(*args, **kwargs): 
        with torch.profiler.record_function(f"{module.__class__.__name__}_prof"): 
            return _old(*args, **kwargs)
    module_name = module.__class__.__name__.lower()
    
    if (ops is None):
        if not ( ("model" in module_name) or ("causallm" in module_name) or ("attention" in module_name) or ("decoder" in module_name) or ("mlp" in module_name) or ("sequential" in module_name) or ("block" in module_name)):
            setattr(module, "forward", new_forward)
    else: 
        if (module_name in ops): 
            setattr(module, "forward", new_forward)

@torch.no_grad()
def profile_model (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True,
                    export_profile: bool = False): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    # Create ttft_logs directory if it doesn't exist
    ttft_logs_dir = "ttft_logs"
    os.makedirs(ttft_logs_dir, exist_ok=True)
    
    # CSV file path for timing logs
    timing_csv_path = os.path.join(ttft_logs_dir, "iteration_times.csv")
    
    # Check if CSV exists to determine if we need to write headers
    file_exists = os.path.isfile(timing_csv_path)
    
    # Extract sequence length from input
    seq_len = input_['input_ids'].shape[1] if 'input_ids' in input_ else input_.get('pixel_values', torch.tensor([[]])).shape[-1]
    
    # Store timing data for each iteration
    timing_data = []
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                    
                    # Record timing data for active iterations
                    if n >= skip_first + wait + warmup:
                        timing_data.append({
                            'model_name': model_name,
                            'seq_length': seq_len,
                            'iteration': n - (skip_first + wait + warmup),
                            'time_seconds': et,
                            'device': device,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                prof.step()
                del out
                gc.collect
    
    # Write timing data to CSV
    with open(timing_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'seq_length', 'iteration', 'time_seconds', 'device', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write all timing data
        writer.writerows(timing_data)
    
    # Calculate and print average execution time
    if timing_data:
        avg_time = sum(item['time_seconds'] for item in timing_data) / len(timing_data)
        print(f"Average execution time: {avg_time} s")
    
    print(f"Timing data saved to {timing_csv_path}")


    
    # Operator-breakdown export: enabled when called with export_profile=True
    if export_profile:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        prof_out_dir = f'{out_dir}/{model_name}'
        os.system(f"mkdir -p {prof_out_dir}")
        if export:
            prof.export_chrome_trace(f"{prof_out_dir}/{model_name}.json")
        filename = f"{prof_out_dir}/{model_name}.csv"
        _analyze_prof(prof, filename, False)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # out_dir = f'{out_dir}/{model_name}'    
    # os.system(f"mkdir -p {out_dir}") 
    
    
    # if (export): 
    #     prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    # filename = f"{out_dir}/{model_name}.csv"
    # _analyze_prof (prof, filename, custom)


@torch.no_grad()
def profile_model_mamba(model_name, 
                        model, 
                        input_, 
                        custom_ops_list, 
                        num_prof_runs, 
                        device, 
                        dynamo=False, 
                        out_dir="./non-gemm-out/", 
                        export=True,
                        tokenizer=None,
                        export_profile: bool = False): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    # fn = lambda: model.generate(
    #     input_ids=input_['input_ids'],
    #     max_length=50,
    #     cg=True,
    #     return_dict_in_generate=True,
    #     output_scores=True,
    #     enable_timing=False,
    #     temperature=1.0,
    #     top_k=1,
    #     top_p=1.0,
    #     min_p=0.0,
    #     repetition_penalty=1.0,
    # )
    fn = lambda: model(
        input_ids=input_['input_ids'],
    )
    # out = fn()
    # print(tokenizer.batch_decode(out.sequences.tolist()))
    schedule = torch.profiler.schedule(skip_first=skip_first, wait=wait, warmup=warmup, active=active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if custom:
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    # Create ttft_logs directory if it doesn't exist
    ttft_logs_dir = "ttft_logs"
    os.makedirs(ttft_logs_dir, exist_ok=True)
    
    # CSV file path for timing logs
    timing_csv_path = os.path.join(ttft_logs_dir, "iteration_times.csv")
    
    # Check if CSV exists to determine if we need to write headers
    file_exists = os.path.isfile(timing_csv_path)
    
    # Extract sequence length from input
    seq_len = input_['input_ids'].shape[1]
    
    # Store timing data for each iteration
    timing_data = []
    
    with torch.profiler.profile(schedule=schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for n in range(skip_first + wait + warmup + active):
            with torch.profiler.record_function(f"Inference_prof"):
                st = time.perf_counter()
                out = fn()
                # out = model(**input_)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                et = time.perf_counter() - st 
                print(f"Time for one inference = {et} s")
                
                # Record timing data for active iterations
                if n >= skip_first + wait + warmup:
                    timing_data.append({
                        'model_name': model_name,
                        'seq_length': seq_len,
                        'iteration': n - (skip_first + wait + warmup),
                        'time_seconds': et,
                        'device': device,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
            prof.step()
            del out
            gc.collect
    
    # Write timing data to CSV
    with open(timing_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'seq_length', 'iteration', 'time_seconds', 'device', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write all timing data
        writer.writerows(timing_data)
    
    print(f"Timing data saved to {timing_csv_path}")

    # Operator-breakdown export: enabled when called with export_profile=True
    # (e.g. from Fig_7 data-collection scripts). Off by default to preserve
    # the original timing-only behaviour used by all other callers.
    if export_profile:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        prof_out_dir = f'{out_dir}/{model_name}'
        os.system(f"mkdir -p {prof_out_dir}")
        if export:
            prof.export_chrome_trace(f"{prof_out_dir}/{model_name}.json")
        filename = f"{prof_out_dir}/{model_name}.csv"
        _analyze_prof(prof, filename, False)

@torch.no_grad()
def profile_model_mamba_generate(model_name, 
                        model, 
                        input_, 
                        custom_ops_list, 
                        num_prof_runs, 
                        device,
                        max_num_tokens=128,
                        dynamo=False, 
                        out_dir="./non-gemm-out/", 
                        export=True,
                        tokenizer=None,
                        csv_dir="tpot_logs",
                        csv_filename="tpot_times.csv"): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    # fn = lambda: model.generate(
    #     input_ids=input_['input_ids'],
    #     max_length=max_num_tokens,
    #     eos_token_id=None,
    #     temperature=1.0,
    #     top_k=1,
    #     top_p=1.0,
    #     min_p=0.0,
    #     repetition_penalty=1.0,
    #     return_dict_in_generate=True,
    #     output_scores=True,
    # )
    fn = lambda: model(
        input_ids=input_['input_ids'],
    )
    fn_generate = lambda: model.generate(
    input_ids=input_['input_ids'],
    max_length=input_['input_ids'].shape[1] + max_num_tokens,
    )
    
    schedule = torch.profiler.schedule(skip_first=skip_first, wait=wait, warmup=warmup, active=active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if custom:
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    # Create csv_dir directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    # CSV file path for timing logs
    tpot_csv_path = os.path.join(csv_dir, csv_filename)
    
    # Check if CSV exists to determine if we need to write headers
    file_exists = os.path.isfile(tpot_csv_path)
    
    # Extract sequence length from input
    seq_len = input_['input_ids'].shape[1]
    
    # Store TPOT data
    tpot_data = []
    
    # with torch.profiler.profile(schedule=schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for n in range(skip_first + wait + warmup):
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # prof.step()
        del out
        gc.collect

    # Calculate prefill time
    with torch.profiler.record_function(f"Prefill_prof"):
        st = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et_prefill = time.perf_counter() - st 
        print(f"Time for prefill = {et_prefill} s")
        
        
    with torch.profiler.record_function(f"Inference_prof"):
        st = time.perf_counter()
        out = fn_generate()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et = time.perf_counter() - st 
        decode_time = et - et_prefill
        tpot = decode_time / max_num_tokens
        print(f"Time for one generation = {et} s")
        print(f"Time per output token = {tpot} s")
        print(f"Throughput = {max_num_tokens / decode_time} tokens/s")

        if tokenizer and n == skip_first + wait + warmup:
            print("Generated text:", tokenizer.batch_decode(out.sequences.tolist())[0])
        
        # Record TPOT data
        tpot_data.append({
            'model_name': model_name,
            'input_seq_length': seq_len,
            'output_tokens': max_num_tokens,
            'prefill_time_seconds': et_prefill,
            'decode_time_seconds': decode_time,
            'total_time_seconds': et,
            'tpot_seconds': tpot,
            'throughput_tokens_per_sec': max_num_tokens / decode_time,
            'device': device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
            
    # prof.step()
    del out
    gc.collect
    
    # Write TPOT data to CSV
    with open(tpot_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'input_seq_length', 'output_tokens', 'prefill_time_seconds', 
                     'decode_time_seconds', 'total_time_seconds', 'tpot_seconds', 
                     'throughput_tokens_per_sec', 'device', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write all TPOT data
        writer.writerows(tpot_data)
    
    print(f"TPOT data saved to {tpot_csv_path}")

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    # out_dir = f'{out_dir}/{model_name}'    
    # os.system(f"mkdir -p {out_dir}") 
    
    # if export: 
    #     prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    
    filename = f"{out_dir}/{model_name}.csv"
    # _analyze_prof(prof, filename, custom)


def profile_model_dynamo (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = True, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import config
    config.cpp.enable_kernel_profile = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"]="1"
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"]="1"
    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof_{n}"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time for inference {n} = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof_dynamo (prof, filename, custom)



def profile_model_shape (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, 2
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = True
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes = True) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(**input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))

    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}_shape.json")
    

    filename = f"{out_dir}/{model_name}_shape.csv"
    _analyze_prof_shape (prof, filename, custom)

@torch.no_grad()
def profile_model_tv (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        st = time.perf_counter()
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
        et = time.perf_counter() 
        print (f"Exporting Trace {et - st} s")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof (prof, filename, custom)

def profile_model_dynamo_tv (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = True, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import config
    config.cpp.enable_kernel_profile = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"]="1"
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"]="1"
    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof_{n}"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time for inference {n} = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof_dynamo (prof, filename, custom)




def profile_model_tv_shape (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = True
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes = True) as prof:
            for n in range (skip_first + wait + warmup + active):
                with torch.profiler.record_function(f"Inference_prof"):
                    st = time.perf_counter()
                    #out = model.generate(**input_, max_length = 8)
                    out = model(input_)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    et = time.perf_counter () - st 
                    print (f"Time fot one inference = {et} s")
                prof.step()
                del out
                gc.collect
    
    
    out_dir = f'{out_dir}/{model_name}_shape'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        st = time.perf_counter()
        prof.export_chrome_trace(f"{out_dir}/{model_name}_shape.json")
        et = time.perf_counter() 
        print (f"Exporting Trace {et - st} s")
    

    filename = f"{out_dir}/{model_name}_shape.csv"
    _analyze_prof_shape (prof, filename, custom)

def profile_model_tv_energy(model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True):
    
    cmd = "nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.memory,utilization.gpu --format=csv --loop-ms=100 > power.log"
    #process = subprocess.Popen (cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os. setsid)
    for i in range (1000): 
        out = model(input_)
        torch.cuda.synchronize()
    #os.killpg(os.getpgid(process.pid), signal.SIGTERM)

@torch.no_grad()
def profile_model_generate (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    max_num_tokens = 128,
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True,
                    csv_dir = "tpot_logs",
                    csv_filename = "tpot_times.csv"): 

    skip_first, wait, warmup, active = 1, 2, 2, num_prof_runs
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    # Create csv_dir directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    # CSV file path for timing logs
    tpot_csv_path = os.path.join(csv_dir, csv_filename)
    
    # Check if CSV exists to determine if we need to write headers
    file_exists = os.path.isfile(tpot_csv_path)
    
    # Extract sequence length from input
    seq_len = input_['input_ids'].shape[1] if 'input_ids' in input_ else input_.get('pixel_values', torch.tensor([[]])).shape[-1]
    
    # Store TPOT data for each iteration
    tpot_data = []
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    num_layers = model.config.num_hidden_layers  # Number of transformer layers
    num_heads = model.config.num_attention_heads  # Number of attention heads
    head_dim = model.config.hidden_size // num_heads  # Dimension per head
    past_kv = tuple(
        (
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device),  # Random key tensor
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device)   # Random value tensor
        )
        for _ in range(num_layers)
    )
    # with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            
    for n in range (skip_first + wait + warmup):
        out = model(**input_, )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # prof.step()
    del out

    # calculate prefill time
    with torch.profiler.record_function(f"Inference_prof_prefill"):
        st = time.perf_counter()
        out = model(**input_, )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et_prefill = time.perf_counter () - st 
        print (f"Time for prefill = {et_prefill} s")
    time.sleep(5)
    with torch.profiler.record_function(f"Inference_prof"):
        st = time.perf_counter()
        out = model.generate(**input_, max_new_tokens = max_num_tokens, eos_token_id=None,)
        # out = model(input_ids = input_.input_ids[:,:1], past_key_values = past_kv, use_cache = True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et = time.perf_counter () - st 
        decode_time = et - et_prefill
        tpot = decode_time / max_num_tokens
        print (f"Time for one generation = {et} s")
        print (f"Time per output token = {tpot} s")
        print (f"Throughput = {max_num_tokens / decode_time} tokens/s")
        
        # Record TPOT data
        tpot_data.append({
            'model_name': model_name,
            'input_seq_length': seq_len,
            'output_tokens': max_num_tokens,
            'prefill_time_seconds': et_prefill,
            'decode_time_seconds': decode_time,
            'total_time_seconds': et,
            'tpot_seconds': tpot,
            'throughput_tokens_per_sec': max_num_tokens / decode_time,
            'device': device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

    del out
    gc.collect
    
    # Write TPOT data to CSV
    with open(tpot_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['model_name', 'input_seq_length', 'output_tokens', 'prefill_time_seconds', 
                     'decode_time_seconds', 'total_time_seconds', 'tpot_seconds', 
                     'throughput_tokens_per_sec', 'device', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write all TPOT data
        writer.writerows(tpot_data)
    
    print(f"TPOT data saved to {tpot_csv_path}")
    
    # out_dir = f'{out_dir}/{model_name}'    
    # os.system(f"mkdir -p {out_dir}") 
    
    
    # if (export): 
    #     prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    # _analyze_prof (prof, filename, custom)

# @torch.no_grad()
# def profile_model_energy(model_name, 
#                     model, 
#                     input_, 
#                     custom_ops_list, 
#                     num_prof_runs, 
#                     device, 
#                     dynamo = False, 
#                     out_dir = "./non-gemm-out/", 
#                     export = True):
    
#     cmd = "nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.memory,utilization.gpu --format=csv --loop-ms=100 > power.log"
#     process = subprocess.Popen (cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os. setsid)
#     for i in range (1000): 
#         out = model(**input_)
#         torch.cuda.synchronize()
#     os.killpg(os.getpgid(process.pid), signal.SIGTERM)

@torch.no_grad()
def profile_model_energy(model_name,
                    model,
                    input_,
                    custom_ops_list,
                    num_prof_runs,
                    device,
                    dynamo = False,
                    out_dir = "./non-gemm-out/",
                    export = True):
    """Measure prefill-phase energy (Joules) for a transformer LM.

    Runs ``max(num_prof_runs, 50)`` forward passes under nvidia-smi power
    logging, then delegates log parsing to
    ``profiling.power_logger.parse_energy_from_log``, and appends one row to
    ``energy_logs/energy_data.csv``.

    The raw power log is kept in ``power_logs/{model_name}_power.log``.
    """
    seq_len = input_['input_ids'].shape[1] if 'input_ids' in input_ else 0

    power_logs_dir  = "power_logs"
    energy_logs_dir = "energy_logs"
    os.makedirs(power_logs_dir,  exist_ok=True)
    os.makedirs(energy_logs_dir, exist_ok=True)

    # Warmup (not recorded)
    print("Starting warmup phase...")
    for _ in range(5):
        out = model(**input_)
        torch.cuda.synchronize()
        del out
        gc.collect()
        torch.cuda.empty_cache()

    # Measurement
    actual_iterations = max(num_prof_runs, 50)
    print(f"Starting energy measurement for {model_name} seq_len={seq_len} ({actual_iterations} iters)...")
    power_log_file = os.path.join(power_logs_dir, f"{model_name}_power.log")
    cmd = (
        f"nvidia-smi --query-gpu=index,power.draw,memory.used,"
        f"utilization.memory,utilization.gpu "
        f"--format=csv --loop-ms=100 > {power_log_file}"
    )
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

    t_start = time.perf_counter()
    try:
        for i in range(actual_iterations):
            if i % 10 == 0:
                print(f"  iteration {i}/{actual_iterations}")
            out = model(**input_)
            torch.cuda.synchronize()
            del out
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during energy measurement: {e}")
    finally:
        t_end = time.perf_counter()
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(0.5)  # let nvidia-smi flush remaining samples

    duration_per_pass_s = (t_end - t_start) / actual_iterations

    # Delegate log parsing to power_logger
    energy_result = parse_energy_from_log(power_log_file, actual_iterations)
    energy_joules = energy_result["total_energy_joules"]
    avg_power_w   = energy_result["avg_power_watts"]

    import math as _math
    print(
        f"[Energy] {model_name} | seq_len={seq_len} | "
        f"{energy_joules:.4f} J/pass | "
        f"avg power={avg_power_w:.1f} W | "
        f"duration/pass={duration_per_pass_s*1000:.2f} ms"
    )
    for i in range(4):
        ej = energy_result.get(f"gpu{i}_energy_joules", float('nan'))
        pw = energy_result.get(f"gpu{i}_avg_power_watts", float('nan'))
        if not _math.isnan(ej):
            print(f"  GPU{i}: {pw:.1f} W  {ej:.4f} J/pass")

    # Append structured result to CSV
    energy_csv_path = os.path.join(energy_logs_dir, "energy_data.csv")
    file_exists = os.path.isfile(energy_csv_path)
    with open(energy_csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'model_name', 'seq_len', 'energy_joules', 'duration_per_pass_s',
            'avg_power_watts', 'num_iterations', 'device', 'timestamp',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model_name':          model_name,
            'seq_len':             seq_len,
            'energy_joules':       energy_joules,
            'duration_per_pass_s': duration_per_pass_s,
            'avg_power_watts':     avg_power_w,
            'num_iterations':      actual_iterations,
            'device':              device,
            'timestamp':           time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    print(f"Energy data appended to {energy_csv_path}")

    gc.collect()
    torch.cuda.empty_cache()

# @torch.no_grad()
# def profile_model_mamba_energy(model_name, 
#                     model, 
#                     input_, 
#                     custom_ops_list, 
#                     num_prof_runs, 
#                     device, 
#                     dynamo = False, 
#                     out_dir = "./non-gemm-out/", 
#                     export = True):
    
#     fn = lambda: model(
#         input_ids=input_['input_ids'],
#     )
#     cmd = "nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.memory,utilization.gpu --format=csv --loop-ms=100 > power.log"
#     process = subprocess.Popen (cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os. setsid)
#     for i in range (1000): 
#         out = fn()
#         torch.cuda.synchronize()

#     os.killpg(os.getpgid(process.pid), signal.SIGTERM)

@torch.no_grad()
def profile_model_mamba_energy(model_name,
                    model,
                    input_,
                    custom_ops_list,
                    num_prof_runs,
                    device,
                    dynamo = False,
                    out_dir = "./non-gemm-out/",
                    export = True):
    """Measure prefill-phase energy (Joules) for a Mamba SSM model.

    Mirrors ``profile_model_energy`` but uses ``model(input_ids=...)``
    instead of ``model(**input_)`` to match the Mamba interface.  Log
    parsing is delegated to ``profiling.power_logger.parse_energy_from_log``.
    Appends one row to ``energy_logs/energy_data.csv``.
    """
    seq_len = int(input_['input_ids'].shape[1]) if 'input_ids' in input_ else 0
    fn = lambda: model(input_ids=input_['input_ids'])

    power_logs_dir  = "power_logs"
    energy_logs_dir = "energy_logs"
    os.makedirs(power_logs_dir,  exist_ok=True)
    os.makedirs(energy_logs_dir, exist_ok=True)

    # Warmup (not recorded)
    print("Starting warmup phase...")
    for _ in range(5):
        out = fn()
        torch.cuda.synchronize()
        del out
        gc.collect()
        torch.cuda.empty_cache()

    # Measurement
    actual_iterations = max(num_prof_runs, 50)
    print(f"Starting energy measurement for {model_name} seq_len={seq_len} ({actual_iterations} iters)...")
    power_log_file = os.path.join(power_logs_dir, f"{model_name}_power.log")
    cmd = (
        f"nvidia-smi --query-gpu=index,power.draw,memory.used,"
        f"utilization.memory,utilization.gpu "
        f"--format=csv --loop-ms=100 > {power_log_file}"
    )
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

    t_start = time.perf_counter()
    try:
        for i in range(actual_iterations):
            if i % 10 == 0:
                print(f"  iteration {i}/{actual_iterations}")
            out = fn()
            torch.cuda.synchronize()
            del out
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during energy measurement: {e}")
    finally:
        t_end = time.perf_counter()
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(0.5)  # let nvidia-smi flush remaining samples

    duration_per_pass_s = (t_end - t_start) / actual_iterations

    # Delegate log parsing to power_logger
    energy_result = parse_energy_from_log(power_log_file, actual_iterations)
    energy_joules = energy_result["total_energy_joules"]
    avg_power_w   = energy_result["avg_power_watts"]

    import math as _math
    print(
        f"[Energy] {model_name} | seq_len={seq_len} | "
        f"{energy_joules:.4f} J/pass | "
        f"avg power={avg_power_w:.1f} W | "
        f"duration/pass={duration_per_pass_s*1000:.2f} ms"
    )
    for i in range(4):
        ej = energy_result.get(f"gpu{i}_energy_joules", float('nan'))
        pw = energy_result.get(f"gpu{i}_avg_power_watts", float('nan'))
        if not _math.isnan(ej):
            print(f"  GPU{i}: {pw:.1f} W  {ej:.4f} J/pass")

    # Append structured result to CSV
    energy_csv_path = os.path.join(energy_logs_dir, "energy_data.csv")
    file_exists = os.path.isfile(energy_csv_path)
    with open(energy_csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'model_name', 'seq_len', 'energy_joules', 'duration_per_pass_s',
            'avg_power_watts', 'num_iterations', 'device', 'timestamp',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model_name':          model_name,
            'seq_len':             seq_len,
            'energy_joules':       energy_joules,
            'duration_per_pass_s': duration_per_pass_s,
            'avg_power_watts':     avg_power_w,
            'num_iterations':      actual_iterations,
            'device':              device,
            'timestamp':           time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    print(f"Energy data appended to {energy_csv_path}")

    gc.collect()
    torch.cuda.empty_cache()


def profile_model_dynamo_generate (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    max_num_tokens = 128,
                    dynamo = False, 
                    out_dir = "./non-gemm-out/", 
                    export = True): 
    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    from torch._inductor import config
    config.cpp.enable_kernel_profile = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"]="1"
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"]="1"
    skip_first, wait, warmup, active = 1, 2, 2, 10
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = False
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    
    #assert False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    num_layers = model.config.num_hidden_layers  # Number of transformer layers
    num_heads = model.config.num_attention_heads  # Number of attention heads
    head_dim = model.config.hidden_size // num_heads  # Dimension per head
    past_kv = tuple(
        (
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device),  # Random key tensor
            torch.randn(1, num_heads, max_num_tokens, head_dim).to(input_.input_ids.dtype).to(input_.input_ids.device)   # Random value tensor
        )
        for _ in range(num_layers)
    )
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            
        for n in range (skip_first + wait + warmup + 10):
            with torch.profiler.record_function(f"Inference_prof_{n}"):
                st = time.perf_counter()
                #out = model.generate(**input_, max_new_tokens = max_num_tokens, eos_token_id=None,)
                out = model(input_ids = input_.input_ids[:,:1], past_key_values = past_kv, use_cache = True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                et = time.perf_counter () - st 
                print (f"Time fot one inference = {et} s")
            prof.step()            
        del out
        gc.collect

    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

    filename = f"{out_dir}/{model_name}.csv"
    _analyze_prof_dynamo (prof, filename, custom)





def profile_generate_shape (model_name, 
                    model, 
                    input_, 
                    custom_ops_list, 
                    num_prof_runs, 
                    device, 
                    max_num_tokens = 1,
                    dynamo = False, 

                    out_dir = "./non-gemm-out/", 
                    export = True): 

    skip_first, wait, warmup, active = 1, 2, 2, 2
    schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
    device = device if torch.cuda.is_available() else 'cpu'
    mem_prof = True
    dynamo = dynamo 
    custom = True
    if (custom):
        #model = model.apply(replace_forward)
        ops = ["conv1d"]
        model = model.apply(lambda module: replace_forward(module, custom_ops_list))
        custom = False
    #assert (len(input_list) == skip_first + wait + warmup + active)
    with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes = True) as prof:
            
            for n in range (skip_first + wait + warmup):
                out = model.generate (**input_, max_new_tokens = 1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.step()
            del out
            
            with torch.profiler.record_function(f"Inference_prof"):
                st = time.perf_counter()
                out = model.generate(**input_, max_new_tokens = max_num_tokens, eos_token_id=None,)
                #out = model(**input_)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                et = time.perf_counter () - st 
                print (f"Time fot one inference = {et} s")
            
            del out
            gc.collect
    
    
    
    out_dir = f'{out_dir}/{model_name}'    
    os.system(f"mkdir -p {out_dir}") 
    
    
    if (export): 
        prof.export_chrome_trace(f"{out_dir}/{model_name}_shape.json")
    

    filename = f"{out_dir}/{model_name}_shape.csv"
    _analyze_prof_shape (prof, filename, custom)


########## Not Stable and Not Used ############
# @torch.no_grad()
# def profile_hf_lm_model(model_name, model_config, path_weight, quantized, device, out_dir, export): 
    
#     skip_first, wait, warmup, active = 1, 2, 2, 2
#     schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
#     device = device if torch.cuda.is_available() else 'cpu'
#     mem_prof = False
#     dynamo = False 
#     dataset = "hello"
    
#     if (quantized):
#         pass
#     else:
#         model_config = model_config if path_weight == None else path_weight
#         model = transformers.AutoModelForCausalLM.from_pretrained(model_config).eval().to(device).to(torch.float16)
#         tokenizer = transformers.AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")
#         model = transformers.AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-AWQ").eval().to("cuda:0")
#         if (dynamo): 
#             model = torch.compile(model, backend="inductor")
#         custom = True
#         if (custom):
#             #model = model.apply(replace_forward)
            
#             model = model.apply(lambda module: replace_forward(module, ops))
#             custom = False
#         #tokenizer = transformers.AutoTokenizer.from_pretrained(model_config)
#         input_ = tokenizer (dataset, return_tensors ="pt").to(device)
        
#         with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof) as prof:
#             for _ in range (skip_first + wait + warmup + active):
#                 with torch.profiler.record_function(f"Inference_prof"):
#                     st = time.perf_counter()
#                     #out = model.generate(**input_, max_length = 8)
#                     out = model(**input_)
#                     torch.cuda.synchronize()
#                     et = time.perf_counter () - st 
#                     print (f"Time fot one inference = {et} s")
#                 prof.step()
        
#         if (export): 
#             prof.export_chrome_trace(f"{model_name}.json")
        
#         os.system(f"mkdir -p {out_dir}")

#         filename = f"{out_dir}/{model_name}.csv"
#         _analyze_prof (prof, filename, custom)

# @torch.no_grad()
# def profile_model_generate (model_name, 
#                     model, 
#                     input_, 
#                     max_tokens_,
#                     custom_ops_list, 
#                     num_prof_runs, 
#                     device, 
#                     dynamo = False, 
#                     out_dir = "./non-gemm-out/", 
#                     export = True): 

#     skip_first, wait, warmup, active = 1,2, 2, num_prof_runs
#     schedule = torch.profiler.schedule(skip_first =skip_first, wait = wait, warmup = warmup, active = active)
#     device = device if torch.cuda.is_available() else 'cpu'
#     mem_prof = False
#     dynamo = dynamo 
#     custom = True
#     if (custom):
#         #model = model.apply(replace_forward)
#         ops = ["conv1d"]
#         model.model = model.model.apply(lambda module: replace_forward(module, custom_ops_list))
#         custom = False
#     #assert (len(input_list) == skip_first + wait + warmup + active)
#     inputs_warmup = {
#         'input_ids':torch.randint(1,5000, (1,2)),
#         'attention_mask':torch.ones(1,2)
#     }
#     max_tokens_warmup = 4
#     max_tokens = max_tokens_warmup
#     inputs = inputs_warmup
#     record_function = "Warmup"
#     with torch.profiler.profile(schedule = schedule, profile_memory=mem_prof) as prof:
#             for n in range (skip_first + wait + warmup + active):
#                 print (f"Profiling Iteration {n}")
#                 with torch.profiler.record_function(record_function):
#                     st = time.perf_counter()
#                     out = model.generate(**inputs, max_new_tokens = max_tokens)
                   
#                     if torch.cuda.is_available():
#                         torch.cuda.synchronize()
#                     et = time.perf_counter () - st 
#                     print (f"Time fot one inference = {et} s")
#                 prof.step()
#                 del out
#                 inputs, max_tokens, record_function = (input_, max_tokens_, "Inference_prof") if (n >=  skip_first + wait + warmup - 1) else (inputs, max_tokens_warmup, record_function)
#                 gc.collect
    
    
#     out_dir = f'{out_dir}/{model_name}'    
#     os.system(f"mkdir -p {out_dir}") 
    
    
#     if (export): 
#         prof.export_chrome_trace(f"{out_dir}/{model_name}.json")
    

#     filename = f"{out_dir}/{model_name}.csv"
#     _analyze_prof (prof, filename, custom)
############################
def main(): 
    profile_hf_lm_model("raed", "gpt2-large", None, False, "cuda", "./non-gemm-out", True) 
    
    #model = torchvision.models.vit_b_16().to(torch.float16)
    custom = False ## Determines Granurality of Profile, if you need coarser grain information at the torch.nn.module set to True
    #profile_model (model, "vit-b16", 'cuda', torch.randn(1,3,224,224).to(torch.float16), custom,True, './non-gemm-out')
    #profile_softmax ("softmax", "softmax", 'cuda', torch.randn(1,64).to(torch.float16), custom,True, './out')



if __name__ =="__main__": 
    main()
    #generate_report("out/vit-b16.csv")
    print ("Done")