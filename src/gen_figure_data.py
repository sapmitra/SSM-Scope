'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-02 03:07:18
 # @ Description:
 '''

import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plot

plot.rcParams.update({'font.size': 20})
#GEMM,activation,logit_computation,nomralization,arithmetic,pooling,interpolation,embedding,memory,roi,other
color_scheme = {"GEMM":'#4C443C' , "NonGEMM":'#DEB841', "SSM_Scan":"#E48D9C", "nomralization":"#DEB841", "activation":"#769FB6", "arithmetic":"#D16666", "interpolation":"#999AC6", "memory":"#55917F",  "other":"#32373B", "pooling":"#BDBBB6", "embedding":"#83D628", "logit_computation":"#254E70", "roi":"#FAE8EB", }


color_scheme_haocheng = {"GEMM":'#7A9E9F' , "gemm":'#7A9E9F' ,"NonGEMM":'#DEB841', "nomralization":"#E97C3E", "activation":"#F1F0CC", "arithmetic":"#1D6B8B", "interpolation":"#373F51", "memory":"#373F51",  "other":"#373F51", "pooling":"#373F51", "embedding":"#373F51", "logit_computation":"#E43F6F", "roi":"#373F51", "attention":"#FAE8EB" }



gemm_ops = ["aten::mm", "aten::matmul", "aten::bmm", "aten::linear", "aten::addmm", "aten::addbmm", "aten::baddbmm", "aten::mv",    "aten::dot",
    "aten::ger", "aten::matmul.out", "aten::scaled_dot_product_attention",
    "aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::conv_tbc",
    "aten::conv_transpose1d", "aten::conv_transpose2d", "aten::conv_transpose3d",
    "aten::slow_conv3d", "aten::slow_conv_dilated2d", "aten::slow_conv_dilated3d", "aten::slow_conv_transpose2d", "aten::slow_conv_transpose3d",
    "aten::thnn_conv2d","aten::thnn_conv_depthwise2d","aten::scaled_dot_product_attention", "aten::linear",'wqlinearmmfunction',"conv1d", "aten::einsum"]

attention_ops = [i for i in gemm_ops if "attention" in i]

gemm_ops_no_attn = [i for i in gemm_ops if not (i in attention_ops)]

# summary_dir = f"./rebuttal_summary"
summary_dir = f"./iiswc_2025_plot"
if not os.path.exists(summary_dir):
    os.system(f"mkdir -p {summary_dir}")

models = [
    'swin-base',
    'swin-small',
    'swin-tiny',
    'vit-huge',
    'vit-large',
    'vit-base',
    'detr',
    'maskformer-base',
    'segformer',
    'segformer-b1',
    'segformer-b3',
    'segformer-b5',


    'llama2-awq',
    'llama2',
    'gpt2-xl',
    'gpt2-large',
    'gpt2',
    'bert',
    'maskrcnn',
    'fasterrcnn'
]

lm = [
    #'llama2-awq',
    # 'gpt2-xl',
    # 'llama2',
    
    #'gpt2-large',
    # 'gpt2',
    # 'bert',
     
    # 'bert_large', 
    # 'llama3',
    # 'mamba-130m',
    # 'mamba-790m',
    # 'mamba2-130m',
    # 'mamba-130m-hf',
    # 'mamba-1.4b',
    # 'mamba-2.8b',
    'zamba2',
    # 'hymba',
    # 'qwen25-instruct',
    # 'qwen25-1.5b-instruct',
    # 'tinyllama',
    # 'gpt-neo-125m',


]
classfication = [
    # 'swin-base',
    #'swin-small',
    #'swin-tiny',
    # 'vit-huge',
    # 'vit-large',
    # 'vit-base',
    'vit-hf-base', 
    'vit-hf-huge', 
]
detection = [
    # 'detr',
    # #'maskrcnn',
    # #'fasterrcnn',
]

segmentaion = [
    #'maskformer-base',
    # 'segformer',
]

haocheng = [
    'bert',     
    'bert_large', 
    'vit-hf-base', 
    'vit-hf-huge', 
    'llama3',
]

non_gemm = ['NonGEMM']
act = ['aten::silu', 'aten::gelu', 'aten::sigmoid', 'aten::relu', 'aten::relu_', 'newgeluactivation_prof', 'triton_poi_fused_mul_silu_8', 'aten::softplus', ]
logit_computation = ['aten::softmax',]
norm = ['aten::layer_norm', 'layernormfn', 'aten::group_norm', 'aten::batch_norm', 'llamarmsnorm_prof', "detrfrozenbatchnorm2d_prof", "mixtralrmsnorm_prof", "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0", 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7', 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9', 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_10',  'zamba2rmsnorm_prof', 'zamba2rmsnormgated_prof', 'qwen2rmsnorm_prof', 'mambarmsnorm_prof', 'hymbarmsnorm_prof']

roi = ['torchvision::roi_align', 'torchvision::nms', ]
arith = [ 'aten::rsub','aten::add', 'aten::add_', 'aten::div', 'aten::mul', 'aten::floor', 'aten::neg',  'aten::mul_', 'aten::gt', 'aten::sub','aten::ge', 'aten::lt', 'aten::le', 'aten::eq', 'aten::ne', 'aten::bitwise_not',  'aten::__and__', 'aten::is_nonzero', 'aten::any','aten::clamp', 'aten::all', 'aten::pow', 'aten::sin', 'aten::cos', 'aten::rsqrt', 'aten::sqrt', 'aten::log2', 'aten::exp', 'aten::max', 'aten::min', 'aten::cumsum', "aten::mean", "aten::div_", "aten::index_add_", 'aten::__or__', "aten::argmax", 'aten::exponential_', 'aten::sum', 'aten::bitwise_and',  'triton_red_fused_add_all_eq_masked_fill_1', 'triton_poi_fused_add_cat_clone_mul_4', 'triton_poi_fused_add_all_bitwise_not_constant_pad_nd_eq_masked_fill_mul_6',  ]

arith_lin_elmt_wise = [ 'aten::add', 'aten::add_', 'aten::div', 'aten::mul', 'aten::floor', 'aten::neg',  'aten::mul_', 'aten::gt', 'aten::sub','aten::ge', 'aten::lt', 'aten::le', 'aten::eq', 'aten::ne', 'aten::bitwise_not',  'aten::__and__', 'aten::is_nonzero', 'aten::clamp', 'aten::all', ]

arith_non_lin_elmt_wise = ['aten::pow', 'aten::sin', 'aten::cos', 'aten::rsqrt', 'aten::sqrt', 'aten::log2', 'aten::exp',]

arith_lin_red = ['aten::max', 'aten::min', 'aten::cumsum',   ]

pooling = ['aten::adaptive_avg_pool1d','aten::max_pool2d', 'aten::adaptive_avg_pool2d',  ]

interpolation = ['aten::upsample_nearest2d', 'aten::upsample_bilinear2d',  ]

embedding = ['aten::embedding',]

mem = ['aten::slice', 'aten::chunk', 'aten::view', 'aten::permute', 'aten::transpose', 'aten::t', 'aten::reshape',  'aten::flatten', 'aten::pad', 'aten::contiguous',  'aten::index', 'aten::unsqueeze', 'aten::to', 'aten::cat', 'aten::copy_', 'aten::empty', 'aten::expand', 'aten::new_empty', 'aten::new_zeros', 'aten::where',  'aten::unbind',  'aten::select', 'aten::new_full', 'aten::masked_fill', 'aten::ones', 'aten::fill_', 'aten::full', 'aten::repeat', 'aten::stack',  'aten::arange',  'aten::type_as', 'aten::_unique2', 'aten::index_put_', 'aten::zeros', 'aten::zero_',   'aten::zeros_like', 'aten::expand_as', 'aten::full_like',  'aten::detach',   'aten::detach_', 'aten::split_with_sizes', 'aten::split', 'aten::tensor_split', "aten::one_hot", "aten::scatter", "aten::new_ones", 'aten::squeeze', 'aten::clone', 'aten::masked_fill_', 'aten::ones_like', 'aten::empty_like', 'aten::resize_' , 'triton_poi_fused__to_copy_2', 'triton_poi_fused__to_copy_3', 'triton_poi_fused_clone_5',  'triton_poi_fused__to_copy_11', 'aten::_unsafe_view', 'aten::item', 'aten::alias', 'aten::concatenate', ]

other = ['aten::dropout', 'aten::lift_fresh', 'aten::meshgrid', 'aten::topk', 'aten::sort', 'aten::argsort','torchdynamo cache lookup','torch-compiled region','aten::_assert_async', 'aten::triu',]

non_gemm_ops = act + logit_computation + norm + roi + arith + pooling + interpolation + embedding + mem + other
non_gemm_ops_dict = {'activation':act, "logit_computation":logit_computation,
                     'nomralization':norm, 'arithmetic':arith, "pooling":pooling,
                     'interpolation':interpolation, 'embedding': embedding,
                     'memory':mem, 'roi':roi, 'other':other,}

gemm_ops_dict = {
    "gemm":gemm_ops_no_attn, 
    "attention":attention_ops,
}

ops_dict = {
    "gemm":gemm_ops_no_attn, 
    "attention":attention_ops,
    'activation':act, "logit_computation":logit_computation,
    'nomralization':norm, 'arithmetic':arith, "pooling":pooling,
    'interpolation':interpolation, 'embedding': embedding,
    'memory':mem, 'roi':roi, 'other':other,

}

batch_sizes = [1]#[1,2,4,8]#[1]#[1,8]#
seq_len = {
    #'llama2-awq':2048,
    # 'gpt2-xl':512,
    # 'llama2':2048,
    #'gpt2-large':256,
    # 'gpt2':64,
    # 'bert':64,#64,
    # 'bert_large':128, 
    # 'llama3':1024,
    #'mistral':2048,
    # 'mamba-130m':64,
    # 'mamba-2.8b':64,
    # 'gpt2':64,
    'zamba2':256,
    # 'qwen25-instruct':64,
    # 'tinyllama':64,
    # 'gpt-neo-125m':64,
    # 'mamba-130m-hf':64,
    }

seq_len_multi = {
    # 'mamba-130m':[256, 1024, 8192, 65536, 131072],#[8,64,512,2048,4096,8192],
    # 'mamba-790m':[1024, 8192, 65536],#[8,64,512,2048,4096,8192],
    # 'mamba2-130m':[256, 1024, 8192, 65536, 131072],#[8,64,512,2048,4096,8192],
    # 'mamba-1.4b':[256, 1024, 8192, 32768],#[8,64,512,2048,4096,8192],
    # 'mamba-2.8b':[64,8192],
    # 'zamba2':[256,1024, 8192, 32768],
    # 'qwen25-instruct':[1024,8192,32768],
    'qwen25-1.5b-instruct':[256,1024,8192,32768],
    # 'gpt-neo-125m':[64,2048],
    # 'mamba-130m-hf':[64,8192],
    # 'hymba':[64, 8192],
    # 'tinyllama':[64,8192],
    }

haocheng_seq_len = {
    #'llama2-awq':2048,
    # 'gpt2-xl':512,
    # 'llama2':2048,
    #'gpt2-large':256,
    #'gpt2':256,
    'bert':[128],#64,
    'bert_large':[128], 
    'vit-hf-base':[0], 
    'vit-hf-huge':[0], 
    'llama3':[128,512,1024], 


    }


devices = ['cpu', 'cuda']

gemm_file = "gemm.csv"
non_gemm_file = "non_gemm.csv"
ssm_scan_file = "ssm_scan.csv"

def get_directories(path: str= "./non-gemm-out"):
    entries = os.listdir(path)
    # Filter only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return directories

def extract_non_gemm (prof_dir:str = "./non-gemm-out",out_dir = None):
    unique_non_gemm_ops = []
    direct = get_directories()
    print (direct)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        tmp = df_nongemm['name'].unique().tolist()
        tmp.remove('Inference')
        tmp.remove('NonGEMM')
        for i in tmp:
            if not (i in unique_non_gemm_ops):
                print (i)
                unique_non_gemm_ops.append(i)
            if (i == "aten::einsum"):
                print (f"Move this to GEMM: {dir}")

    print (unique_non_gemm_ops)

def summarize_ops (prof_dir:str = "./non-gemm-out",out_dir = None): 
    direct = get_directories(prof_dir)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        df_summary = pd.read_csv(f"{prof_dir}/{dir}/{dir}.csv")
        df_gng = df_summary[df_summary['name'].isin(['GEMM', 'NonGEMM'])]
        df_ops = pd.DataFrame()
        for group, list_ in ops_dict.items():
            df_ = filter_dataframes(df_summary, list_)
            df_, summary_row = sum_df_append(df_, group)
            df_ops = pd.concat([df_ops, summary_row], ignore_index=True).drop(columns = ['Unnamed: 0'])
            #print (df_)
            #break
            df_.to_csv(f"{prof_dir}/{dir}/{group}.csv")
        #break
        df_ops.to_csv(f"{prof_dir}/{dir}/summary_{dir}.csv")
        #df_gng.to_csv(f"{prof_dir}/{dir}/gng_{dir}.csv")

        df_summary_transpose = df_ops[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']
        df_gng_transpose = df_gng[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']

        df_ = df_summary_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/pct_{dir}.csv')

        df_ = df_gng_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/gng_pct_{dir}.csv')

def check_new_non_gemm (unique_non_gemm): 
    new_non_gemm = []

    for op in unique_non_gemm: 
        if not (op in non_gemm_ops): 
            new_non_gemm.append(op)
    print (f"New Non-GEMM Operators:") 
    print (new_non_gemm)

# after pasting the directories (only bring those which you need plot of, rest do not keep here) under consideration for visualization , create necessary child csv under each model directory
def summarize_non_gemm(prof_dir:str = "./non-gemm-out",out_dir = None):
    direct = get_directories(prof_dir)
    for dir in direct:
        data_file = f"{prof_dir}/{dir}/{non_gemm_file}"
        if not (os.path.exists(data_file)):
            continue
        df_nongemm = pd.read_csv(data_file)
        unique_nongemm = df_nongemm['name'].unique().tolist()
        check_new_non_gemm(unique_nongemm)
        df_summary = pd.read_csv(f"{prof_dir}/{dir}/{dir}.csv")
        df_gng = df_summary[df_summary['name'].isin(['GEMM', 'NonGEMM'])]
        df_gng_ssm = df_summary[df_summary['name'].isin(['GEMM', 'NonGEMM', 'SSM_Scan'])]
        df_summary = df_summary[df_summary['name'].isin(['GEMM', 'SSM_Scan'])]
        for group, list_ in non_gemm_ops_dict.items():
            df_ = filter_dataframes(df_nongemm, list_)
            df_, summary_row = sum_df_append(df_, group)
            df_summary = pd.concat([df_summary, summary_row], ignore_index=True).drop(columns = ['Unnamed: 0'])
            #print (df_)
            #break
            df_.to_csv(f"{prof_dir}/{dir}/{group}.csv")
        #break
        df_summary.to_csv(f"{prof_dir}/{dir}/summary_{dir}.csv")
        df_gng.to_csv(f"{prof_dir}/{dir}/gng_{dir}.csv")
        df_gng_ssm.to_csv(f"{prof_dir}/{dir}/gng_ssm_{dir}.csv")

        df_summary_transpose = df_summary[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']
        # df_summary_transpose_cpu = df_summary[["name", "cpu_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']
        df_gng_transpose = df_gng[["name", "total_time (us)"]]#.set_index ('name')#[df_summary['name'] == 'total_time (us)']
        df_gng_ssm_transpose = df_gng_ssm[["name", "total_time (us)"]]

        df_ = df_summary_transpose
        # df_ = df_summary_transpose_cpu
        sum_ = df_['total_time (us)'].sum()
        # sum_ = df_['cpu_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        # df_['pct'] = (df_['cpu_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/pct_{dir}.csv')

        df_ = df_gng_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/gng_pct_{dir}.csv')

        df_ = df_gng_ssm_transpose
        sum_ = df_['total_time (us)'].sum()
        df_['pct'] = (df_['total_time (us)'] / sum_) * 100
        df_.to_csv(f'{prof_dir}/{dir}/gng_ssm_pct_{dir}.csv')



def plot_classsification(prof_directory: str = "./non-gemm-out"):
    task = "classification_t"
    for device in devices:
        df_cls = None
        for model in classfication:
            for n in batch_sizes:
                filename = f"{prof_directory}/{model}_{device}_{n}/pct_{model}_{device}_{n}.csv"
                model_ = f"{model}_{device}_{n}"
                if not os.path.exists(filename):
                    print(f"File {model}_{device}_{n}/pct_{model}_{device}_{n}.csv does not exist.")
                    print (f"Skipping...")
                    continue
                df_ = pd.read_csv(filename)
                df_ [model_] = df_['total_time (us)']
                df_t = df_.set_index('name')
                df_t = df_t [[model_]]
                #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                print (df_t.T.reset_index().columns)
                df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        df_cls.set_index('index').to_csv(f"{summary_dir}/{task}_{device}.csv")
    #plot_figure_op_breakdown(summary_dir, task,)

def plot_detection(prof_directory: str = "./non-gemm-out"):
    task = "detection_detr"
    for device in devices:
        df_cls = None
        for model in detection:
            for n in batch_sizes:
                filename = f"{prof_directory}/{model}_{device}_{n}/pct_{model}_{device}_{n}.csv"
                model_ = f"{model}_{device}_{n}"
                if not os.path.exists(filename):
                    print(f"File {model}_{device}_{n}/pct_{model}_{device}_{n}.csv does not exist.")
                    print (f"Skipping...")
                    continue
                df_ = pd.read_csv(filename)
                df_ [model_] = df_['total_time (us)']
                df_t = df_.set_index('name')
                df_t = df_t [[model_]]
                #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                print (df_t.T.reset_index().columns)
                df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        df_cls.set_index('index').to_csv(f"{summary_dir}/{task}_{device}.csv")
    #plot_figure_op_breakdown(summary_dir, task,)
def plot_segmentation(prof_directory: str = "./non-gemm-out"):
    task = "segmentation_seg"
    for device in devices:
        df_cls = None
        for model in segmentaion:
            for n in batch_sizes:
                filename = f"{prof_directory}/{model}_{device}_{n}/pct_{model}_{device}_{n}.csv"
                model_ = f"{model}_{device}_{n}"
                if not os.path.exists(filename):
                    print(f"File {model}_{device}_{n}/pct_{model}_{device}_{n}.csv does not exist.")
                    print (f"Skipping...")
                    continue
                df_ = pd.read_csv(filename)
                df_ [model_] = df_['total_time (us)']
                df_t = df_.set_index('name')
                df_t = df_t [[model_]]
                #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                print (df_t.T.reset_index().columns)
                df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        df_cls.set_index('index').to_csv(f"{summary_dir}/{task}_{device}.csv")
    #plot_figure_op_breakdown(summary_dir, task,)


def plot_haocheng(prof_directory: str = "./haocheng"):
    task = "haocheng"
    llama = [128, 512, 1024]
    bert = [128]
    devices = ["cuda"]
    for device in devices:
        df_cls = None
        for model in haocheng:
            s_ = haocheng_seq_len[model]
            for s in s_: 
                for n in batch_sizes:
                    if "vit" in model:
                        filename = f"{prof_directory}/{model}_{device}_{n}/pct_{model}_{device}_{n}.csv"
                        model_ = f"{model}_{device}_{n}"
                        
                    else: 
                        filename = f"{prof_directory}/{model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv"
                        model_ = f"{model}_{device}_{n}_{s}"
                    if not os.path.exists(filename):
                        print(f"File {filename}.csv does not exist.")
                        print (f"Skipping...")
                        continue
                    df_ = pd.read_csv(filename)
                    df_ [model_] = df_['pct']#df_['total_time (us)']
                    df_t = df_.set_index('name')
                    df_t = df_t [[model_]]
                    #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                    print (df_t.T.reset_index().columns)
                    df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
            df_cls.set_index('index').to_csv(f"haocheng_summary/{task}_{device}.csv")
    #plot_figure_op_breakdown(summary_dir, task,)
def plot_haocheng_non_gemm(prof_directory: str = "./haocheng"):
    task = "haocheng"
    llama = [128, 512, 1024]
    bert = [128]
    #devices = ["cuda"]
    for device in devices:
        df_cls = None
        for model in haocheng:
            s_ = haocheng_seq_len[model]
            for s in s_: 
                for n in batch_sizes:
                    if "vit" in model:
                        filename = f"{prof_directory}/{model}_{device}_{n}/pct_{model}_{device}_{n}.csv"
                        model_ = f"{model}_{device}_{n}"
                        
                    else: 
                        filename = f"{prof_directory}/{model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv"
                        model_ = f"{model}_{device}_{n}_{s}"
                    if not os.path.exists(filename):
                        print(f"File {filename}.csv does not exist.")
                        print (f"Skipping...")
                        continue
                    df_ = pd.read_csv(filename)
                    df_ [model_] = df_['pct']#df_['total_time (us)']
                    df_t = df_.set_index('name')
                    df_t = df_t [[model_]]
                    #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                    print (df_t.T.reset_index().columns)
                    df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
            df_cls.set_index('index').to_csv(f"haocheng_summary/{task}_{device}.csv")
# bring perentage breakdown from models under consideration according to the task and seq length defined
def plot_lm(prof_directory: str = "./non-gemm-out"):
    task = "lm"
    for device in devices:
        df_cls = None
        for model in lm:
            s = seq_len[model]
            for n in batch_sizes:
                filename = f"{prof_directory}/{model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv"
                model_ = f"{model}_{device}_{n}_{s}"
                if not os.path.exists(filename):
                    print(f"File {model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv does not exist.")
                    print (f"Skipping...")
                    continue
                df_ = pd.read_csv(filename)
                df_ [model_] = df_['pct']
                df_t = df_.set_index('name')
                df_t = df_t [[model_]]
                #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                print (df_t.T.reset_index().columns)
                df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        if (df_cls is not None):
            df_cls.set_index('index').to_csv(f"{summary_dir}/{task}_{device}.csv")
    #plot_figure_op_breakdown(summary_dir, task,)

# bring perentage breakdown from models under consideration according to all seq lengths considered
def plot_lm_seq(prof_directory: str = "./non-gemm-out"):
    task = "lm"
    for device in devices:
        df_cls = None
        for model in lm:
            for n in batch_sizes:
                for s in seq_len_multi[model]:
                    filename = f"{prof_directory}/{model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv"
                    model_ = f"{model}_{device}_{n}_{s}"
                    if not os.path.exists(filename):
                        print(f"File {model}_{device}_{n}_{s}/pct_{model}_{device}_{n}_{s}.csv does not exist.")
                        print (f"Skipping...")
                        continue
                    df_ = pd.read_csv(filename)
                    df_ [model_] = df_['pct']
                    df_t = df_.set_index('name')
                    df_t = df_t [[model_]]
                    #df_t["name"] = df_t["name"].replace("pct", "mem_ops")
                    print (df_t.T.reset_index().columns)
                    df_cls = df_t.T.reset_index() if df_cls is None else pd.concat([df_cls, df_t.T.reset_index()])
        if (df_cls is not None):
            df_cls.set_index('index').to_csv(f"{summary_dir}/{task}_{device}.csv")
    #plot_figure_op_breakdown(summary_dir, task,)

def plot_haocheng_figure_op_breakdown(summary_directory: str="./summary", task: str ="classification",color_scheme_ = color_scheme ,op_order: list = []):

    import numpy as np

    cuda_file = f"{summary_directory}/{task}_cuda.csv"
    cpu_file = f"{summary_directory}/{task}_cpu.csv"
    cuda_df = pd.read_csv(cuda_file)
    cpu_df = pd.read_csv(cpu_file)

    # cuda_order = sort_df_cols(cuda_df)
    # cpu_order = sort_df_cols(cpu_df)

    priority_order = ["gemm","attention", "nomralization", "activation", "arithmetic"]
    remaining_cols = [col for col in cuda_df.columns if col not in priority_order]

    # Randomize the order of remaining columns
    np.random.shuffle(remaining_cols)

    # Create the new column order
    new_column_order = priority_order + remaining_cols

    cuda_df = cuda_df[new_column_order]
    cpu_df = cpu_df[new_column_order]


    #plot cuda
    plt_cuda = cuda_df.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = color_scheme_)#figsize = (6,4)
    plt_cuda.tick_params(labelbottom=False)
    plt_cuda.tick_params(labelleft=False)
    plot.savefig(f"{summary_directory}/{task}_cuda.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    #plt cpu
    plt_cpu = cpu_df.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = color_scheme_)#figsize = (6,4)
    plt_cpu.tick_params(labelbottom=False)
    plt_cpu.tick_params(labelleft=False)
    plot.savefig(f"{summary_directory}/{task}_cpu.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()

# creates the plot from the summary csv file like lm_cuda.csv
def plot_figure_op_breakdown(summary_directory: str="./summary", task: str ="classification",color_scheme_ = color_scheme ,op_order: list = []):

    cuda_file = f"{summary_directory}/{task}_cuda.csv"
    cpu_file = f"{summary_directory}/{task}_cpu.csv"
    cuda_df = pd.read_csv(cuda_file)
    # cpu_df = pd.read_csv(cpu_file)

    # Read the model name from the index of df to a list
    model_names = cuda_df['index'].tolist()
    print(model_names)

    cuda_order = sort_df_cols(cuda_df)
    # cpu_order = sort_df_cols(cpu_df)

    cuda_df = cuda_df[cuda_order]
    # cpu_df = cpu_df[cpu_order]


    #plot cuda
    plt_cuda = cuda_df.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = color_scheme_)#figsize = (6,4)
    plt_cuda.tick_params(labelbottom=True)
    plt_cuda.tick_params(labelleft=True)
    plt_cuda.set_xticklabels(model_names, rotation=0, fontsize=5, ha='center')
    plt_cuda.tick_params(axis='y', labelsize=8)
    plt_cuda.set_ylabel('Runtime Breakdown (%)', fontsize=12)
    plt_cuda.legend(loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=8)
    plot.savefig(f"{summary_directory}/{task}_cuda.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    #plot cpu
    # plt_cpu = cpu_df.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = color_scheme)#figsize = (6,4)
    # plt_cpu.tick_params(labelbottom=False)
    # plt_cpu.tick_params(labelleft=False)
    # plot.savefig(f"{summary_directory}/{task}_cpu.png", format="png", bbox_inches="tight", dpi=300)
    # plot.close()

def plot_gng(prof_dir: str = "./non-gemm-out", model_name: str ='bert', batch_size: int = 1, seq_len = None ): 
    seq_len_ = f"_{seq_len}" if (seq_len is not None) else ""
    filename_cpu = f"{prof_dir}/{model_name}_cpu_{batch_size}{seq_len_}/gng_pct_{model_name}_cpu_{batch_size}{seq_len_}.csv"
    filename_cuda = f"{prof_dir}/{model_name}_cuda_{batch_size}{seq_len_}/gng_pct_{model_name}_cuda_{batch_size}{seq_len_}.csv"

    if not (os.path.exists(filename_cpu) and os.path.exists(filename_cuda)): 
        print (f"We need to get CPU and/or GPU data for {model_name}_{batch_size}{seq_len_}")
        print ("Not Generating Plots")
        return
    df_cpu = pd.read_csv(filename_cpu).set_index('name')
    df_cuda = pd.read_csv(filename_cuda).set_index('name')
    df_cpu = df_cpu [['pct']].T
    df_cuda = df_cuda [['pct']].T
    
    df_ = pd.concat([df_cpu, df_cuda])
    print (df_)
    dir_ = f"gng/{model_name}_cpu_{batch_size}{seq_len_}/"
    os.system (f"mkdir -p {dir_}")
    plt_ = df_.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.5, color = color_scheme)
    plt_.tick_params(labelbottom=False)
    plt_.tick_params(labelleft=False)
    plot.savefig(f"{dir_}/gng_pct_{model_name}_{batch_size}{seq_len_}.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    print (df_cpu)

def plot_all_gng (prof_dir: str = "./non-gemm-out"): 
    for model in classfication: 
        for n in batch_sizes: 
            plot_gng(prof_dir, model, n)
    for model in segmentaion: 
        for n in batch_sizes: 
            plot_gng(prof_dir, model, n)
    for model in detection: 
        for n in batch_sizes: 
            plot_gng(prof_dir, model, n)
    sequence_lens = [16,32,64,128,256,512,1024,2048,4096,8192]
    for model in lm: 
        for n in batch_sizes: 
            for s in sequence_lens: 
                plot_gng(prof_dir, model, n, s)

    return 


def plot_gng_batch(prof_dir: str = "./non-gemm-out", model_name: str ='bert', batch_size: int = 1, seq_len = None ):
    seq_len_ = f"_{seq_len}" if (seq_len is not None) else ""
    df_ = pd.DataFrame()
    for n in batch_sizes: 
        filename_b1 = f"{prof_dir}/{model_name}_cuda_{n}{seq_len_}/gng_pct_{model_name}_cuda_{n}{seq_len_}.csv"
    
    # filename_b2 = f"{prof_dir}/{model_name}_cuda_{2}{seq_len_}/gng_pct_{model_name}_cuda_{2}{seq_len_}.csv"
    # filename_b4 = f"{prof_dir}/{model_name}_cuda_{4}{seq_len_}/gng_pct_{model_name}_cpu_{4}{seq_len_}.csv"
    # filename_b8 = f"{prof_dir}/{model_name}_cuda_{8}{seq_len_}/gng_pct_{model_name}_cuda_{8}{seq_len_}.csv"


        if not (os.path.exists(filename_b1)): 
            print (f"We need to get CPU and/or GPU data for {model_name}_{n}{seq_len_}")
            print ("Not Generating Plots")
            return
        df_cuda = pd.read_csv(filename_b1).set_index('name')
        df_cuda = df_cuda [['pct']].T
        
        df_ = pd.concat([df_, df_cuda])
        #print (df_)
    
    plt_ = df_.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.8, color = color_scheme)
    plt_.tick_params(labelbottom=False)
    plt_.tick_params(labelleft=False)
    out_dir = f"gng-pt-batch/{model_name}_cuda_{batch_size}{seq_len_}/"
    os.system(f"mkdir -p {out_dir}")
    plot.savefig(f"{out_dir}/gng_pct_{model_name}_{batch_size}{seq_len_}.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    df_.to_csv(f"{out_dir}/gng_pct_{model_name}_{batch_size}{seq_len_}.csv")
    print (df_cuda)  

def plot_all_gng_batch(prof_dir: str = "./non-gemm-out"): 
    models = ['swin-base', 'detr', 'swin-tiny', 'segformer', 'segformer-b1', 'segformer-b3']
    for model in models: 
        plot_gng_batch(prof_dir, model)



def plot_gng_seq(prof_dir: str = "./non-gemm-out", model_name: str ='bert', batch_size: int = 1, seq_lens = None ):
    
    #seq_len_ = f"_{seq_len}" if (seq_len is not None) else ""
    df_ = pd.DataFrame()
    for seq_len_ in seq_lens: 
        filename_b1 = f"{prof_dir}/{model_name}_cuda_1_{seq_len_}/gng_pct_{model_name}_cuda_1_{seq_len_}.csv"
    
    # filename_b2 = f"{prof_dir}/{model_name}_cuda_{2}{seq_len_}/gng_pct_{model_name}_cuda_{2}{seq_len_}.csv"
    # filename_b4 = f"{prof_dir}/{model_name}_cuda_{4}{seq_len_}/gng_pct_{model_name}_cpu_{4}{seq_len_}.csv"
    # filename_b8 = f"{prof_dir}/{model_name}_cuda_{8}{seq_len_}/gng_pct_{model_name}_cuda_{8}{seq_len_}.csv"


        if not (os.path.exists(filename_b1)): 
            print (f"We need to get CPU and/or GPU data for {model_name}_1_{seq_len_}")
            print ("Not Generating Plots")
            return
        df_cuda = pd.read_csv(filename_b1).set_index('name')
        df_cuda = df_cuda [['pct']].T
        
        df_ = pd.concat([df_, df_cuda])
        #print (df_)
    
    plt_ = df_.plot.bar(stacked=True,legend = False, figsize = (4,6), width = 0.8, color = color_scheme)
    plt_.tick_params(labelbottom=False)
    plt_.tick_params(labelleft=False)
    out_dir = f"gng-pt-batch/{model_name}_cuda_1_{seq_len_}/"
    os.system(f"mkdir -p {out_dir}")
    plot.savefig(f"{out_dir}/gng_pct_{model_name}_1_{seq_len_}.png", format="png", bbox_inches="tight", dpi=300)
    plot.close()
    df_.to_csv(f"{out_dir}/gng_pct_{model_name}_1_{seq_len_}.csv")
    print (df_cuda)

def plot_all_gng_seq(prof_dir: str = "./non-gemm-out"): 
    models = {'gpt2-xl':[512, 1024, 2048, 4096], 'llama2':[1024, 2048, 4096, 8192]}
    for model, seq_lens in models.items(): 
        plot_gng_seq(prof_dir, model,1, seq_lens)


## utils##
def filter_dataframes(df, list):
    # Filter DataFrame rows where "name" is in the current list
    df_ = df[df['name'].isin(list)]
    return df_

def sum_df_append (filtered_df, name):
    summed_row = filtered_df.drop(columns=["name"]).sum()
    # Add a new row with the sum and a custom 'name' value
    summed_row["name"] = name
    df = pd.concat([filtered_df, pd.DataFrame([summed_row])], ignore_index=True)
    # summary_row = filtered_df.drop(columns=["name"], errors='ignore').sum(numeric_only=True)
    # summary_row["name"] = name  # Add the list's name
    # filtered_df = pd.concat([filtered_df, summary_row.to_frame()])
    return df, summed_row.to_frame().T

def get_percentages(df):
    df_ = df.drop(columns=["Unnamed: 0"])

def sort_df_cols(df): 
    sorted_list = []
    columns = list(df.columns)
    columns.remove("GEMM")
    columns.remove("SSM_Scan")
    columns.remove("index")

    df_ = df[columns]
    df_ = df_.loc[0, :].to_dict()
    sorted_keys = dict(sorted(df_.items(), key=lambda k: k[1], reverse = True))
    sorted_columns = list(sorted_keys.keys())
    new_column_order = ["SSM_Scan"] + ["GEMM"] +  sorted_columns
    
    return new_column_order 

def sort_df_cols_haocheng(df): 
    sorted_list = []
    columns = list(df.columns)
    columns.remove("gemm")
    columns.remove("index")

    df_ = df[columns]
    df_ = df_.loc[0, :].to_dict()
    sorted_keys = dict(sorted(df_.items(), key=lambda k: k[1], reverse = True))
    sorted_columns = list(sorted_keys.keys())
    new_column_order = ["GEMM"] + sorted_columns
    
    return new_column_order 
##########

# summarize_non_gemm(prof_dir='plot_model_nongemm')
# summarize_non_gemm(prof_dir = 'rebuttal-dynamo')
# summarize_ops(prof_dir="haocheng")
# plot_haocheng()
# plot_haocheng_figure_op_breakdown(summary_directory = "haocheng_summary", task='haocheng', color_scheme_= color_scheme_haocheng)
# # plot_classsification()
# # plot_detection()
# # plot_segmentation()
# plot_lm()
# plot_lm(prof_directory = 'plot_model_nongemm')
# plot_lm_seq(prof_directory = 'plot_model_nongemm')

# # plot_figure_op_breakdown(task ="classification", )
# # plot_figure_op_breakdown(task = "segmentation", )
# # plot_figure_op_breakdown(task = "detection", )
# # plot_figure_op_breakdown(task = "lm")
plot_figure_op_breakdown(summary_directory="./iiswc_2025_plot",task = "lm")

# # plot_all_gng()
# plot_all_gng_batch()
# # plot_all_gng_seq()
