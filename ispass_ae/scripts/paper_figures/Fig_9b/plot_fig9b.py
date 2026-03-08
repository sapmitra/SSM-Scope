'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-08
 # @ Description: Reproduce Figure 9b from the paper.  See the README for usage instructions.
 '''

"""
Reproduce Figure 9b — Cross-device GPU kernel-time breakdown for all model
families (Transformer, SSM, Hybrid) on Desktop GPU vs. NVIDIA Jetson Orin Nano.

For each model the chart shows a **side-by-side stacked bar pair**:
  left bar  = Desktop GPU
  right bar = NVIDIA Jetson Orin Nano

Models are grouped into three architecture families separated by dashed
vertical dividers.

Reads per-operator ``pct_*.csv`` files produced by ``collect_fig9b_data.py``
from the desktop (``--desktop_dir``) and Jetson (``--jetson_dir``) profile
directories.  If ``summarize_non_gemm`` has not been run yet, the script
runs it automatically before loading the data.

Usage (from repo root, any venv with matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_9b/plot_fig9b.py

    # With explicit directories:
    python ispass_ae/scripts/paper_figures/Fig_9b/plot_fig9b.py \\
        --desktop_dir src/profile_logs \\
        --jetson_dir  src/profile_logs_jetson \\
        --seq_len 1024 \\
        --out_dir ispass_ae/scripts/paper_figures/Fig_9b

Output files
------------
``fig9b_device_comparison_seq<N>.png``
    Publication-quality side-by-side comparison (300 DPI).
``fig9b_device_comparison_seq<N>_annotated.png``
    Same with full axis labels, legend, and category headers (150 DPI).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import os
import warnings
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ===========================================================================
# Operator categorisation — kept in sync with plotting_ops_cross_device.ipynb
# ===========================================================================

color_scheme = {
    "GEMM":             '#4C443C',
    "NonGEMM":          '#DEB841',
    "SSM_Scan":         "#E48D9C",
    "nomralization":    "#DEB841",
    "activation":       "#769FB6",
    "arithmetic":       "#D16666",
    "interpolation":    "#999AC6",
    "memory":           "#55917F",
    "other":            "#32373B",
    "pooling":          "#BDBBB6",
    "embedding":        "#83D628",
    "logit_computation":"#254E70",
    "roi":              "#FAE8EB",
}

gemm_ops = [
    "aten::mm", "aten::matmul", "aten::bmm", "aten::linear", "aten::addmm",
    "aten::addbmm", "aten::baddbmm", "aten::mv", "aten::dot", "aten::ger",
    "aten::matmul.out", "aten::scaled_dot_product_attention",
    "aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::conv_tbc",
    "aten::conv_transpose1d", "aten::conv_transpose2d", "aten::conv_transpose3d",
    "aten::slow_conv3d", "aten::slow_conv_dilated2d", "aten::slow_conv_dilated3d",
    "aten::slow_conv_transpose2d", "aten::slow_conv_transpose3d",
    "aten::thnn_conv2d", "aten::thnn_conv_depthwise2d",
    "aten::scaled_dot_product_attention", "aten::linear",
    "wqlinearmmfunction", "conv1d", "aten::einsum",
]

attention_ops = [op for op in gemm_ops if "attention" in op]
gemm_ops_no_attn = [op for op in gemm_ops if op not in attention_ops]

act = [
    'aten::silu', 'aten::gelu', 'aten::sigmoid', 'aten::relu', 'aten::relu_',
    'newgeluactivation_prof', 'triton_poi_fused_mul_silu_8', 'aten::softplus',
]
logit = ['aten::softmax']
norm = [
    'aten::layer_norm', 'layernormfn', 'aten::group_norm', 'aten::batch_norm',
    'llamarmsnorm_prof', 'detrfrozenbatchnorm2d_prof', 'mixtralrmsnorm_prof',
    'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0',
    'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7',
    'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9',
    'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_10',
    'zamba2rmsnorm_prof', 'zamba2rmsnormgated_prof', 'qwen2rmsnorm_prof',
    'mambarmsnorm_prof', 'hymbarmsnorm_prof',
]
roi = ['torchvision::roi_align', 'torchvision::nms']
arith = [
    'aten::rsub', 'aten::add', 'aten::add_', 'aten::div', 'aten::mul',
    'aten::floor', 'aten::neg', 'aten::mul_', 'aten::gt', 'aten::sub',
    'aten::ge', 'aten::lt', 'aten::le', 'aten::eq', 'aten::ne',
    'aten::bitwise_not', 'aten::__and__', 'aten::is_nonzero', 'aten::any',
    'aten::clamp', 'aten::all', 'aten::pow', 'aten::sin', 'aten::cos',
    'aten::rsqrt', 'aten::sqrt', 'aten::log2', 'aten::exp', 'aten::max',
    'aten::min', 'aten::cumsum', 'aten::mean', 'aten::div_',
    'aten::index_add_', 'aten::__or__', 'aten::argmax', 'aten::exponential_',
    'aten::sum', 'aten::bitwise_and',
    'triton_red_fused_add_all_eq_masked_fill_1',
    'triton_poi_fused_add_cat_clone_mul_4',
    'triton_poi_fused_add_all_bitwise_not_constant_pad_nd_eq_masked_fill_mul_6',
]
pooling = [
    'aten::adaptive_avg_pool1d', 'aten::max_pool2d', 'aten::adaptive_avg_pool2d',
]
interp = ['aten::upsample_nearest2d', 'aten::upsample_bilinear2d']
embed = ['aten::embedding']
mem = [
    'aten::slice', 'aten::chunk', 'aten::view', 'aten::permute', 'aten::transpose',
    'aten::t', 'aten::reshape', 'aten::flatten', 'aten::pad', 'aten::contiguous',
    'aten::index', 'aten::unsqueeze', 'aten::to', 'aten::cat', 'aten::copy_',
    'aten::empty', 'aten::expand', 'aten::new_empty', 'aten::new_zeros', 'aten::where',
    'aten::unbind', 'aten::select', 'aten::new_full', 'aten::masked_fill', 'aten::ones',
    'aten::fill_', 'aten::full', 'aten::repeat', 'aten::stack', 'aten::arange',
    'aten::type_as', 'aten::_unique2', 'aten::index_put_', 'aten::zeros', 'aten::zero_',
    'aten::zeros_like', 'aten::expand_as', 'aten::full_like', 'aten::detach',
    'aten::detach_', 'aten::split_with_sizes', 'aten::split', 'aten::tensor_split',
    'aten::one_hot', 'aten::scatter', 'aten::new_ones', 'aten::squeeze', 'aten::clone',
    'aten::masked_fill_', 'aten::ones_like', 'aten::empty_like', 'aten::resize_',
    'triton_poi_fused__to_copy_2', 'triton_poi_fused__to_copy_3',
    'triton_poi_fused_clone_5', 'triton_poi_fused__to_copy_11',
    'aten::_unsafe_view', 'aten::item', 'aten::alias', 'aten::concatenate',
]
other = [
    'aten::dropout', 'aten::lift_fresh', 'aten::meshgrid', 'aten::topk',
    'aten::sort', 'aten::argsort', 'torchdynamo cache lookup',
    'torch-compiled region', 'aten::_assert_async', 'aten::triu',
]

non_gemm_ops = (act + logit + norm + roi + arith + pooling + interp + embed + mem + other)
non_gemm_ops_dict = {
    'activation':        act,
    'logit_computation': logit,
    'nomralization':     norm,
    'arithmetic':        arith,
    'pooling':           pooling,
    'interpolation':     interp,
    'embedding':         embed,
    'memory':            mem,
    'roi':               roi,
    'other':             other,
}

# ---------------------------------------------------------------------------
# Model groups (display names → profile directory names)
# ---------------------------------------------------------------------------
MODEL_CATEGORIES = {
    'Transformer': [
        'qwen25-instruct',
        'qwen25-1.5b-instruct',
        'llama3_2',
        'tinyllama',
        'gpt-neo-125m',
    ],
    'SSM': [
        'mamba-130m',
        'mamba2-130m',
    ],
    'Hybrid': [
        'zamba2',
        'hymba',
    ],
}

# Human-readable x-tick labels for each model key
MODEL_LABELS = {
    'qwen25-instruct':      'Qwen2.5-0.5B',
    'qwen25-1.5b-instruct': 'Qwen2.5-1.5B',
    'llama3_2':             'LLaMA-3.2-1B',
    'tinyllama':            'TinyLlama-1.1B',
    'gpt-neo-125m':         'GPT-Neo-125m',
    'mamba-130m':           'Mamba-130m',
    'mamba2-130m':          'Mamba2-130m',
    'zamba2':               'Zamba2-1.2B',
    'hymba':                'Hymba-1.5B',
}


# ===========================================================================
# Helper: summarise non-GEMM ops
# ===========================================================================

def _check_new_non_gemm(unique_names):
    new_ops = [op for op in unique_names if op not in non_gemm_ops]
    if new_ops:
        print(f"  [info] Unrecognised non-GEMM operators: {new_ops}")


def _filter_df(df, op_list):
    return df[df['name'].isin(op_list)]


def _sum_and_append(df, category_name):
    summed = df.drop(columns=['name']).sum()
    summed['name'] = category_name
    return pd.concat([df, pd.DataFrame([summed])], ignore_index=True), summed.to_frame().T


def summarize_non_gemm(prof_dir: str):
    """
    For every model subdirectory in *prof_dir*, read the raw ``non_gemm.csv``
    and aggregate times into named operator categories.  Writes ``pct_*.csv``
    (and companion files) in-place.  Idempotent — safe to call repeatedly.
    """
    if not os.path.isdir(prof_dir):
        warnings.warn(f"Profile directory not found: {prof_dir}")
        return

    for dir_name in sorted(os.listdir(prof_dir)):
        dir_path = os.path.join(prof_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        non_gemm_file = os.path.join(dir_path, "non_gemm.csv")
        main_csv      = os.path.join(dir_path, f"{dir_name}.csv")
        if not os.path.exists(non_gemm_file) or not os.path.exists(main_csv):
            continue

        df_nongemm = pd.read_csv(non_gemm_file)
        _check_new_non_gemm(df_nongemm['name'].unique().tolist())

        df_summary_raw = pd.read_csv(main_csv)
        df_gng     = df_summary_raw[df_summary_raw['name'].isin(['GEMM', 'NonGEMM'])]
        df_gng_ssm = df_summary_raw[df_summary_raw['name'].isin(['GEMM', 'NonGEMM', 'SSM_Scan'])]
        df_summary = df_summary_raw[df_summary_raw['name'].isin(['GEMM', 'SSM_Scan'])]
        if 'Unnamed: 0' in df_summary.columns:
            df_summary = df_summary.drop(columns=['Unnamed: 0'])

        for group, op_list in non_gemm_ops_dict.items():
            filtered = _filter_df(df_nongemm, op_list)
            filtered, summary_row = _sum_and_append(filtered, group)
            df_summary = pd.concat([df_summary, summary_row], ignore_index=True)
            filtered.to_csv(os.path.join(dir_path, f"{group}.csv"))

        df_summary.to_csv(os.path.join(dir_path, f"summary_{dir_name}.csv"))
        df_gng.to_csv(os.path.join(dir_path, f"gng_{dir_name}.csv"))
        df_gng_ssm.to_csv(os.path.join(dir_path, f"gng_ssm_{dir_name}.csv"))

        for df_, prefix in [
            (df_summary[["name", "total_time (us)"]],  "pct"),
            (df_gng[["name", "total_time (us)"]],      "gng_pct"),
            (df_gng_ssm[["name", "total_time (us)"]], "gng_ssm_pct"),
        ]:
            total = df_['total_time (us)'].sum()
            df_ = df_.copy()
            df_['pct'] = (df_['total_time (us)'] / total) * 100
            df_.to_csv(os.path.join(dir_path, f"{prefix}_{dir_name}.csv"))


# ===========================================================================
# Data loading
# ===========================================================================

def load_pct_csv(prof_dir: str, model: str, seq_len: int,
                 device='cuda', batch_size=1) -> Optional[pd.DataFrame]:
    """Load ``pct_<model>_<device>_<bs>_<seq>.csv`` for a single (model, device)."""
    tag   = f"{model}_{device}_{batch_size}_{seq_len}"
    fpath = os.path.join(prof_dir, tag, f"pct_{tag}.csv")
    if not os.path.exists(fpath):
        return None
    return pd.read_csv(fpath)


def extract_comparison_data(model_categories, seq_len, desktop_dir, jetson_dir):
    """
    Return a nested dict:  data[model] = {'desktop': df | None, 'jetson': df | None}
    where df has columns ['name', 'pct'].
    """
    data = {}
    all_models = [m for models in model_categories.values() for m in models]
    for model in all_models:
        desktop_df = load_pct_csv(desktop_dir, model, seq_len)
        jetson_df  = load_pct_csv(jetson_dir,  model, seq_len)
        if desktop_df is None:
            warnings.warn(f"Desktop data not found for '{model}' at seq_len={seq_len} "
                          f"in '{desktop_dir}'")
        if jetson_df is None:
            warnings.warn(f"Jetson data not found for '{model}' at seq_len={seq_len} "
                          f"in '{jetson_dir}'")
        data[model] = {'desktop': desktop_df, 'jetson': jetson_df}
    return data


# ===========================================================================
# Plotting
# ===========================================================================

def _global_op_order(data):
    """Pin GEMM and SSM_Scan at bottom; sort rest by average contribution across all models.
    Used for the legend only — actual bar stacking uses per-model order."""
    pinned  = ['SSM_Scan', 'GEMM']
    tallies = {}
    for model_data in data.values():
        for device, df in model_data.items():
            if df is None:
                continue
            for _, row in df.iterrows():
                op = row['name']
                if op in pinned:
                    continue
                tallies.setdefault(op, []).append(float(row.get('pct', 0)))
    avg = {op: (sum(v) / len(v)) for op, v in tallies.items() if v}
    sorted_rest = sorted(avg, key=avg.get, reverse=True)
    return pinned + sorted_rest


def _model_op_order(model_data, global_op_order):
    """Return per-model stacking order.

    Mirrors the notebook: GEMM and SSM_Scan are pinned at the very bottom,
    then non-GEMM ops are sorted by *this model's* average contribution
    (desktop + Jetson) in descending order so the highest-contributing
    non-GEMM op sits just above GEMM.

    Any op that appears in global_op_order but has zero contribution for
    this model is kept at the end so it is rendered (as a zero-height bar)
    without disrupting the ordering.
    """
    pinned = ['SSM_Scan', 'GEMM']
    tallies = {}
    for device, df in model_data.items():
        if df is None:
            continue
        for _, row in df.iterrows():
            op = row['name']
            if op in pinned:
                continue
            tallies.setdefault(op, []).append(float(row.get('pct', 0)))
    avg = {op: (sum(v) / len(v)) for op, v in tallies.items() if v}
    # All non-pinned ops from the global list, sorted by this model's avg (desc)
    rest = [op for op in global_op_order if op not in pinned]
    rest_sorted = sorted(rest, key=lambda op: avg.get(op, 0), reverse=True)
    return pinned + rest_sorted


def _get_pct(df, op_name):
    if df is None:
        return 0.0
    rows = df[df['name'] == op_name]
    if rows.empty:
        return 0.0
    return float(rows.iloc[0]['pct'])


def _build_op_legend(op_order):
    """Build Patch handles for the operator-category legend."""
    handles, labels = [], []
    seen = set()
    for op in op_order:
        if op in color_scheme and op not in seen:
            handles.append(mpatches.Patch(color=color_scheme[op], label=op))
            labels.append(op)
            seen.add(op)
    return handles, labels


def plot_device_comparison(
    model_categories,
    data,
    seq_len: int,
    annotated: bool = False,
):
    """
    Create the paired stacked-bar cross-device comparison figure.

    Parameters
    ----------
    model_categories : dict[str, list[str]]
        Architecture family → ordered list of model keys.
    data : dict[str, dict[str, pd.DataFrame|None]]
        Nested dict from extract_comparison_data().
    seq_len : int
        Sequence length used (for title / filename).
    annotated : bool
        If True, add axis labels, title and a full legend.

    Returns
    -------
    (fig, ax)
    """
    all_models = [m for models in model_categories.values() for m in models]
    valid_models = [m for m in all_models
                    if data[m]['desktop'] is not None or data[m]['jetson'] is not None]
    if not valid_models:
        print("No valid model data found.  Nothing to plot.")
        return None, None

    # Global order is used only for the legend so colour assignment is consistent.
    global_op_order = _global_op_order({m: data[m] for m in valid_models})

    n_models  = len(valid_models)
    bar_width = 0.35
    indices   = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(n_models * 2, 14), 8))

    for i, model in enumerate(valid_models):
        desktop_df = data[model]['desktop']
        jetson_df  = data[model]['jetson']

        # Per-model stacking order: highest-contributing non-GEMM op sits
        # just above GEMM, matching the notebook's plot_device_comparison().
        model_order = _model_op_order(data[model], global_op_order)

        desktop_bottom = 0.0
        jetson_bottom  = 0.0

        for op in model_order:
            color        = color_scheme.get(op, '#808080')
            desktop_val  = _get_pct(desktop_df, op)
            jetson_val   = _get_pct(jetson_df,  op)

            if desktop_val > 0:
                ax.bar(
                    indices[i] - bar_width / 2, desktop_val, bar_width,
                    bottom=desktop_bottom,
                    color=color, edgecolor='white', linewidth=0.5,
                )
                desktop_bottom += desktop_val

            if jetson_val > 0:
                ax.bar(
                    indices[i] + bar_width / 2, jetson_val, bar_width,
                    bottom=jetson_bottom,
                    color=color, edgecolor='white', linewidth=0.5,
                )
                jetson_bottom += jetson_val

        # "D" / "J" micro-labels above each bar pair
        if annotated:
            label_y = max(desktop_bottom, jetson_bottom) + 1.0
            ax.text(indices[i] - bar_width / 2, label_y, 'D',
                    ha='center', va='bottom', fontsize=7, color='#333')
            ax.text(indices[i] + bar_width / 2, label_y, 'J',
                    ha='center', va='bottom', fontsize=7, color='#333')

    # Category dividers and headers
    start_idx = 0
    for category, models in model_categories.items():
        valid_cat = [m for m in models if m in valid_models]
        if not valid_cat:
            continue
        n = len(valid_cat)
        mid = start_idx + n / 2 - 0.5
        if start_idx > 0:
            ax.axvline(x=start_idx - 0.5, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.6)
        if annotated:
            ax.text(mid, 104, category, ha='center', fontsize=11, fontweight='bold')
        start_idx += n

    # X-axis ticks
    ax.set_xticks(indices)
    if annotated:
        x_labels = [MODEL_LABELS.get(m, m) for m in valid_models]
        ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=9)
    else:
        ax.set_xticklabels(['' for _ in valid_models])

    ax.set_ylim(0, 108 if annotated else 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if annotated:
        ax.set_ylabel('GPU kernel time (%)', fontsize=12)
        ax.set_title(
            f'Cross-Device Operator Breakdown: Desktop vs Jetson '
            f'(seq_len = {seq_len})',
            fontsize=14,
        )
        # Device legend
        device_handles = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='D = Desktop GPU'),
            mpatches.Patch(facecolor='white', edgecolor='black',
                           hatch='//', label='J = Jetson Orin Nano'),
        ]
        # Operator legend (use global order for consistent colour assignment)
        op_handles, op_labels = _build_op_legend(global_op_order)
        legend1 = ax.legend(
            op_handles, op_labels,
            loc='upper center', bbox_to_anchor=(0.5, -0.20),
            ncol=min(6, len(op_handles)), fontsize=8,
            title='Operator category', title_fontsize=9,
        )
        ax.add_artist(legend1)
        ax.legend(
            device_handles,
            ['D = Desktop GPU', 'J = Jetson Orin Nano'],
            loc='upper center', bbox_to_anchor=(0.5, -0.32),
            ncol=2, fontsize=8,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38)
    else:
        plt.tight_layout()

    return fig, ax


# ===========================================================================
# CLI
# ===========================================================================

def _default_dir(rel):
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
    return os.path.join(_src, rel)


def parse_args():
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Reproduce Figure 9b — cross-device operator breakdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--desktop_dir",
        type=str,
        default=_default_dir("profile_logs"),
        help="Root directory containing Desktop profiling CSVs "
             "(output of collect_fig9b_data.py on the workstation).",
    )
    p.add_argument(
        "--jetson_dir",
        type=str,
        default=_default_dir("profile_logs_jetson"),
        help="Root directory containing Jetson profiling CSVs "
             "(transferred from the Jetson board).",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length to compare (must match collected data).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=_script_dir,
        help="Directory where output PNG files are written.",
    )
    p.add_argument(
        "--skip_summarize",
        action="store_true",
        help="Skip summarize_non_gemm() if pct_*.csv files already exist.",
    )
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    desktop_dir = os.path.abspath(args.desktop_dir)
    jetson_dir  = os.path.abspath(args.jetson_dir)
    seq_len     = args.seq_len
    out_dir     = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Desktop profile dir : {desktop_dir}")
    print(f"Jetson  profile dir : {jetson_dir}")
    print(f"Sequence length     : {seq_len}")
    print(f"Output  dir         : {out_dir}")

    # Summarise non-GEMM ops so pct_*.csv files exist
    if not args.skip_summarize:
        print("\nSummarising non-GEMM operators (desktop) …")
        summarize_non_gemm(desktop_dir)
        print("Summarising non-GEMM operators (Jetson) …")
        summarize_non_gemm(jetson_dir)

    # Load comparison data
    data = extract_comparison_data(MODEL_CATEGORIES, seq_len, desktop_dir, jetson_dir)

    # --- Publication figure (no annotations) --------------------------------
    fig, ax = plot_device_comparison(
        MODEL_CATEGORIES, data, seq_len, annotated=False)
    if fig is not None:
        pub_path = os.path.join(out_dir, f"fig9b_device_comparison_seq{seq_len}.png")
        fig.savefig(pub_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"\nSaved (publication) : {pub_path}")

    # --- Annotated figure ---------------------------------------------------
    fig, ax = plot_device_comparison(
        MODEL_CATEGORIES, data, seq_len, annotated=True)
    if fig is not None:
        ann_path = os.path.join(
            out_dir, f"fig9b_device_comparison_seq{seq_len}_annotated.png")
        fig.savefig(ann_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved (annotated)   : {ann_path}")


if __name__ == "__main__":
    main()
