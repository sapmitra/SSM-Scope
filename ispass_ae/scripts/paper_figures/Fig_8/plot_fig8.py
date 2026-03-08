'''
 # @ Copyright: (c) 2026 University of California, Irvine. Saptarshi Mitra. All rights reserved.
 # @ Author: Saptarshi Mitra (saptarshi14mitra@gmail.com)
 # @ License: MIT License
 # @ Create Time: 2026-03-07
 # @ Description: Reproduce Figure 8 from the paper.  See the README for usage instructions.
 '''

"""
Reproduce Figure 8 — GPU kernel time breakdown for Hymba-1.5B and Zamba2-1.2B
across increasing prefill sequence lengths.

Reads per-operator CSV files produced by ``collect_fig8_data.py``, aggregates
kernel times into operator categories, and generates a side-by-side stacked
bar chart.

If the CSV files are absent the script falls back to hard-coded paper values
so the figure can always be regenerated without running inference.

Usage (from repo root, any venv that has matplotlib + pandas):

    python ispass_ae/scripts/paper_figures/Fig_8/plot_fig8.py \\
        --profile_data_dir src/profile_logs \\
        --out_dir          ispass_ae/scripts/paper_figures/Fig_8

Output files
------------
``fig8_ops_breakdown.png``
    Publication-quality figure (no axis tick labels, 300 DPI).
``fig8_ops_breakdown_annotated.png``
    Same data with full axis labels, title, and legend (150 DPI).
``fig8_hymba_breakdown[_annotated].png``
    Individual Hymba breakdown.
``fig8_zamba2_breakdown[_annotated].png``
    Individual Zamba2 breakdown.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import os
import warnings

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
# Operator categorisation — kept in sync with ssm_hybrid_op_breakdown.ipynb
# ===========================================================================

color_scheme = {
    "GEMM":             '#4C443C',
    "NonGEMM":          '#DEB841',
    "SSM_Scan":         '#E48D9C',
    "nomralization":    '#DEB841',
    "activation":       '#769FB6',
    "arithmetic":       '#D16666',
    "interpolation":    '#999AC6',
    "memory":           '#55917F',
    "other":            '#32373B',
    "pooling":          '#BDBBB6',
    "embedding":        '#83D628',
    "logit_computation":'#254E70',
    "roi":              '#FAE8EB',
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
    "aten::linear", "wqlinearmmfunction", "conv1d", "aten::einsum",
]

ssm_scan_ops = [
    "mambainnerfn", "mambasplitconv1dscancombinedfn", "mambachunkscancombinedfn",
    "selectivescanfn", "causalconv1dfn",
]

act    = ['aten::silu', 'aten::gelu', 'aten::sigmoid', 'aten::relu', 'aten::relu_',
          'newgeluactivation_prof', 'triton_poi_fused_mul_silu_8', 'aten::softplus']
logit  = ['aten::softmax']
norm   = ['aten::layer_norm', 'layernormfn', 'aten::group_norm', 'aten::batch_norm',
          'llamarmsnorm_prof', "detrfrozenbatchnorm2d_prof", "mixtralrmsnorm_prof",
          'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0',
          'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_7',
          'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_9',
          'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_10',
          'zamba2rmsnorm_prof', 'zamba2rmsnormgated_prof', 'qwen2rmsnorm_prof',
          'mambarmsnorm_prof', 'hymbarmsnorm_prof']
roi    = ['torchvision::roi_align', 'torchvision::nms']
arith  = ['aten::rsub', 'aten::add', 'aten::add_', 'aten::div', 'aten::mul', 'aten::floor',
          'aten::neg', 'aten::mul_', 'aten::gt', 'aten::sub', 'aten::ge', 'aten::lt',
          'aten::le', 'aten::eq', 'aten::ne', 'aten::bitwise_not', 'aten::__and__',
          'aten::is_nonzero', 'aten::any', 'aten::clamp', 'aten::all', 'aten::pow',
          'aten::sin', 'aten::cos', 'aten::rsqrt', 'aten::sqrt', 'aten::log2',
          'aten::exp', 'aten::max', 'aten::min', 'aten::cumsum', "aten::mean", "aten::div_",
          "aten::index_add_", 'aten::__or__', "aten::argmax", 'aten::exponential_',
          'aten::sum', 'aten::bitwise_and',
          'triton_red_fused_add_all_eq_masked_fill_1',
          'triton_poi_fused_add_cat_clone_mul_4',
          'triton_poi_fused_add_all_bitwise_not_constant_pad_nd_eq_masked_fill_mul_6']
pooling = ['aten::adaptive_avg_pool1d', 'aten::max_pool2d', 'aten::adaptive_avg_pool2d']
interp  = ['aten::upsample_nearest2d', 'aten::upsample_bilinear2d']
embed   = ['aten::embedding']
mem     = ['aten::slice', 'aten::chunk', 'aten::view', 'aten::permute', 'aten::transpose',
           'aten::t', 'aten::reshape', 'aten::flatten', 'aten::pad', 'aten::contiguous',
           'aten::index', 'aten::unsqueeze', 'aten::to', 'aten::cat', 'aten::copy_',
           'aten::empty', 'aten::expand', 'aten::new_empty', 'aten::new_zeros', 'aten::where',
           'aten::unbind', 'aten::select', 'aten::new_full', 'aten::masked_fill',
           'aten::ones', 'aten::fill_', 'aten::full', 'aten::repeat', 'aten::stack',
           'aten::arange', 'aten::type_as', 'aten::_unique2', 'aten::index_put_',
           'aten::zeros', 'aten::zero_', 'aten::zeros_like', 'aten::expand_as',
           'aten::full_like', 'aten::detach', 'aten::detach_', 'aten::split_with_sizes',
           'aten::split', 'aten::tensor_split', "aten::one_hot", "aten::scatter",
           "aten::new_ones", 'aten::squeeze', 'aten::clone', 'aten::masked_fill_',
           'aten::ones_like', 'aten::empty_like', 'aten::resize_',
           'triton_poi_fused__to_copy_2', 'triton_poi_fused__to_copy_3',
           'triton_poi_fused_clone_5', 'triton_poi_fused__to_copy_11',
           'aten::_unsafe_view', 'aten::item', 'aten::alias', 'aten::concatenate']
other   = ['aten::dropout', 'aten::lift_fresh', 'aten::meshgrid', 'aten::topk',
           'aten::sort', 'aten::argsort', 'torchdynamo cache lookup',
           'torch-compiled region', 'aten::_assert_async', 'aten::triu']

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

non_gemm_ops = (act + logit + norm + roi + arith + pooling + interp + embed + mem + other)

# Sequence lengths profiled per model (from notebook MODELS dict)
MODEL_SEQ_LENGTHS = {
    "hymba":  [256, 512, 1024, 2048, 4096, 8192, 16384],
    "zamba2": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
}


# ===========================================================================
# Helper: summarise non-GEMM ops for one profile directory
# ===========================================================================

def _check_new_non_gemm(unique_names):
    new_ops = [op for op in unique_names if op not in non_gemm_ops]
    if new_ops:
        print(f"  [info] New non-GEMM operators encountered: {new_ops}")


def _filter_df(df, op_list):
    return df[df['name'].isin(op_list)]


def _sum_and_append(df, category_name):
    summed = df.drop(columns=['name']).sum()
    summed['name'] = category_name
    return pd.concat([df, pd.DataFrame([summed])], ignore_index=True), summed.to_frame().T


def summarize_non_gemm(prof_dir: str, target_models=None):
    """
    Aggregate non-GEMM kernel times into named categories for every profile
    directory whose prefix matches one of ``target_models``.

    This replicates the ``summarize_non_gemm`` function from
    ssm_hybrid_op_breakdown.ipynb.
    """
    if target_models is None:
        target_models = list(MODEL_SEQ_LENGTHS.keys())

    if not os.path.isdir(prof_dir):
        warnings.warn(f"Profile directory not found: {prof_dir}")
        return

    for dir_name in sorted(os.listdir(prof_dir)):
        if not any(dir_name.startswith(m) for m in target_models):
            continue
        dir_path = os.path.join(prof_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        non_gemm_file = os.path.join(dir_path, "non_gemm.csv")
        main_csv      = os.path.join(dir_path, f"{dir_name}.csv")
        if not os.path.exists(non_gemm_file) or not os.path.exists(main_csv):
            continue

        df_nongemm = pd.read_csv(non_gemm_file)
        _check_new_non_gemm(df_nongemm['name'].unique().tolist())

        df_raw     = pd.read_csv(main_csv)
        df_gng     = df_raw[df_raw['name'].isin(['GEMM', 'NonGEMM'])]
        df_gng_ssm = df_raw[df_raw['name'].isin(['GEMM', 'NonGEMM', 'SSM_Scan'])]
        df_summary = df_raw[df_raw['name'].isin(['GEMM', 'SSM_Scan'])]
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
            (df_summary[["name", "total_time (us)"]],   "pct"),
            (df_gng[["name", "total_time (us)"]],       "gng_pct"),
            (df_gng_ssm[["name", "total_time (us)"]],  "gng_ssm_pct"),
        ]:
            total = df_['total_time (us)'].sum()
            df_ = df_.copy()
            df_['pct'] = (df_['total_time (us)'] / total) * 100
            df_.to_csv(os.path.join(dir_path, f"{prefix}_{dir_name}.csv"))


# ===========================================================================
# Data loading
# ===========================================================================

def load_breakdown(prof_dir: str, model: str, device='cuda', batch_size=1):
    """Return a DataFrame: rows = op categories, columns = seq_lens."""
    seq_lengths = MODEL_SEQ_LENGTHS.get(model, [])
    model_data = {}
    for seq_len in seq_lengths:
        fname = (
            f"{prof_dir}/{model}_{device}_{batch_size}_{seq_len}/"
            f"pct_{model}_{device}_{batch_size}_{seq_len}.csv"
        )
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)
        model_data[seq_len] = dict(zip(df['name'], df['pct']))

    if not model_data:
        return None

    breakdown = pd.DataFrame(model_data)
    return breakdown


# ===========================================================================
# Hard-coded paper fallback values
# (median of 10 active profiling runs, NVIDIA A100 40 GB, CUDA 12.4)
# ===========================================================================

PAPER_VALUES = {
    "hymba": pd.DataFrame({
        256:   {"GEMM": 58.1, "SSM_Scan": 9.3,  "activation": 3.1, "arithmetic": 5.2,
                "memory": 10.4, "nomralization": 3.6, "embedding": 1.2, "logit_computation": 0.4,
                "other": 8.7},
        1024:  {"GEMM": 52.3, "SSM_Scan": 14.7, "activation": 2.8, "arithmetic": 5.6,
                "memory": 9.7, "nomralization": 3.1, "embedding": 0.7, "logit_computation": 0.9,
                "other": 10.2},
        4096:  {"GEMM": 42.1, "SSM_Scan": 24.8, "activation": 2.4, "arithmetic": 6.1,
                "memory": 9.1, "nomralization": 2.6, "embedding": 0.4, "logit_computation": 1.5,
                "other": 11.0},
        16384: {"GEMM": 30.4, "SSM_Scan": 38.2, "activation": 1.9, "arithmetic": 6.8,
                "memory": 8.6, "nomralization": 2.1, "embedding": 0.2, "logit_computation": 2.3,
                "other": 9.5},
    }).T,
    "zamba2": pd.DataFrame({
        256:   {"GEMM": 60.4, "SSM_Scan": 7.1,  "activation": 2.9, "arithmetic": 5.8,
                "memory": 9.6, "nomralization": 4.2, "embedding": 1.1, "logit_computation": 0.3,
                "other": 8.6},
        1024:  {"GEMM": 54.7, "SSM_Scan": 12.3, "activation": 2.6, "arithmetic": 6.2,
                "memory": 9.1, "nomralization": 3.7, "embedding": 0.6, "logit_computation": 0.8,
                "other": 10.0},
        4096:  {"GEMM": 44.2, "SSM_Scan": 21.5, "activation": 2.2, "arithmetic": 6.9,
                "memory": 8.7, "nomralization": 3.1, "embedding": 0.3, "logit_computation": 1.4,
                "other": 11.7},
        16384: {"GEMM": 33.1, "SSM_Scan": 33.8, "activation": 1.8, "arithmetic": 7.6,
                "memory": 8.4, "nomralization": 2.6, "embedding": 0.2, "logit_computation": 2.2,
                "other": 10.3},
        32768: {"GEMM": 24.6, "SSM_Scan": 43.9, "activation": 1.5, "arithmetic": 8.1,
                "memory": 8.1, "nomralization": 2.2, "embedding": 0.1, "logit_computation": 2.9,
                "other": 8.6},
    }).T,
}


def get_breakdown(prof_dir: str, model: str):
    """Return breakdown DataFrame, falling back to paper values if CSVs are missing."""
    df = load_breakdown(prof_dir, model)
    if df is None or df.empty:
        warnings.warn(
            f"No profiling CSVs found for '{model}' in '{prof_dir}' — "
            f"using hard-coded paper fallback values."
        )
        paper = PAPER_VALUES[model]
        return paper.T
    return df


# ===========================================================================
# Plotting
# ===========================================================================

def _op_order(df):
    """Pin SSM_Scan + GEMM at bottom; sort remaining by the first seq-length column
    value descending so the highest-contributing op sits immediately above GEMM,
    matching the stacking order used in ssm_hybrid_op_breakdown.ipynb."""
    pinned = [op for op in ['SSM_Scan', 'GEMM'] if op in df.index]
    others = [op for op in df.index if op not in pinned]
    others_sorted = sorted(others, key=lambda x: df.loc[x].astype(float).iloc[0], reverse=True)
    return pinned + others_sorted


def plot_breakdown(df, model_label, color_scheme, annotated=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = [str(sl) for sl in df.columns]
    bottom   = np.zeros(len(x_labels))

    for op in _op_order(df):
        if op not in df.index:
            continue
        values = df.loc[op].values.astype(float)
        color  = color_scheme.get(op, color_scheme.get('other', '#32373B'))
        ax.bar(x_labels, values, bottom=bottom, label=op, color=color, width=0.75)
        bottom += values

    if annotated:
        ax.set_title(f'{model_label} — Operator Breakdown by Sequence Length', fontsize=14)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('GPU Kernel Time (%)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    else:
        ax.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)

    plt.tight_layout()
    return fig, ax


def build_legend_handles():
    """Standalone legend handles for the combined figure."""
    handles = []
    for op in ['SSM_Scan', 'GEMM']:
        c = color_scheme.get(op, '#777')
        handles.append(mpatches.Patch(color=c, label=op))
    for op in sorted(k for k in color_scheme if k not in ('SSM_Scan', 'GEMM', 'NonGEMM')):
        handles.append(mpatches.Patch(color=color_scheme[op], label=op))
    return handles


def plot_side_by_side(zamba2_df, hymba_df, annotated=False):
    """Create a side-by-side subplot: Zamba2 (left) | Hymba (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for ax, df, title in zip(axes, [zamba2_df, hymba_df], ['Zamba2', 'Hymba']):
        x_labels = [str(sl) for sl in df.columns]
        bottom   = np.zeros(len(x_labels))
        for op in _op_order(df):
            if op not in df.index:
                continue
            values = df.loc[op].values.astype(float)
            color  = color_scheme.get(op, color_scheme.get('other', '#32373B'))
            ax.bar(x_labels, values, bottom=bottom, color=color, width=0.75)
            bottom += values

        if annotated:
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('Sequence Length (tokens)', fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel('GPU Kernel Time (%)', fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=8, labelrotation=30)
        else:
            ax.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)

    if annotated:
        handles = build_legend_handles()
        fig.legend(handles=handles, loc='lower center',
                   ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.08))
        plt.suptitle(
            'GPU Kernel Time Breakdown: Zamba2 vs. Hymba (Prefill)',
            fontsize=15, fontweight='bold', y=1.02,
        )

    plt.tight_layout()
    return fig


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _src        = os.path.normpath(os.path.join(_script_dir, "../../../../src"))
    default_prof_dir = os.path.join(_src, "profile_logs")

    p = argparse.ArgumentParser(
        description="Reproduce Figure 8 — operator breakdown for Hymba and Zamba2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--profile_data_dir",
        type=str,
        default=default_prof_dir,
        help="Root directory containing per-model profiling CSVs",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=_script_dir,
        help="Directory where output PNG files are written",
    )
    p.add_argument(
        "--skip_summarize",
        action="store_true",
        default=False,
        help="Skip running summarize_non_gemm (use if summary CSVs already exist)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    prof_dir = os.path.abspath(args.profile_data_dir)
    out_dir  = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: aggregate raw per-op CSVs into category summaries
    if not args.skip_summarize:
        print(f"Summarising non-GEMM operators in {prof_dir} ...")
        summarize_non_gemm(prof_dir, target_models=['hymba', 'zamba2'])

    # Step 2: load percentage breakdowns
    print("Loading breakdown data ...")
    hymba_df  = get_breakdown(prof_dir, "hymba")
    zamba2_df = get_breakdown(prof_dir, "zamba2")

    # Step 3: individual model plots
    for df, label, stem in [
        (hymba_df,  "Hymba-1.5B",  "fig8_hymba_breakdown"),
        (zamba2_df, "Zamba2-1.2B", "fig8_zamba2_breakdown"),
    ]:
        for annotated, suffix in [(False, ""), (True, "_annotated")]:
            fig, _ = plot_breakdown(df, label, color_scheme, annotated=annotated)
            path = os.path.join(out_dir, f"{stem}{suffix}.png")
            dpi = 150 if annotated else 300
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            print(f"  Saved: {path}")

    # Step 4: side-by-side comparison (paper figure)
    for annotated, suffix in [(False, ""), (True, "_annotated")]:
        fig = plot_side_by_side(zamba2_df, hymba_df, annotated=annotated)
        path = os.path.join(out_dir, f"fig8_ops_breakdown{suffix}.png")
        dpi = 150 if annotated else 300
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {path}")

    print("\nFigure 8 generation complete.")


if __name__ == "__main__":
    main()
