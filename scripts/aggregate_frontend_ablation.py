import argparse
import json
import os
import pandas as pd
from pathlib import Path

# The expected columns and metrics
METRICS = [
    "accuracy", "balanced_accuracy", "macro_f1", "top1_accuracy", "top5_accuracy",
    "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "median_rank",
    "semantic_similarity", "wer", "cer", "params", "trainable_params",
    "flops_or_macs", "latency_ms", "peak_memory_mb", "checkpoint_size_mb"
]

VARIANTS_CONFIG = {
    'baseline':              {'ghost': False, 'dwaspp': False, 'cas': False, 'normal_conv': False, 'purpose': 'Reference Graph-Conformer'},
    'ghost_only':            {'ghost': True,  'dwaspp': False, 'cas': False, 'normal_conv': False, 'purpose': 'Parameter-efficient projection'},
    'dwaspp_only':           {'ghost': False, 'dwaspp': True,  'cas': False, 'normal_conv': False, 'purpose': 'Multi-scale temporal modelling'},
    'cas_only':              {'ghost': False, 'dwaspp': False, 'cas': True,  'normal_conv': False, 'purpose': 'Frequency-time recalibration'},
    'ghost_dwaspp':          {'ghost': True,  'dwaspp': True,  'cas': False, 'normal_conv': False, 'purpose': 'Lightweight multi-scale extraction'},
    'ghost_cas':             {'ghost': True,  'dwaspp': False, 'cas': True,  'normal_conv': False, 'purpose': 'Lightweight projection with attention'},
    'dwaspp_cas':            {'ghost': False, 'dwaspp': True,  'cas': True,  'normal_conv': False, 'purpose': 'Multi-scale extraction with attention'},
    'full_ghost_dwaspp_cas': {'ghost': True,  'dwaspp': True,  'cas': True,  'normal_conv': False, 'purpose': 'Proposed lightweight front-end'},
    'normalconv_dwaspp_cas': {'ghost': False, 'dwaspp': True,  'cas': True,  'normal_conv': True,  'purpose': 'Check Ghost cost vs normal conv'}
}

def determine_primary_metric(row):
    """Dynamically determine the primary metric string to display based on available data."""
    if pd.notna(row.get('macro_f1')):
        return f"{row['macro_f1']:.4f} (Macro-F1)"
    elif pd.notna(row.get('recall_at_5')):
        return f"{row['recall_at_5']:.4f} (Recall@5)"
    elif pd.notna(row.get('wer')):
        return f"{row['wer']:.4f} (WER)"
    return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Aggregate frontend ablation results")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--seed", required=True, help="Random seed used")
    args = parser.parse_args()

    base_dir = Path(f"results/ablations/frontend/{args.dataset}/{args.task}")
    
    rows = []
    
    for variant, config in VARIANTS_CONFIG.items():
        row = {
            "variant": variant,
            "use_ghost": config['ghost'],
            "use_dwaspp": config['dwaspp'],
            "use_cas": config['cas'],
            "use_normal_conv": config['normal_conv'],
            "purpose": config['purpose']
        }
        
        metrics_path = base_dir / variant / f"seed_{args.seed}" / "metrics.json"
        
        # Default all metrics to NaN
        for m in METRICS:
            row[m] = pd.NA
            
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    for m in METRICS:
                        if m in data:
                            row[m] = data[m]
            except Exception as e:
                print(f"Error reading {metrics_path}: {e}")
        else:
            print(f"Warning: metrics.json not found for {variant} at {metrics_path}")
            
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # 1. Output CSV
    csv_path = base_dir / "summary.csv"
    os.makedirs(base_dir, exist_ok=True)
    
    csv_columns = ["variant", "use_ghost", "use_dwaspp", "use_cas", "use_normal_conv"] + METRICS
    df[csv_columns].to_csv(csv_path, index=False, na_rep='NaN')
    
    # 2. Output Markdown
    df['Primary metric'] = df.apply(determine_primary_metric, axis=1)
    
    # Format markdown table
    md_cols = [
        "variant", "use_ghost", "use_dwaspp", "use_cas", "use_normal_conv",
        "Primary metric", "params", "flops_or_macs", "latency_ms", "purpose"
    ]
    
    # Rename columns for the paper table
    md_renames = {
        "variant": "Variant",
        "use_ghost": "Ghost",
        "use_dwaspp": "DWASPP",
        "use_cas": "CAS",
        "use_normal_conv": "Normal Conv",
        "params": "Params",
        "flops_or_macs": "FLOPs/MACs",
        "latency_ms": "Latency (ms)",
        "purpose": "Purpose"
    }
    
    md_df = df[md_cols].rename(columns=md_renames)
    
    # Replace boolean with checkmarks / crosses
    for col in ["Ghost", "DWASPP", "CAS", "Normal Conv"]:
        md_df[col] = md_df[col].map({True: '✓', False: '✗'})
        
    md_path = base_dir / "summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Frontend Ablation Summary\n\n")
        f.write(f"**Dataset:** {args.dataset} | **Task:** {args.task} | **Seed:** {args.seed}\n\n")
        f.write(md_df.to_markdown(index=False))
        f.write("\n")

    print(f"Aggregation complete.")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")

if __name__ == "__main__":
    main()
