import os
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "examples/healthcare.txt"), sep=" ")
train_df = df.iloc[:1600]
test_df = df.iloc[1600:]
cols = ["A", "C", "D", "H", "I", "O", "T"]

dag_edges = [
    ("A", "H"), ("A", "D"), ("A", "C"), ("A", "O"),
    ("H", "D"), ("D", "I"), ("C", "I"), ("I", "T"), ("O", "T"),
]

# 컬럼별 고유값 확인
print("Column unique values:")
for col in cols:
    print(f"  {col}: {sorted(df[col].unique())}")

# ================================================================
# Label Encoding
# ================================================================
def label_encode(df, cols):
    df = df.copy()
    for col in cols:
        if df[col].dtype == object:
            vals = sorted(df[col].unique())
            df[col] = df[col].map({v: i for i, v in enumerate(vals)})
    return df

train_df = label_encode(train_df, cols)
test_df = label_encode(test_df, cols)

# ================================================================
# SCM 학습
# ================================================================
def train_scm_model(dag_edges, data_df, cols):
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    G.add_edges_from(dag_edges)

    causal_model = gcm.StructuralCausalModel(G)

    for node in G.nodes:
        parents = list(G.predecessors(node))
        if len(parents) == 0:
            causal_model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        else:
            # causal_model.set_causal_mechanism(
            #     node, gcm.ClassifierFCM(gcm.ml.create_logistic_regression_classifier())
            # )
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_hist_gradient_boost_regressor()))


    gcm.fit(causal_model, data_df[cols])
    return causal_model

# ================================================================
# Interventional 샘플 생성
# ================================================================
def interv_gen(causal_model, inv_col, inv_val, cols, sz=1000):
    samples = gcm.interventional_samples(
        causal_model,
        {inv_col: lambda y: inv_val},
        num_samples_to_draw=sz
    )
    return samples[cols]

# ================================================================
# MAE 계산
# ================================================================
def compute_intervention_mae(causal_model_gt, causal_model_syn, data_df, cols, sz=1000):
    mae_list = []

    for inv_col in cols:
        inv_vals = sorted(data_df[inv_col].unique())
        for inv_val in inv_vals:
            gt_samples = interv_gen(causal_model_gt, inv_col, inv_val, cols, sz)
            syn_samples = interv_gen(causal_model_syn, inv_col, inv_val, cols, sz)

            other_cols = [c for c in cols if c != inv_col]
            col_maes = []
            for col in other_cols:
                gt_dist = gt_samples[col].value_counts(normalize=True).sort_index()
                syn_dist = syn_samples[col].value_counts(normalize=True).sort_index()
                all_vals = gt_dist.index.union(syn_dist.index)
                gt_dist = gt_dist.reindex(all_vals, fill_value=0)
                syn_dist = syn_dist.reindex(all_vals, fill_value=0)
                col_maes.append(np.mean(np.abs(gt_dist.values - syn_dist.values)))

            mae_list.append(np.mean(col_maes))

    return np.mean(mae_list)

# ================================================================
# 실행
# ================================================================
print("\nGround Truth SCM 학습 중...")
causal_model_gt = train_scm_model(dag_edges, train_df, cols)

all_mae = []

for i in range(1, 6):
    csv_path = f"result/health/topology/health_test_topology_bias{i}.csv"
    syn_df = pd.read_csv(csv_path, index_col=0)
    syn_df = syn_df[cols].copy()
    syn_df = label_encode(syn_df, cols)

    print(f"\n{'='*50}")
    print(f"Sample {i} - SCM 학습 중...")
    causal_model_syn = train_scm_model(dag_edges, syn_df, cols)

    mae = compute_intervention_mae(causal_model_gt, causal_model_syn, train_df, cols)
    all_mae.append(mae)
    print(f"Intervention MAE: {mae:.4f}")

print(f"\n{'='*50}")
print(f"Mean Intervention MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")

with open("result/health/topology/intervention_mae_bias.txt", "w") as f:
    f.write(f"Mean Intervention MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}\n")