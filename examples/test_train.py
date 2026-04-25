import sys
import os
import logging
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from be_great.great import GReaT
from sklearn import datasets
from utils import set_logging_level

# # healthcare data
# base_dir = os.path.dirname(os.path.abspath(__file__))
# df = pd.read_csv(os.path.join(base_dir, "healthcare.txt"), sep=" ")
# train_df = df.iloc[:1600]
# test_df = df.iloc[1600:]
# cols = ["A", "C", "D", "H", "I", "O", "T"]

# asia data
# import bnlearn as bn
# df = bn.import_example("asia")
# train_df = df.iloc[:8000]
# test_df = df.iloc[8000:]
# cols = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"]

# adult data
from ucimlrepo import fetch_ucirepo
adult = fetch_ucirepo(id=2)
df = adult.data.features.copy()
df["income"] = adult.data.targets["income"]
df["income"] = df["income"].str.replace(".", "", regex=False)
df = df.drop(columns=["fnlwgt", "education-num", "capital-gain", "capital-loss"])
train_df = df.iloc[:32561]
test_df = df.iloc[32561:]
cols = ["age", "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "hours-per-week", "native-country", "income"]


logger = set_logging_level(logging.INFO)

# adult
dag_edges = [
    ("race", "education"),
    ("race", "occupation"),
    ("race", "income"),
    ("age", "education"),
    ("age", "hours-per-week"),
    ("age", "workclass"),
    ("age", "marital-status"),
    ("age", "occupation"),
    ("age", "income"),
    ("native-country", "education"),
    ("native-country", "hours-per-week"),
    ("native-country", "marital-status"),
    ("native-country", "occupation"),
    ("native-country", "income"),
    ("sex", "education"),
    ("sex", "hours-per-week"),
    ("sex", "marital-status"),
    ("sex", "relationship"),
    ("sex", "income"),
    ("education", "hours-per-week"),
    ("education", "workclass"),
    ("education", "marital-status"),
    ("education", "occupation"),
    ("education", "income"),
    ("hours-per-week", "marital-status"),
    ("hours-per-week", "occupation"),
    ("hours-per-week", "income"),
    ("workclass", "marital-status"),
    ("workclass", "occupation"),
    ("marital-status", "occupation"),
    ("marital-status", "relationship"),
    ("marital-status", "income"),
    ("occupation", "income"),
    ("relationship", "income"),
]


# healthcare
# dag_edges = [
#     ("A", "H"),
#     ("A", "D"),
#     ("A", "C"),
#     ("A", "O"),
#     ("H", "D"),
#     ("D", "I"),
#     ("C", "I"),
#     ("I", "T"),
#     ("O", "T"),
# ]

# asia
# dag_edges = [
#     ("asia", "tub"),   
#     ("tub", "either"),
#     ("smoke", "lung"),
#     ("smoke", "bronc"),
#     ("lung", "either"),
#     ("either", "xray"),
#     ("either", "dysp"),
#     ("bronc", "dysp"),
# ]

great = GReaT(
    "distilgpt2",
    epochs=200,
    save_steps=200000,
    logging_steps=5,
    experiment_dir="trained/trainer_adult_topology_bias",
    # lr_scheduler_type="constant", learning_rate=5e-5
)

dag_alpha = 1.0 * 1.0
dag_beta = 0.0
dag_gamma = 1.0 * 1.0
dag_bias_learnable = False  # False로 두면 alpha/beta/gamma 고정

trainer = great.fit(
    train_df,
    column_names=cols,
    dag_edges=dag_edges,
    dag_alpha_init=dag_alpha,
    dag_beta_init=dag_beta,
    dag_gamma_init=dag_gamma,
    dag_bias_learnable=dag_bias_learnable,
    conditional_col="race", #topology
    random_conditional_col=False, #topology
)

dag_bias = getattr(great.model, "dag_attention_bias", None)
# if dag_bias is not None:
#     print("\n[DAG Attention Bias Scalars]")
#     print(f"alpha: {dag_bias.alpha.item():.6f}")
#     print(f"beta : {dag_bias.beta.item():.6f}")
#     print(f"gamma: {dag_bias.gamma.item():.6f}")

great.save("trained/adult_topology_bias")

os.makedirs("result/adult/topology/", exist_ok=True)
for i in range(1, 2):
    samples = great.sample(
        n_samples=16281, # adult
        # n_samples=1600, # healthcare
        # n_samples=8000, # asia
        device="cuda", 
        guided_sampling=True,
        start_col="race" #topology
        )

    # print(samples)
    samples.to_csv(f"result/adult/topology/adult_test_topology_bias{i}.csv")
