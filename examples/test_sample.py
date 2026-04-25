import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from be_great import GReaT

import logging
from utils import set_logging_level
from sklearn import datasets

logger = set_logging_level(logging.INFO)

great = GReaT.load_from_dir("trained/adult_topology")

# Continuous column as start
# data, target = datasets.load_iris(return_X_y=True)
# sepal = list(data[:, 0])
# samples = great.sample(20, device="cpu", k=5, start_col="sepal length", start_col_dist=sepal)

# Random Start
# samples = great.sample(12, device="cpu", k=6)

# Categorical column as start

os.makedirs("result/adult/topology/", exist_ok=True)
for i in range(1, 4):
    samples = great.sample(
        n_samples=16281, # adult
        # n_samples=1600, # healthcare
        # n_samples=8000, # asia
        device="cuda", 
        guided_sampling=True,
        start_col="race" #topology
        )

    # print(samples)
    samples.to_csv(f"result/adult/topology/adult_test_topology{i}.csv")
