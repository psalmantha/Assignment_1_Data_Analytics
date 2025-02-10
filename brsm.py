import pandas as pd
import numpy as np
import pulp
from scipy.spatial.distance import mahalanobis
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("brsm_dataset.csv")

required_columns = {"unit_id", "treatment", "age", "income", "risk_score"}
if not required_columns.issubset(df.columns):
    raise ValueError("Dataset is missing required columns.")

# separate treated and control groups
treated = df[df["treatment"] == 1]
control = df[df["treatment"] == 0]

if len(treated) > len(control):
    raise ValueError("More treated units than control units. Consider relaxing matching constraints.")

# compute covariance matrix using only control group
cov_matrix = np.cov(control[["age", "income"]].T)
cov_matrix_inv = np.linalg.pinv(cov_matrix)

# compute mahalanobis distance matrix
distance_matrix = np.zeros((len(treated), len(control)))
for i, (_, treat) in enumerate(treated.iterrows()):
    for j, (_, ctrl) in enumerate(control.iterrows()):
        diff = treat[["age", "income"]] - ctrl[["age", "income"]]
        distance_matrix[i, j] = mahalanobis(diff, np.zeros_like(diff), cov_matrix_inv)

# linear programming for optimal matching
problem = pulp.LpProblem("Mahalanobis_Matching", pulp.LpMinimize)
vars_dict = {(i, j): pulp.LpVariable(f"match_{i}_{j}", cat=pulp.LpBinary)
             for i in range(len(treated)) for j in range(len(control))}

# objective: minimize total distance
problem += pulp.lpSum(vars_dict[i, j] * distance_matrix[i, j] for i in range(len(treated)) for j in range(len(control)))

# constraints: each treated unit is matched once
for i in range(len(treated)):
    problem += pulp.lpSum(vars_dict[i, j] for j in range(len(control))) == 1

# constraints: each control unit is matched at most once
for j in range(len(control)):
    problem += pulp.lpSum(vars_dict[i, j] for i in range(len(treated))) <= 1

problem.solve()

# extract and save matched pairs
matched_pairs = []
for i, j in vars_dict:
    if vars_dict[i, j].varValue == 1:
        matched_pairs.append((treated.iloc[i]["unit_id"], control.iloc[j]["unit_id"]))

matched_df = pd.DataFrame(matched_pairs, columns=["treated_id", "control_id"])
matched_df.to_csv("matched_pairs.csv", index=False)

# statistical test (wilcoxon signed-rank test)
treated_risk_scores = df.loc[df["unit_id"].isin(matched_df["treated_id"]), "risk_score"].values
control_risk_scores = df.loc[df["unit_id"].isin(matched_df["control_id"]), "risk_score"].values

if len(treated_risk_scores) != len(control_risk_scores):
    raise ValueError("Mismatch in risk score length for Wilcoxon test.")

wilcoxon_result = wilcoxon(treated_risk_scores, control_risk_scores)
print(f"Wilcoxon test p-value: {wilcoxon_result.pvalue:.5f}")

# visualization
plt.figure(figsize=(8, 6))
sns.boxplot(data=[treated_risk_scores, control_risk_scores], palette=["#FF9999", "#99CCFF"])
plt.xticks([0, 1], ["Treated", "Control"])
plt.ylabel("Risk Score")
plt.title("Risk Score Distribution for Matched Pairs")
plt.annotate(f"p = {wilcoxon_result.pvalue:.5f}", xy=(0.5, max(treated_risk_scores)*0.9), ha='center')
plt.show()
