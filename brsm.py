import pandas as pd
import numpy as np
import pulp
from scipy.spatial.distance import mahalanobis
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("brsm_dataset.csv")

required_columns = {"unit_id", "treatment", "age", "income", "risk_score", "health_status", "entry_time", "treatment_time"}
if not required_columns.issubset(df.columns):
    raise ValueError("Dataset is missing required columns.")

# convert categorical column 'health_status' to numeric
df["health_status"] = df["health_status"].astype("category").cat.codes

# separate treated and control groups
treated = df[df["treatment"] == 1].reset_index(drop=True)
control = df[df["treatment"] == 0].reset_index(drop=True)

# use all numeric covariates for mahalanobis distance calculation
cov_features = ["age", "income", "health_status"]
cov_matrix = np.cov(control[cov_features].T.astype(float))
cov_matrix_inv = np.linalg.pinv(cov_matrix)

# compute mahalanobis distance matrix
distance_matrix = np.full((len(treated), len(control)), np.inf)

for i, (_, treat) in enumerate(treated.iterrows()):
    for j, (_, ctrl) in enumerate(control.iterrows()):
        # ensure control entered study before treated unit was treated
        if treat["treatment_time"] >= ctrl["entry_time"]:
            diff = treat[cov_features].values - ctrl[cov_features].values
            distance_matrix[i, j] = mahalanobis(diff, np.zeros_like(diff), cov_matrix_inv)

# handle unmatched cases by replacing 'inf' with a large value
if np.all(np.isinf(distance_matrix)):
    raise ValueError("No valid matches found. Adjust temporal constraints or dataset.")

distance_matrix[np.isinf(distance_matrix)] = 1e6  # large number for solver compatibility

# linear programming for optimal matching
problem = pulp.LpProblem("Balanced_Risk_Set_Matching", pulp.LpMinimize)
vars_dict = {(i, j): pulp.LpVariable(f"match_{i}_{j}", cat=pulp.LpBinary)
             for i in range(len(treated)) for j in range(len(control))}

# objective function: minimize mahalanobis distance
problem += pulp.lpSum(vars_dict[i, j] * distance_matrix[i, j] for i in range(len(treated)) for j in range(len(control)))

# constraint: each treated unit gets exactly one match
for i in range(len(treated)):
    problem += pulp.lpSum(vars_dict[i, j] for j in range(len(control))) == 1

# constraint: each control unit can be matched up to max_matches times
max_matches = min(5, len(treated))  
for j in range(len(control)):
    problem += pulp.lpSum(vars_dict[i, j] for i in range(len(treated))) <= max_matches

problem.solve()

# extract matched pairs
matched_pairs = [(treated.iloc[i]["unit_id"], control.iloc[j]["unit_id"]) 
                 for (i, j), var in vars_dict.items() if pulp.value(var) == 1]

# handle no matched pairs case
if len(matched_pairs) == 0:
    raise ValueError("No matched pairs found. Check dataset or matching constraints.")

# print(f"Matched pairs found: {len(matched_pairs)}")

# save matched pairs
matched_df = pd.DataFrame(matched_pairs, columns=["treated_id", "control_id"])
valid_treated = df[df["unit_id"].isin(matched_df["treated_id"])]
valid_control = df[df["unit_id"].isin(matched_df["control_id"])]

# ensure equal number of treated and control units for wilcoxon test
min_size = min(len(valid_treated), len(valid_control))
valid_treated = valid_treated.iloc[:min_size]
valid_control = valid_control.iloc[:min_size]

# extract risk scores
treated_risk_scores = valid_treated["risk_score"].values
control_risk_scores = valid_control["risk_score"].values

# perform wilcoxon signed-rank test
wilcoxon_result = wilcoxon(treated_risk_scores, control_risk_scores)
print(f"Wilcoxon test p-value: {wilcoxon_result.pvalue:.5f}")

# visualization
plt.figure(figsize=(8, 6))
sns.boxplot(data=[treated_risk_scores, control_risk_scores], palette=["#FF9999", "#99CCFF"])
plt.xticks([0, 1], ["Treated", "Control"])
plt.ylabel("Risk Score")
plt.title("Risk Score Distribution for Matched Pairs")
plt.annotate(f"p = {wilcoxon_result.pvalue:.5f}", xy=(0.5, max(treated_risk_scores) * 0.9), ha='center')
plt.show()
