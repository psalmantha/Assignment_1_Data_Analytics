import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
import pulp

# load data from csv file
df = pd.read_csv('brsm_dataset.csv')

# define the covariates to balance
covariates = ['age', 'income', 'risk_score']

# function to compute the mahalanobis distance between treated and control units
def calculate_mahalanobis(treated_vec, control_vec, cov_matrix):
    diff = treated_vec - control_vec
    inv_cov = np.linalg.pinv(cov_matrix)  # use pseudoinverse for numerical stability
    return np.sqrt(diff.dot(inv_cov).dot(diff))

# function for balanced risk set matching
def balanced_risk_set_matching(df):
    # check if all necessary columns are present in the dataframe
    required_cols = covariates + ['unit_id', 'treatment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
    
    # work on a copy of the dataframe to preserve the original
    df = df.copy()
    
    # step 1: create tertile-based groupings for each covariate (age, income, risk_score)
    for col in covariates:
        values = pd.to_numeric(df[col], errors='coerce')  # convert to numeric and handle non-numeric values
        tertiles = np.percentile(values, [33.33, 66.67])  # calculate tertiles
        df[f'{col}_low'] = (values <= tertiles[0]).astype(int)
        df[f'{col}_med'] = ((values > tertiles[0]) & (values <= tertiles[1])).astype(int)
        df[f'{col}_high'] = (values > tertiles[1]).astype(int)
    
    # step 2: separate treated and control units
    treated = df[df['treatment'] == 1].sort_values('unit_id')
    controls = df[df['treatment'] == 0].copy()

    # step 3: generate all possible edges (matches between treated and control units)
    edges = [(t_row['unit_id'], c_row['unit_id']) for _, t_row in treated.iterrows() for _, c_row in controls.iterrows()]

    # step 4: compute mahalanobis distances for each edge (treated-control pair)
    cov_matrix = np.cov(df[covariates].values.T)  # Calculate the covariance matrix of covariates
    distances = {}
    for (t, c) in edges:
        t_vec = df.loc[df['unit_id'] == t, covariates].values.ravel()
        c_vec = df.loc[df['unit_id'] == c, covariates].values.ravel()
        distances[(t, c)] = calculate_mahalanobis(t_vec, c_vec, cov_matrix)

    # step 5: set up an optimization problem to minimize total mahalanobis distance
    prob = pulp.LpProblem("BalancedMatching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("pair", edges, cat='Binary')
    
    # step 6: define the objective function (minimizing the weighted sum of distances)
    prob += pulp.lpSum([distances[e] * x[e] for e in edges])
    
    # step 7: add constraints for one-to-one matching between treated and control units
    control_ids = controls['unit_id'].unique()
    for t in treated['unit_id']:
        prob += pulp.lpSum([x[(t, c)] for c in control_ids if (t, c) in edges]) == 1
    
    for c in control_ids:
        prob += pulp.lpSum([x[(t, c)] for t in treated['unit_id'] if (t, c) in edges]) <= 1
    
    # step 8: solve the optimization problem using CBC solver
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # step 9: extract the matched pairs based on the solution
    matched_pairs = [(t, c) for (t, c) in edges if x[(t, c)].value() > 0.5]
    
    return matched_pairs

# perform the balanced risk set matching
matched_pairs = balanced_risk_set_matching(df)

# convert matched pairs into a DataFrame and save it as csv
matched_pairs_df = pd.DataFrame(matched_pairs, columns=['treated_unit_id', 'control_unit_id'])
matched_pairs_df.to_csv('matched_pairs.csv', index=False)

print("Matched pairs have been saved to 'matched_pairs.csv'.")
