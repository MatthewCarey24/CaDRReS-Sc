import pandas as pd
from scipy import stats
import numpy as np


def calculate_spearman(obs_df, pred_df, sample_list, drug_list, prefix=''):
    """
    Calculate Spearman correlation per sample and per drug.

    Parameters:
    obs_df (pd.DataFrame): Observed values dataframe with samples as rows and drugs as columns.
    pred_df (pd.DataFrame): Predicted values dataframe with samples as rows and drugs as columns.
    sample_list (list): List of samples to include in the evaluation.
    drug_list (list): List of drugs to include in the evaluation.
    prefix (str): Prefix for the output column names.

    Returns:
    pd.DataFrame, pd.DataFrame: Per-sample and per-drug Spearman correlation results.
    """
    obs_df = obs_df.loc[sample_list, drug_list]
    pred_df = pred_df.loc[sample_list, drug_list]

    # Per-sample correlation
    results = []
    for s in sample_list:
        x = obs_df.loc[s].values
        y = pred_df.loc[s].values
        scor, pval = stats.spearmanr(x, y)
        results.append([s, scor, pval])
    per_sample_df = pd.DataFrame(results, columns=['sample', f'{prefix}scor', f'{prefix}pval']).set_index('sample')

    # Per-drug correlation
    results = []
    for d in drug_list:
        x = obs_df[d].values
        y = pred_df[d].values
        scor, pval = stats.spearmanr(x, y)
        results.append([d, scor, pval])
    per_drug_df = pd.DataFrame(results, columns=['drug', f'{prefix}scor', f'{prefix}pval']).set_index('drug')

    return per_sample_df, per_drug_df


def calculate_spearman_multi_pred(obs_df, pred_df_dict, sample_list, drug_list):
    """
    Calculate Spearman correlation for multiple predictions.

    Parameters:
    obs_df (pd.DataFrame): Observed values dataframe with samples as rows and drugs as columns.
    pred_df_dict (dict): Dictionary of predicted dataframes {name: dataframe}.
    sample_list (list): List of samples to include in the evaluation.
    drug_list (list): List of drugs to include in the evaluation.

    Returns:
    pd.DataFrame, pd.DataFrame: Aggregated per-sample and per-drug Spearman correlation results.
    """
    per_sample_df_list = []
    per_drug_df_list = []

    for pred_name, pred_df in pred_df_dict.items():
        per_sample_df, per_drug_df = calculate_spearman(
            obs_df, pred_df, sample_list, drug_list, prefix=f"{pred_name}_"
        )
        per_sample_df_list.append(per_sample_df)
        per_drug_df_list.append(per_drug_df)

    all_per_sample_df = pd.concat(per_sample_df_list, axis=1)
    all_per_drug_df = pd.concat(per_drug_df_list, axis=1)

    return all_per_sample_df, all_per_drug_df


def calculate_ndcg(obs_df, pred_df, k=10):
    """
    Calculate NDCG (Normalized Discounted Cumulative Gain) per sample.

    Parameters:
    obs_df (pd.DataFrame): Observed values dataframe with samples as rows and drugs as columns.
    pred_df (pd.DataFrame): Predicted values dataframe with samples as rows and drugs as columns.
    k (int): Rank cutoff for NDCG calculation.

    Returns:
    pd.DataFrame: Per-sample NDCG results.
    """
    ndcg_results = []

    for sample in obs_df.index:
        true_values = obs_df.loc[sample].values
        pred_values = pred_df.loc[sample].values

        # Sort indices by predicted values (descending)
        sorted_indices = np.argsort(-pred_values)

        # Calculate DCG
        dcg = 0.0
        for i in range(min(k, len(sorted_indices))):
            rel = true_values[sorted_indices[i]]
            dcg += (2**rel - 1) / np.log2(i + 2)

        # Calculate ideal DCG
        sorted_true_indices = np.argsort(-true_values)
        idcg = 0.0
        for i in range(min(k, len(sorted_true_indices))):
            rel = true_values[sorted_true_indices[i]]
            idcg += (2**rel - 1) / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_results.append([sample, ndcg])

    ndcg_df = pd.DataFrame(ndcg_results, columns=['sample', f'ndcg@{k}']).set_index('sample')

    return ndcg_df

# Example function usage:
# per_sample_df, per_drug_df = calculate_spearman(obs_df, pred_df, sample_list, drug_list)
# all_per_sample_df, all_per_drug_df = calculate_spearman_multi_pred(obs_df, pred_df_dict, sample_list, drug_list)
# ndcg_df = calculate_ndcg(obs_df, pred_df, k=10)
