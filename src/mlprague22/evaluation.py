from collections import Counter
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def get_user_prediction(user_predictions, per_user_predicted_items=20):
    return np.argpartition(user_predictions, -per_user_predicted_items)[
        -per_user_predicted_items:
    ]


def sample_items(all_news, random_state, frac=0.10, stratify_on="subcategory"):
    return all_news.groupby(stratify_on, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=random_state)
    )["newsid"]


def sample_users(df, random_state, frac=0.01, quantiles=(0.2, 0.4, 0.6, 0.8, 1.0)):
    df["history_size"] = df.apply(lambda x: len(x.history.split()), axis=1)
    history_size_quantiles = [0] + df.history_size.quantile(quantiles).tolist()

    df["history_bin"] = pd.cut(df.history_size, history_size_quantiles)
    users = (
        df.drop_duplicates("userid")
        .groupby("history_bin", group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=random_state))
    )
    df.drop(columns=["history_bin", "history_size"])
    return users


def compute_binned_performance_metric(
    df, metric, metric_name, y_test_column, predictions_columns, predictions_columns_titles, bin_col="history_bin"
):
    history_bin_results = [
        df.groupby(bin_col)
        .apply(lambda df: metric(df[y_test_column].tolist(), df[predictions_column].tolist()))
        .to_frame(metric_name)
        for predictions_column in predictions_columns
    ]

    result = pd.concat(history_bin_results, axis=1)
    result.columns = [f"{metric_name} {column_title}" for column_title in predictions_columns_titles]

    return result


def compute_binned_qualitative_metric(
    df, metric, metric_name, predictions_columns, predictions_columns_titles, bin_col="history_bin"
):
    history_bin_results = [
        df.groupby(bin_col).apply(lambda df: metric(df[predictions_column].tolist())).to_frame(metric_name)
        for predictions_column in predictions_columns
    ]
    result = pd.concat(history_bin_results, axis=1)
    result.columns = [f"{metric_name} {column_title}" for column_title in predictions_columns_titles]

    return result


def flatten(t):
    return [item for sublist in t for item in sublist]


def perplexity(recommendations):
    rec_length = recommendations[0].shape[0] if recommendations else 0.0
    cntr = Counter(flatten(recommendations))
    seq = list(cntr.values())
    all_sum = np.sum(seq)
    Px = [float(x) / all_sum for x in seq if x > 0]
    PxLog = [px * np.log2(px) for px in Px]
    result = np.exp2(-np.sum(PxLog))
    return float(result) / rec_length


def novelty(recommendations):
    users_count = len(recommendations)
    popularity_cntr = Counter(flatten(recommendations))
    novelty = {item: -np.log2(item_popularity / users_count) for item, item_popularity in popularity_cntr.items()}
    rec_novelties = [np.mean([novelty[item] for item in rec]) for rec in recommendations]
    return np.mean(rec_novelties)


def create_history_bins(df):
    df["history_size"] = df.apply(lambda x: len(x.history.split()), axis=1)

    history_size_quantiles = df.history_size.quantile([0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    history_size_quantiles.insert(0, 0.0)
    history_bins = pd.cut(df.history_size, history_size_quantiles)
    df.drop(columns=["history_size"])
    return history_bins


def evaluate(
    sampled_users,
    test_df,
    test_prediction_columns,
    y_test_column,
    display_fn=None
):
    """
    - both dataframes should contain a history binning column, eg. 'history_bin'
    - sampled_users should contain 'predictions' and 'predictions_no_cats' for perplexity and novelty
    - test_df is a test dataset and should contain 'y_test', 'predictions_test', 'predictions_test_no_cats'
    """
    novelties = [novelty(sampled_users[column].tolist()) for column in test_prediction_columns]
    aucs = [metrics.roc_auc_score(test_df[y_test_column], test_df[column]) for column in test_prediction_columns]

    if display_fn:
        display_fn(
            pd.DataFrame(
                [
                    (f"novelty {column_title}", novelty)
                    for column_title, novelty in zip(test_prediction_columns, novelties)
                ]
                + [(f"AUC {column_title}", auc) for column_title, auc in zip(test_prediction_columns, aucs)],
                columns=["metric", "value"],
            )
        )

    nvl_binned = compute_binned_qualitative_metric(
        sampled_users,
        novelty,
        "novelty",
        test_prediction_columns,
        test_prediction_columns,
    )
    nvl_binned.plot.bar(figsize=(10, 10))
    plt.title("novelty per history size")

    auc_binned_test = compute_binned_performance_metric(
        test_df,
        metrics.roc_auc_score,
        "AUC",
        y_test_column,
        test_prediction_columns,
        test_prediction_columns,
    )
    auc_binned_test.plot.bar(figsize=(10, 10))
    plt.title("AUC per history size")

