from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


def transform_behaviors_to_coldstart(behaviors, news, cold_start_category):
    # filter histories to news-only
    behaviors_histories_news_only = (
        behaviors[["slateid", "history"]]
        .assign(history=lambda x: x["history"].fillna("").str.split())
        .explode("history")
        .reset_index(drop=True)
        .reset_index(drop=False)  # trick to preserve original ordering through merge
        .merge(
            news[news.category == cold_start_category][["newsid"]],
            left_on="history",
            right_on="newsid",
            how="inner",
            sort=False,
        )
        .sort_values("index")
        .drop("index", axis=1)  # restore original ordering
        .groupby("slateid", as_index=False)
        .agg({"history": list})
        .assign(history=lambda x: x["history"].str.join(" "))
    )

    # filter impressions to news-only
    behaviors_impressions_news_only = (
        behaviors[["slateid", "impressions"]]
        .assign(impressions=lambda x: x["impressions"].fillna("").str.split())
        .explode("impressions")
        .assign(impression_id=lambda x: x["impressions"].str.split("-").str[0])
        .reset_index(drop=True)
        .reset_index(drop=False)  # trick to preserve original ordering through merge
        .merge(
            news[news.category == cold_start_category][["newsid"]],
            left_on="impression_id",
            right_on="newsid",
            how="inner",
            sort=False,
        )
        .sort_values("index")
        .drop("index", axis=1)  # restore original ordering
        .groupby("slateid", as_index=False)
        .agg({"impressions": list})
        .assign(impressions=lambda x: x["impressions"].str.join(" "))
    )

    # generate categories and subcategories side-data from user histories
    behaviors_categories = (
        behaviors[["slateid", "history"]]
        .assign(history=lambda x: x["history"].fillna("").str.split())
        .explode("history")
        .reset_index(drop=True)
        .reset_index(drop=False)  # trick to preserve original ordering through merge
        .merge(
            news[["newsid", "category", "subcategory", "title"]],
            left_on="history",
            right_on="newsid",
        )
        .sort_values("index")
        .drop("index", axis=1)  # restore original ordering
        .groupby("slateid", as_index=False)
        .agg({"category": list, "subcategory": list, "title": list})
        .assign(category=lambda x: x["category"].str.join(" "))
        .assign(subcategory=lambda x: x["subcategory"].str.join(" "))
        .assign(title=lambda x: x["title"].str.join(";"))
        .rename(
            columns={
                "category": "history_all_categories",
                "subcategory": "history_all_subcategories",
                "title": "history_all_title"
            }
        )
    )

    # join all data together
    return (
        behaviors.rename(columns={"history": "history_all"})
        .rename(columns={"impressions": "impressions_all"})
        .merge(behaviors_impressions_news_only, on="slateid", how="inner")
        .merge(behaviors_histories_news_only, on="slateid", how="left")
        # ensure that newly created cold-start users have valid history
        .assign(history=lambda x: x.history.fillna(" "))
        .merge(behaviors_categories, on="slateid", how="left")
        # remove original cold-start users
        .dropna()
    )


def prepare_dataset_rf(df,
                       include_history_len=False,
                       include_categories=False,
                       include_subcategories=False,
                       enc=None,
                       enc_cat=None,
                       enc_subcat=None,
                       categories_vocabulary=None,
                       subcategories_vocabulary=None,
                       news_vocabulary=None
    ):
    df_e = (
        df[["userid", "slateid", "history", "impressions", "history_all_categories", "history_all_subcategories"]].assign(
            impression_arr=lambda x: x.impressions.map(
                lambda ii: [i.split("-") for i in ii.split(" ")])
        ).explode("impression_arr")
        .assign(impression=lambda x: x.impression_arr.map(lambda xx: xx[0]))
        .assign(click=lambda x: x.impression_arr.map(lambda xx: int(xx[1])))
    )

    if enc is None:
        enc = CountVectorizer(
            binary=True, vocabulary=news_vocabulary, lowercase=False)

    histories = enc.transform(df_e.history)
    impressions = enc.transform(df_e.impression)

    feats = [histories, impressions]

    if include_history_len:
        feats.append(df_e["history"].map(len).to_numpy().reshape(-1, 1))

    if include_categories:
        if enc_cat is None:
            enc_cat = CountVectorizer(vocabulary=categories_vocabulary)

        cats_transformed = enc_cat.transform(df_e["history_all_categories"].tolist())
        feats.append(cats_transformed)

    if include_subcategories:
        if enc_subcat is None:
            enc_subcat = CountVectorizer(vocabulary=subcategories_vocabulary)

        subcats_transformed = enc_subcat.transform(df_e["history_all_subcategories"].tolist())
        feats.append(subcats_transformed)

    X = hstack(feats)

    y = df_e.click

    return (X, y, enc, enc_cat, enc_subcat, df_e)
