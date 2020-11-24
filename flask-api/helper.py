import pandas as pd


def error_msg(attribute):
    return f"category <{attribute}> not supported", 400


def ord_bin_enc(df):
    df["education"] = df["education"].map({'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': -1, }).astype('int8')
    df["default"] = df["default"].apply(lambda x: 1 if x == 'no' else 0).astype('uint8')
    df["housing"] = df["housing"].apply(lambda x: 1 if x == 'yes' else 0).astype('uint8')
    df["loan"] = df["loan"].apply(lambda x: 1 if x == 'yes' else 0).astype('uint8')
    df["poutcome"] = df["poutcome"].apply(lambda x: 1 if x == 'success' else 0).astype('uint8')
    df["previous"] = df["previous"].apply(lambda x: x if x == 0 else 1).astype('uint8')
    return df


def ohe_encoding(df, encoder):
    cols = ['job', 'marital']
    X_enc = pd.DataFrame(encoder.transform(df[cols]), columns=encoder.get_feature_names()).reset_index(drop=True)
    df = df.drop(cols, axis=1).reset_index(drop=True)
    df = df.merge(X_enc, left_index=True, right_index=True)
    return df


def yeoj_transform(df, transformer):
    cols = ['age', 'day', 'balance', 'duration', 'campaign', 'pdays']
    df[cols] = transformer.transform(df[cols])
    return df
