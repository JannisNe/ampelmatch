import hashlib
import json

from pandas.util import hash_pandas_object

cache_dir = "ampelmatch_cache"


def model_hash(models, kwds):
    dicts = [m.model_dump() for m in models]
    for k, v in kwds.items():
        if "config" in k:
            dicts.append(v.model_dump())
    strings = [json.dumps(d, sort_keys=True) for d in dicts]
    s = "".join(strings).encode()
    return hashlib.sha256(s).hexdigest()


def dataframe_hash(df):
    h1 = hashlib.sha256(hash_pandas_object(df).values).hexdigest()
    h2 = hashlib.sha256(df.columns.to_numpy().astype(str)).hexdigest()
    h3 = hashlib.sha256(df.index.to_numpy()).hexdigest()
    return hashlib.sha256((h1 + h2 + h3).encode()).hexdigest()


def compute_density_hash(args, kwargs):
    if len(args) > 0:
        raise ValueError("args not supported")
    dfs = kwargs.pop("data")
    nside = kwargs.pop("nside")
    if len(kwargs) > 0:
        raise ValueError(f"unexpected kwargs: {list(kwargs.keys())}!")
    df_hash = hashlib.sha256(
        str(tuple(dataframe_hash(df) for df in dfs)).encode()
    ).hexdigest()
    nside_hash = hashlib.sha256(str(nside).encode()).hexdigest()
    return hashlib.sha256((df_hash + nside_hash).encode()).hexdigest()
