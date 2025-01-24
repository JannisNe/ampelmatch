cache_dir = "ampelmatch_cache"


def model_hash(models, kwds):
    import hashlib
    import json
    dicts = [m.model_dump() for m in models]
    for k, v in kwds.items():
        if "config" in k:
            dicts.append(v.model_dump())
    strings = [json.dumps(d, sort_keys=True) for d in dicts]
    s = "".join(strings).encode()
    return hashlib.sha256(s).hexdigest()
