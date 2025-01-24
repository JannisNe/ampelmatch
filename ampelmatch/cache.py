cache_dir = "ampelmatch_cache"


def model_hash(models, kwds):
    import hashlib
    import json
    strings = [json.dumps(m, sort_keys=True) for m in models]
    s = "".join(strings).encode()
    return hashlib.sha256(s).hexdigest()
