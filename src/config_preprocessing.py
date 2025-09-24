

# Normalize and validate config values coming from YAML (strings, numbers, booleans)
def _coerce_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)

def normalize_config(cfg: dict) -> dict:
    out = dict(cfg or {})
    int_keys = ["search_range", "height", "width", "batch_size", "num_epochs", "epoch_steps", "save_freq", "epoch_evals"]
    float_keys = ["lr", "min_lr", "decay"]
    bool_keys = ["shuffle", "log_tensorboard", "augment"]

    for k in int_keys:
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except Exception:
                raise ValueError(f"Config key '{k}' must be an integer. Got: {out[k]!r}")

    for k in float_keys:
        if k in out and out[k] is not None:
            try:
                out[k] = float(out[k])
            except Exception:
                raise ValueError(f"Config key '{k}' must be a float. Got: {out[k]!r}")

    for k in bool_keys:
        if k in out:
            out[k] = _coerce_bool(out[k])

    # Normalize empty strings to None for path-like fields
    for k in ["train_left_dir", "train_right_dir", "train_disp_dir", "val_left_dir", "val_right_dir", "val_disp_dir", "weights_path", "output_dir"]:
        if k in out and isinstance(out[k], str) and out[k].strip() == "":
            out[k] = None

    # Provide sensible defaults if missing
    if "output_dir" not in out or out.get("output_dir") is None:
        out["output_dir"] = "./model_weights"

    return out