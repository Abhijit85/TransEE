# Code_ccmurp

Isolated launcher directory for CCMuRP-family experiments.

## Purpose
- Keep CCMuRP runs separated from generic `Code/` entrypoints.
- Use `CODE_PATH=Code_ccmurp` in env files to run through this launcher.

## Notes
- Core implementation remains in `Code/model.py` (model class: `CCMuRP`).
- This launcher intentionally reuses shared training pipeline from `Code/driver.py`.

## Run
```bash
ENV_FILE=.env.ccmurp_standalone bash run.sh train CCMuRP wn18rr 7 wn18rr_ccmurp_v1
```
