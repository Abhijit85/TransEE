# Relate-V Branch Layout

This branch contains a dedicated `RelateV` single-model copy based on `Code_vanilla_relate`.

## Added paths
- `Code_relate_v/`: model/training code copy for `RelateV`
- `.env.relate_v`: default environment for `RelateV`
- `scripts/run_relate_v.sh`: launcher using `Code_relate_v`
- `data_relate_v/`: copied datasets and type maps for this branch

## Run
```bash
bash scripts/run_relate_v.sh train RelateV wn18rr
```

By default, `RelateV` applies an enhanced profile (Lookahead + EMA + grad clip + selected RelatE feature toggles). Set `RELATEV_APPLY_PROFILE=0` to disable profile defaults while keeping the `RelateV` alias.
