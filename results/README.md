# Result directories

`synthetic/<case_id>/seed_XXXX.json.gz` and
`llm/<model>/<selector>/<nuisance>/<design>/seed_XXXX.json.gz` store one
compressed JSON record per seed: pointwise estimates and intervals, and all
Value-max, Improvement-LCB and Naive BoN selections for every grid prefix.
`llm/<model>/<selector>/score_spline/oracles.json` stores the exact targets.
Records are skipped when they already exist, so interrupted runs resume.

`python run_experiment.py aggregate {synthetic,llm}` turns complete records into
the cross-seed tables under `plotting/data/`. Response banks and score stores
are external inputs and are not stored here.
