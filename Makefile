generic_pmcc:
	uv run ./notebooks/generic_pmcc.py
	# uv run ./scripts/generic_pmcc.py

generic_pmcc_notebook:
	uv run -- marimo edit ./notebooks/generic_pmcc.py

get_data:
	uv run ./ml/get_data.py

train_mdl:
	uv run ./ml/train.py
