.PHONY: install assets run list clean ch02 ch03 ch04 ch05 ch06 ch07 ch08

install:   ## create .venv and install dependencies (needs uv)
	uv sync

assets:    ## download files needed by Chapter 4, Case Study 2
	uv run python scripts/fetch_assets.py

run:       ## run every case study
	uv run python scripts/run_all.py

list:
	uv run python scripts/run_all.py --list

ch02 ch03 ch04 ch05 ch06 ch07 ch08:
	uv run python scripts/run_all.py --chapter $(subst ch,,$@)

clean:     ## remove generated figures and caches
	find outputs -mindepth 1 ! -name .gitkeep -delete
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
