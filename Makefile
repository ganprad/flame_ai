# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions
conda-gpu:
	conda env update --prune -f env_configs/environment.yml

# Install without CUDA
conda-cpu:
	conda env update --prune -f env_configs/environment_cpu.yml

# Compile and install exact pip packages
pip-tools:
	pip-compile requirements/prod.in && pip-compile requirements/dev.in && pip-compile requirements/lint.in
	pip-sync requirements/prod.txt  requirements/dev.txt  requirements/lint.txt
