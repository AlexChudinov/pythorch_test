TORCH_VENV_DIR = torch
DIRS = torch

torch-venv:
	python3 -m venv $(TORCH_VENV_DIR)
	. $(TORCH_VENV_DIR)/bin/activate
	pip3 install torch

clean:
	rm -rf $(DIRS)
