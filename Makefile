TORCH_VENV_DIR = venv
DIRS = $(TORCH_VENV_DIR)
PIP_VENV = $(TORCH_VENV_DIR)/bin/pip3

torch-venv:
	python3 -m venv $(TORCH_VENV_DIR)
	$(PIP_VENV) install torch torchvision matplotlib pyqt5
	$(PIP_VENV) install jupyter

clean:
	rm -rf $(DIRS)
