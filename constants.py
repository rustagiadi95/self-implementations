import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
TRANSFORMERS_DIR = os.path.join(ROOT_DIR, "transformers_fmt")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

