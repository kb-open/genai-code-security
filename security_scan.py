# Automated Security Scanning (CodeQL & SonarQube)

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import random
import subprocess
import re
from utils import log_results

def run_security_scan(file_path):
    try:
        subprocess.run(["codeql", "database", "analyze", file_path, "--format=json"], check=True)
        subprocess.run(["sonar-scanner", f"-Dsonar.projectBaseDir={file_path}"], check=True)
        log_results("Security scan completed successfully.")
    except subprocess.CalledProcessError as e:
        log_results(f"Security scan failed: {e}")

def scan_code():
    run_security_scan("./models/secure_coding_model")