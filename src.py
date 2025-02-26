import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import random
import subprocess
import re

# Step 1: Load Security-Audited Dataset (e.g., OWASP Benchmark, SARD)
dataset = load_dataset("owasp/sard_secure_coding")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def tokenize_function(examples):
    return tokenizer(examples["code"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 2: Fine-Tune the Model on Secure Coding Examples
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./secure_coding_model")

tokenizer.save_pretrained("./secure_coding_model")

# Step 3: Reinforcement Learning with Human Feedback (RLHF)
def security_penalty_score(code_snippet):
    penalties = {
        "eval_exec": re.search(r"\beval\(|\bexec\(", code_snippet),
        "subprocess_unsanitized": re.search(r"\bsubprocess\.run\(|\bsubprocess\.Popen\(", code_snippet),
        "hardcoded_secrets": re.search(r"(password|secret|api_key)\s*=\s*["'].*["']", code_snippet, re.IGNORECASE),
        "weak_random": re.search(r"\brandom\.(randint|random|choice)\(", code_snippet),
        "unsafe_file_access": re.search(r"\bopen\(input\(", code_snippet),
        "insecure_deserialization": re.search(r"\bpickle\.load\(", code_snippet),
        "sql_injection": re.search(r"SELECT \* FROM .*WHERE.*['"].*['"]", code_snippet, re.IGNORECASE),
        "hardcoded_ips": re.search(r"\b(\d{1,3}\.){3}\d{1,3}\b", code_snippet),
    }
    
    score = 1  # Default positive score
    for key, issue in penalties.items():
        if issue:
            score -= 1  # Penalize for each detected issue
    
    return score

def apply_rlhf(model_output):
    score = security_penalty_score(model_output)
    return model_output, score

# Step 4: Automated Security Scanning (CodeQL & SonarQube)
def run_security_scan(file_path):
    try:
        subprocess.run(["codeql", "database", "analyze", file_path, "--format=json"], check=True)
        subprocess.run(["sonar-scanner", f"-Dsonar.projectBaseDir={file_path}"], check=True)
        print("Security scan completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Security scan failed:", e)

# Example usage
sample_code = "input = eval(input('Enter command: '))"
secure_code, score = apply_rlhf(sample_code)
print("RLHF Score:", score)
run_security_scan("./secure_coding_model")
