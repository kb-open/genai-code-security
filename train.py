# Fine-Tuning of the Model

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import random
import subprocess
import re
from utils import log_results

def load_security_dataset():
    return load_dataset("./data/owasp/sard_secure_coding")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["code"], padding="max_length", truncation=True)

def train_model():
    # Step 1: Load Security-Audited Dataset (e.g., OWASP Benchmark, SARD)
    dataset = load_security_dataset()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
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
    model.save_pretrained("./models/secure_coding_model")
    tokenizer.save_pretrained("./models/secure_coding_model")
    
    log_results("Model training completed successfully.")