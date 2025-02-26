import os

def log_results(log_message):
    with open("./results/evaluation_logs.txt", "a") as log_file:
        log_file.write(log_message + "\n")