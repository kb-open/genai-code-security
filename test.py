# Test the core functionalities

from train import train_model
from rlhf import apply_reinforcement_learning
from security_scan import scan_code

if __name__ == "__main__":
    train_model()
    apply_reinforcement_learning()
    scan_code()