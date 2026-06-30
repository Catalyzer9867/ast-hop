import sys
from ast_hop.train_code import run_code_training

def main():
    print("=== Launching 1 Million Parameter Codebase Training Profile ===")
    
    # Run the training loop for 40 epochs
    model, metrics = run_code_training(epochs=40)
    
    print("\n=== Validation Results ===")
    if metrics["accuracy"] >= 0.90:
        print("Success: Validation Accuracy achieved target (>= 90%)")
    else:
        print("Note: Validation Accuracy is below target (90%) - policy still optimizing")
        
    if metrics["read_ratio"] < 0.40:
        print(f"Success: Read ratio achieved skimming target (< 40%). Best read: {metrics['read_ratio']:.2%}")
    else:
        print(f"Note: Skimming target (< 40%) not met. Best read: {metrics['read_ratio']:.2%}")
        
    print("\n1M parameter codebase training run completed successfully.")

if __name__ == "__main__":
    main()
