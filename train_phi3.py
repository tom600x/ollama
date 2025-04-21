#!/usr/bin/env python3
"""
Phi-3 model training script using Ollama
This script prepares and trains a Phi-3 model using the dataset.json file
"""

import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path

def check_ollama_installed():
    """Check if Ollama is installed on the system"""
    try:
        subprocess.run(["ollama", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Ollama is installed")
        return True
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def install_ollama():
    """Install Ollama if not already installed"""
    print("Installing Ollama...")
    try:
        # Run the official Ollama install script
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        print("✅ Ollama installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Ollama")
        return False

def validate_dataset(dataset_path):
    """Validate the dataset file exists and has correct format"""
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Dataset should be a JSON array")
            return False
        
        print(f"✅ Dataset validated: {len(data)} training examples found")
        return True
    except json.JSONDecodeError:
        print("❌ Invalid JSON format in dataset file")
        return False

def create_modelfile(model_name, dataset_path):
    """Create an Ollama Modelfile for training"""
    modelfile_content = f"""
FROM {model_name}

# System prompt that sets the context and behavior for the model
SYSTEM """

    # Add system prompt specific to the conversion task
    modelfile_content += """You are an AI assistant specialized in converting PL/SQL code to C# LINQ. 
Provide clear, accurate, and efficient conversion of database queries and operations."""

    # Add dataset reference for training
    modelfile_content += f"""

# Training data
DATASET {dataset_path}
"""

    # Write the Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    print("✅ Created Modelfile for training")
    return True

def train_model(output_model_name):
    """Train the model using Ollama"""
    print(f"Training model {output_model_name}...")
    
    try:
        # Create the model with the Modelfile
        subprocess.run(["ollama", "create", output_model_name, "-f", "Modelfile"], check=True)
        print(f"✅ Model {output_model_name} created and trained successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train Phi-3 model with Ollama")
    parser.add_argument("--model-path", default="/home/TomAdmin/phi-3-mini-128k-instruct", 
                       help="Path to the Phi-3 model")
    parser.add_argument("--dataset", default="dataset.json", 
                       help="Path to the dataset JSON file")
    parser.add_argument("--output-name", default="phi3-sql-to-linq", 
                       help="Name for the trained model in Ollama")
    parser.add_argument("--format", choices=["ggml", "gguf"], default="gguf",
                       help="Model format to use (ggml or gguf)")
    args = parser.parse_args()

    print("=" * 50)
    print("Phi-3 Model Training with Ollama")
    print("=" * 50)

    # Ensure absolute paths
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    dataset_path = os.path.abspath(os.path.expanduser(args.dataset))
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        if not install_ollama():
            return
    
    # Validate the dataset
    if not validate_dataset(dataset_path):
        return
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return
    
    # Create Ollama Modelfile
    model_name = f"phi3-mini-instruct:{args.format}"
    if not create_modelfile(model_name, dataset_path):
        return
    
    # Train the model
    if not train_model(args.output_name):
        return
    
    print("\n" + "=" * 50)
    print(f"Training complete! You can use your model with: ollama run {args.output_name}")
    print("=" * 50)

if __name__ == "__main__":
    main()
