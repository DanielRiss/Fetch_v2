# Game of 24 Policy Fine-tuning Trainer
# Fine-tunes Llama 3.1 8B on policy (propose) objective
# Run this SECOND after processing your data

import os
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

class Game24PolicyTrainer:
    """Fine-tune Llama 3.1 8B for Game of 24 POLICY (propose) objective using LoRA"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        data_path: str = None,
        output_dir: str = "ft_go24_policy"
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: HuggingFace model ID
            data_path: Path to processed JSONL data
            output_dir: Directory to save fine-tuned model
        """
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
    
    def setup(self):
        """Initialize tokenizer and model with LoRA"""
        rank = int(os.environ.get("RANK", 0))
        
        if rank == 0:
            print("\n[1/4] Loading tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set special tokens for Llama 3.1
        self.tokenizer.eos_token = "<|eot_id|>"
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        
        if rank == 0:
            print(f"✓ EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
            print(f"✓ PAD token ID: {self.tokenizer.pad_token_id}")
            print("\n[2/4] Loading base model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )
        
        if rank == 0:
            print(f"✓ Model loaded: {self.model.num_parameters():,} parameters")
            print("\n[3/4] Adding LoRA adapter...")
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"], 
        )
        self.model = get_peft_model(self.model, lora_config)
        
        if rank == 0:
            print(f"✓ LoRA adapter added")
            self.model.print_trainable_parameters()
    
    def format_prompts(self, example):
        """Format example for SFT training"""
        # Question already contains the prompt structure from data processor
        # Example: "Input: 1 1 4 6\nPossible next steps:"
        # Answer contains the steps
        return (
            f"{example['question']}\n"
            f"{example['answer']}"
        )
    
    def train(
        self,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
    ):
        """
        Fine-tune the policy model.
        
        Args:
            per_device_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate for LoRA
        """
        print("\n[4/4] Setting up training...")
        
        # Load dataset
        print(f"  Loading data from {self.data_path}...")
        dataset = load_dataset("json", data_files={"train": self.data_path}, split="train")
        print(f"  ✓ Loaded {len(dataset)} examples")

        # NEW: Show depth distribution
        depth_counts = {}
        for example in dataset.select(range(min(1000, len(dataset)))):  # Sample first 1000
            depth = example.get('depth', 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        print(f"\n  Data distribution (sampled):")
        for depth in sorted(depth_counts.keys()):
            pct = (depth_counts[depth] / len(list(depth_counts.values()))) * 100
            print(f"    Depth {depth} ({4-depth} numbers): {pct:.1f}%")
        print()
                
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.05,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            logging_first_step=True,
            save_strategy="epoch",
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Trainer
        print(f"  Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            data_collator=data_collator,
            args=training_args,
            formatting_func=self.format_prompts,
        )
        
        print(f"\nStarting training ({num_train_epochs} epochs, {len(dataset)} examples)...\n")
        trainer.train()
        
        # Save
        print(f"\n✓ Training completed")
        print(f"Saving model to {self.output_dir}...")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✓ Policy model and tokenizer saved to {self.output_dir}")
        
        return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B for Game of 24 Policy (Propose)"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to processed training data (from data_processor.py)"
    )
    parser.add_argument(
        "--output_dir",
        default="ft_go24_policy",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device (default: 2 for A100)"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4 for LoRA)"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model (default: Llama 3.1 8B Instruct)"
    )
    
    args = parser.parse_args()
    
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        print("=" * 80)
        print("GAME OF 24 POLICY FINE-TUNING (4-GPU Data Parallel)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Data: {args.data_path}")
        print(f"  Output: {args.output_dir}")
        print(f"  Model: {args.model}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Grad accum: {args.grad_accum}")
        print(f"  Effective batch (4 GPUs): {args.batch_size * args.grad_accum * 4}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
    
    # Train
    trainer = Game24PolicyTrainer(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    trainer.setup()
    trainer.train(
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
