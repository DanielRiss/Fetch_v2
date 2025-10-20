#!/usr/bin/env python3
"""
Prepare training data for Game of 24 models.

Creates two types of datasets:
1. Policy training data
2. Verifier training data (with labels for scoring)
"""

import csv
import json
import random
from pathlib import Path
from evaluateAnswer import evaluate_answer

# Configuration
INPUT_CSV = Path(__file__).parent / "go24-full-with-solutions.csv"
OUTPUT_POLICY = Path(__file__).parent / "go24_policy_data.json"
OUTPUT_VERIFIER = Path(__file__).parent / "go24_verifier_data.json"
RANDOM_SEED = 42

# Helper prompt (simplified - removed "step-by-step" instruction)
HELPER_PROMPT = "You are playing the Game of 24. Given 4 numbers, you may use addition, subtraction, multiplication, and division to combine them to reach the target number of 24. You must use each input number exactly once and can use parentheses to define the order of operations.\n\n"

random.seed(RANDOM_SEED)

def load_puzzles_from_csv(csv_path):
    # Load puzzles and solutions from CSV
    puzzles = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle = row['Puzzles'].strip()
            solutions_str = row['Solution(s)'].strip()

            # Parse multiple solutions
            solutions = [s.strip() for s in solutions_str.split(',') if s.strip()]

            puzzles.append({
                'puzzle': puzzle,
                'solutions': solutions,
                'rank': int(row['Rank'])
            })

    return puzzles

def create_policy_data(puzzles):
    # Create policy training data
    examples = []

    for puzzle_data in puzzles:
        puzzle = puzzle_data['puzzle']
        # Use all solutions for every puzzle
        for solution in puzzle_data['solutions']:
            examples.append({
                'question': f"{HELPER_PROMPT}Question: {puzzle}",
                'answer': solution
            })

    return examples

def generate_negative_examples(puzzle_data, all_puzzles):
    # Generate negative training examples (incorrect solutions)
    examples = []
    puzzle = puzzle_data['puzzle']
    puzzle_nums = puzzle.split()

    # Strat 1: Use solutions from other puzzles (wrong numbers)
    other_puzzle = random.choice(all_puzzles)
    if other_puzzle['puzzle'] != puzzle and other_puzzle['solutions']:
        wrong_solution = other_puzzle['solutions'][0]
        if not evaluate_answer(puzzle, wrong_solution):
            examples.append({
                'text': f"{HELPER_PROMPT}Question: {puzzle}\nAnswer: {wrong_solution}",
                'label': -1
            })

    # Strat 2: Incomplete expressions
    if puzzle_data['solutions']:
        correct = puzzle_data['solutions'][0]
        if len(correct) > 3:
            for cutoff in [3, 5, 7]:
                if cutoff < len(correct):
                    incomplete = correct[:cutoff]
                    if any(op in incomplete for op in ['+', '-', '*', '/']):
                        if not evaluate_answer(puzzle, incomplete):
                            examples.append({
                                'text': f"{HELPER_PROMPT}Question: {puzzle}\nAnswer: {incomplete}",
                                'label': -1
                            })
                            break

    # Strat 3: Wrong operations with correct numbers
    if len(puzzle_nums) == 4:
        nums = puzzle_nums.copy()
        ops = ['+', '-', '*', '/']

        for _ in range(2):
            random.shuffle(nums)
            random_ops = random.choices(ops, k=3)
            wrong_expr = f"{nums[0]}{random_ops[0]}{nums[1]}{random_ops[1]}{nums[2]}{random_ops[2]}{nums[3]}"

            if not evaluate_answer(puzzle, wrong_expr):
                examples.append({
                    'text': f"{HELPER_PROMPT}Question: {puzzle}\nAnswer: {wrong_expr}",
                    'label': -1
                })
                break

    return examples

def create_verifier_data(puzzles):
    # Create verifier training data (correct + incorrect with labels)
    examples = []

    print("Generating verifier training examples...")

    for i, puzzle_data in enumerate(puzzles):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(puzzles)} puzzles")

        puzzle = puzzle_data['puzzle']

        # Positive examples (correct solutions)
        for solution in puzzle_data['solutions']:
            examples.append({
                'text': f"{HELPER_PROMPT}Question: {puzzle}\nAnswer: {solution}",
                'label': 1
            })

        # Negative examples (incorrect solutions)
        neg_examples = generate_negative_examples(puzzle_data, puzzles)
        examples.extend(neg_examples)

    return examples

def save_json(data, filepath):
    # Save data as JSON (one object per line)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():

    # Load puzzles
    print(f"\nLoading puzzles from {INPUT_CSV}...")
    puzzles = load_puzzles_from_csv(INPUT_CSV)
    print(f"Loaded {len(puzzles)} puzzles")

    # Policy data
    print("\nCreating Policy Training Data")

    policy_data = create_policy_data(puzzles)
    print(f"Generated {len(policy_data)} examples")

    save_json(policy_data, OUTPUT_POLICY)
    print(f"Saved to: {OUTPUT_POLICY}")

    # Verifier data
    print("\nCreating Verifier Training Data")

    verifier_data = create_verifier_data(puzzles)

    pos_count = sum(1 for ex in verifier_data if ex['label'] == 1)
    neg_count = sum(1 for ex in verifier_data if ex['label'] == -1)
    print(f"\nGenerated {len(verifier_data)} total examples:")
    print(f"Positive (correct): {pos_count}")
    print(f"Negative (incorrect): {neg_count}")

    save_json(verifier_data, OUTPUT_VERIFIER)
    print(f"Saved to: {OUTPUT_VERIFIER}")

    # Summary
    print("\nGenerated files:")
    print(f"Policy:   {OUTPUT_POLICY}")
    print(f"Verifier: {OUTPUT_VERIFIER}")
    print("\nNote: Training scripts should handle train/test splitting!")

if __name__ == "__main__":
    main()
