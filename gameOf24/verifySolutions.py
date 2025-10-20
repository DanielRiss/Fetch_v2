import json
from evaluateAnswer import evaluate_answer

# Read JSON
with open('go24-full-with-solutions.json', 'r') as f:
    data = json.load(f)

total = 0
valid = 0
invalid = 0

for entry in data:
    puzzle = entry['Puzzles']
    solutions_str = entry.get('Solution(s)', '')

    # Split multiple solutions by comma
    solutions = [s.strip() for s in solutions_str.split(',') if s.strip()]
    # print(solutions)
    for solution in solutions:
        total += 1
        if evaluate_answer(puzzle, solution):
            valid += 1
        else:
            invalid += 1
            print(f"INVALID: {puzzle} -> {solution}")

print(f"\nTotal solutions: {total}")
print(f"Valid: {valid}")
print(f"Invalid: {invalid}")
