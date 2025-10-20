import csv

# Read solutions file
solutions_dict = {}
with open('go24-solutions_unordered_raw.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            # Find the puzzle (4 numbers) and all solutions
            puzzle = None
            solutions = []
            for part in parts:
                part = part.strip()
                # Check if it's a puzzle (4 numbers separated by spaces)
                tokens = part.split()
                if len(tokens) == 4 and all(t.isdigit() for t in tokens):
                    puzzle = part
                # Check if it's a solution (contains operators)
                elif any(op in part for op in ['×', '+', '-', '/', '(', ')']):
                    solutions.append(part.replace('×', '*'))

            if puzzle and solutions:
                solutions_dict[puzzle] = ', '.join(solutions)

# Read CSV and add solutions
rows = []
with open('go24-full.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    header.append('Solution(s)')
    rows.append(header)

    for row in reader:
        puzzle = row[1]  # Puzzles column

        # Linear search for solution
        solution = solutions_dict.get(puzzle, '')

        # Add solution with space after
        row_with_solution = row + [solution + ' ' if solution else '']
        rows.append(row_with_solution)

# Write output
with open('go24-full-with-solutions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"{len(solutions_dict)} solutions.")

