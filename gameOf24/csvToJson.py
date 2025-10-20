import csv
import json

# Read CSV with solutions
rows = []
with open('go24-full-with-solutions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Write JSON
with open('go24-full-with-solutions.json', 'w') as f:
    json.dump(rows, f, indent=2)

print(f"Converted {len(rows)} puzzles to JSON")
print("Output: go24-full-with-solutions.json")
