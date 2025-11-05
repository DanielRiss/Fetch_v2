import random
import pandas as pd

def load_gpqa_csv(path):
    df = pd.read_csv(path)
    # Remove rows with missing data
    df = df.dropna(subset=["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"])
    # Build input/label pairs
    rows = []
    for _, row in df.iterrows():
        answers = [
            ("A", row["Correct Answer"], 1),
            ("B", row["Incorrect Answer 1"], 0),
            ("C", row["Incorrect Answer 2"], 0),
            ("D", row["Incorrect Answer 3"], 0),
        ]
        random.shuffle(answers)  # Shuffle choices per row
        input_text = f"Question: {row['Question']}\n"
        # Re-assign A, B, C, D after shuffle
        for i in range(len(answers)):
            answers[i] = (chr(ord('A') + i), answers[i][1], answers[i][2])
        # Add answer options to input text
        for letter, ans, _ in answers:
            input_text += f"{letter}. {ans}\n"
        # The label is a list: 1 for correct, 0 for incorrect, in the order presented
        label = [lbl for _, _, lbl in answers]
        rows.append({"text": input_text.strip(), "label": label})
    return rows

if __name__ == "__main__":
    csv_path = "gpqa/dataset/gpqa_eval_split.csv"
    data = load_gpqa_csv(csv_path)
    print(data[:2])  # Print first 2 processed rows as a sample