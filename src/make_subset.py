from datasets import load_dataset
import pandas as pd

# MT-Bench has 80 questions (IDs 81-160), 10 per category, in this order
CATEGORIES = ["writing", "roleplay", "reasoning", "math",
              "coding", "extraction", "stem", "humanities"]

def get_category(qid):
    return CATEGORIES[(qid - 81) // 10]

df = ds = load_dataset("lmsys/mt_bench_human_judgments")["human"].to_pandas()
print("question_id range:", df["question_id"].min(), "-", df["question_id"].max())

# --- Look at one full example ---
ex = df.iloc[0]
print("\n=== QUESTION ===\n", ex["conversation_a"][0]["content"])
print("\n=== ANSWER A (", ex["model_a"], ") ===\n", ex["conversation_a"][1]["content"])
print("\n=== ANSWER B (", ex["model_b"], ") ===\n", ex["conversation_b"][1]["content"])
print("\n=== HUMAN WINNER:", ex["winner"], "===\n")

# --- Filtering ---
keys = ["question_id", "model_a", "model_b"]
print("All rows:", len(df))

df = df[df["turn"] == 1]
print("Turn 1 only:", len(df))

# keep pairs where every human judge gave the same verdict
df = df.groupby(keys).filter(lambda g: g["winner"].nunique() == 1)
df = df.drop_duplicates(keys)
print("Unique pairs with unanimous judges:", len(df))

df = df[df["winner"] != "tie"]
print("Without ties:", len(df))

df["category"] = df["question_id"].apply(get_category)
df = df[~df["category"].isin(["math", "coding"])]
print("Without math/coding:", len(df))

# extract the text
df["question"] = df["conversation_a"].apply(lambda c: c[0]["content"])
df["answer_a"] = df["conversation_a"].apply(lambda c: c[1]["content"])
df["answer_b"] = df["conversation_b"].apply(lambda c: c[1]["content"])

# balanced sample: shuffle, then take up to 25 per category
subset = df.sample(frac=1, random_state=42).groupby("category").head(25)

subset = subset[["question_id", "category", "model_a", "model_b",
                 "winner", "question", "answer_a", "answer_b"]]

print("\nFinal subset:", len(subset))
print(subset["category"].value_counts())
print(subset["winner"].value_counts())

subset.to_csv("data/subset_en.csv", index=False, encoding="utf-8")
print("\nSaved to data/subset_en.csv")