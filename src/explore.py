from datasets import load_dataset

ds = load_dataset("lmsys/mt_bench_human_judgments")
print(ds)

df = ds["human"].to_pandas()
print(df.shape)
print(df.columns.tolist())
print(df["winner"].value_counts())
print(df["turn"].value_counts())