import pandas as pd

from .train import *


resume_df = pd.read_csv("data/resume-dataset.csv")
resume_df["Keywords"] = resume_df["Resume"].apply(clean_text)
resume_df["Tag"], labels = encode_labels(resume_df["Category"])
word_counts, words = generate_dictionary(resume_df)

print(word_counts)
