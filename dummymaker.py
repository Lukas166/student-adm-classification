import pandas as pd
import numpy as np

np.random.seed(42)

names = [f"Kandidat {i+1}" for i in range(100)]

data = {
    "Nama": names,
    "GRE_Score": np.random.randint(260, 341, 100),
    "TOEFL_Score": np.random.randint(0, 121, 100),
    "University_Rating": np.random.randint(1, 6, 100).astype(int),
    "SOP": np.round(np.random.uniform(1.0, 5.0, 100), 1),
    "LOR": np.round(np.random.uniform(1.0, 5.0, 100), 1),
    "GPA": np.round(np.random.uniform(0.00, 4.00, 100), 2),
    "Research": np.random.randint(0, 2, 100).astype(int)
}

df_dummy = pd.DataFrame(data)
df_dummy.to_csv("dummy_kandidat_100.csv", index=False)

df_dummy.head()