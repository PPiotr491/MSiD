import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter

# # Download latest version
# path = kagglehub.dataset_download("mohankrishnathalla/sleep-health-and-daily-performance-dataset", output_dir="C:/Users/Super/PycharmProjects/MSiD/resources/")
#
# print("Path to dataset files:", path)
#
# file_path = "C:/Users/Super/PycharmProjects/MSiD/resources/student_mental_health_burnout.csv"

# df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,
#                             "sehaj1104/student-mental-health-and-burnout-dataset",
#                             file_path,
#                             )

df = pd.read_csv(file_path, )

print("First 5 records:", df.head())

# pandas.set_option('display.max_columns', None)
# print(df.loc[0:5])

print(df.info())

pd.set_option('display.max_columns', None)
print(df.describe())

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['age'].dropna(), kde=True, color='skyblue')
plt.title('Rozkład wieku pasażerów')


