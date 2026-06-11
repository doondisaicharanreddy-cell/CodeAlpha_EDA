# ============================================================
# TASK 2: Exploratory Data Analysis (EDA)
# Dataset: Titanic Passenger Data
# ============================================================

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style (makes charts look clean)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 60)
print("   TITANIC DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# SECTION 1: LOAD THE DATA
# ============================================================

df = pd.read_csv(r"D:\titanic.csv")
print("\n✅ Dataset loaded successfully!")

# ============================================================
# SECTION 2: UNDERSTAND THE STRUCTURE
# ============================================================

print("\n--- 2.1 First 5 rows of data ---")
print(df.head())
# head() shows the first 5 rows so you can see what data looks like

print("\n--- 2.2 Last 5 rows of data ---")
print(df.tail())

print("\n--- 2.3 Shape of dataset (rows, columns) ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- 2.4 Column names ---")
print(df.columns.tolist())

print("\n--- 2.5 Data types of each column ---")
print(df.dtypes)
# This tells you if a column is a number (int/float) or text (object)

print("\n--- 2.6 Basic info ---")
print(df.info())
# info() shows column names, data types, and non-null counts

# ============================================================
# SECTION 3: STATISTICAL SUMMARY
# ============================================================

print("\n--- 3.1 Statistical summary (numbers only) ---")
print(df.describe())
# describe() gives: count, mean, min, max, std, 25%, 50%, 75%
# This only works on numeric columns

print("\n--- 3.2 Statistical summary (text columns) ---")
print(df.describe(include="object"))
# include="object" shows stats for text columns

# ============================================================
# SECTION 4: MISSING VALUES ANALYSIS
# ============================================================

print("\n--- 4.1 Missing values count per column ---")
print(df.isnull().sum())
# isnull() marks each cell True/False (True = missing)
# .sum() counts the Trues per column

print("\n--- 4.2 Missing values percentage ---")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.round(2))

# Visualize missing values
plt.figure(figsize=(10, 5))
missing = df.isnull().sum()
missing = missing[missing > 0]  # Only show columns with missing values
missing.plot(kind="bar", color="tomato", edgecolor="black")
plt.title("Missing Values Per Column", fontsize=16)
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot1_missing_values.png")
plt.show()
print("✅ Plot 1 saved: Missing Values")

# ============================================================
# SECTION 5: HANDLE MISSING VALUES
# ============================================================

print("\n--- 5.1 Handling missing values ---")

# Age: fill missing with median age (median is better than mean for skewed data)
median_age = df["Age"].median()
df["Age"].fillna(median_age, inplace=True)
print(f"Age: filled {df['Age'].isnull().sum()} missing values with median ({median_age})")

# Embarked: fill missing with most common value (mode)
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"].fillna(mode_embarked, inplace=True)
print(f"Embarked: filled missing with mode ('{mode_embarked}')")

# Cabin: too many missing (77%) — drop the column
df.drop(columns=["Cabin"], inplace=True)
print("Cabin column dropped (too many missing values)")

print("\n--- 5.2 Verify no more missing values ---")
print(df.isnull().sum())

# ============================================================
# SECTION 6: UNIVARIATE ANALYSIS
# (Looking at ONE column at a time)
# ============================================================

print("\n--- 6.1 Survival Count ---")
print(df["Survived"].value_counts())
print(f"Survival Rate: {df['Survived'].mean()*100:.2f}%")

# Plot: Survival count
plt.figure(figsize=(7, 5))
df["Survived"].value_counts().plot(
    kind="bar",
    color=["tomato", "steelblue"],
    edgecolor="black"
)
plt.title("Survival Count (0=Died, 1=Survived)", fontsize=16)
plt.xlabel("Survived")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot2_survival_count.png")
plt.show()
print("✅ Plot 2 saved: Survival Count")

# Plot: Age distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="steelblue")
# kde=True adds a smooth curve on top of the histogram
plt.title("Age Distribution of Passengers", fontsize=16)
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot3_age_distribution.png")
plt.show()
print("✅ Plot 3 saved: Age Distribution")

# Plot: Passenger Class count
plt.figure(figsize=(7, 5))
df["Pclass"].value_counts().sort_index().plot(
    kind="bar",
    color=["gold", "silver", "peru"],
    edgecolor="black"
)
plt.title("Passengers by Class", fontsize=16)
plt.xlabel("Class (1=First, 2=Second, 3=Third)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot4_passenger_class.png")
plt.show()
print("✅ Plot 4 saved: Passenger Class")

# ============================================================
# SECTION 7: BIVARIATE ANALYSIS
# (Comparing TWO columns to find relationships)
# ============================================================

print("\n--- 7.1 Survival by Gender ---")
print(df.groupby("Sex")["Survived"].mean() * 100)
# groupby groups rows by Sex, then calculates survival rate

# Plot: Survival by Gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Sex", y="Survived", palette="Set2")
plt.title("Survival Rate by Gender", fontsize=16)
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("plot5_survival_by_gender.png")
plt.show()
print("✅ Plot 5 saved: Survival by Gender")

print("\n--- 7.2 Survival by Passenger Class ---")
print(df.groupby("Pclass")["Survived"].mean() * 100)

# Plot: Survival by Class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Pclass", y="Survived", palette="Blues_d")
plt.title("Survival Rate by Passenger Class", fontsize=16)
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("plot6_survival_by_class.png")
plt.show()
print("✅ Plot 6 saved: Survival by Class")

# Plot: Age vs Survival (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Survived", y="Age", palette="Set3")
# Boxplot shows the spread of ages for survived vs died
plt.title("Age Distribution: Survived vs Died", fontsize=16)
plt.xlabel("Survived (0=Died, 1=Survived)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("plot7_age_vs_survival.png")
plt.show()
print("✅ Plot 7 saved: Age vs Survival")

# ============================================================
# SECTION 8: CORRELATION ANALYSIS
# (How strongly are columns related to each other?)
# ============================================================

print("\n--- 8.1 Correlation Matrix ---")
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()
print(correlation)
# Values close to 1 = strong positive relationship
# Values close to -1 = strong negative relationship
# Values close to 0 = no relationship

# Plot: Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,        # Show numbers on the heatmap
    fmt=".2f",         # Round to 2 decimal places
    cmap="coolwarm",   # Color scheme
    linewidths=0.5
)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("plot8_correlation_heatmap.png")
plt.show()
print("✅ Plot 8 saved: Correlation Heatmap")

# ============================================================
# SECTION 9: KEY INSIGHTS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("   KEY INSIGHTS FROM EDA")
print("=" * 60)

total = len(df)
survived = df["Survived"].sum()
survival_rate = df["Survived"].mean() * 100

female_survival = df[df["Sex"] == "female"]["Survived"].mean() * 100
male_survival = df[df["Sex"] == "male"]["Survived"].mean() * 100

class1_survival = df[df["Pclass"] == 1]["Survived"].mean() * 100
class3_survival = df[df["Pclass"] == 3]["Survived"].mean() * 100

avg_age = df["Age"].mean()

print(f"\n📊 Total Passengers   : {total}")
print(f"💀 Total Survived     : {survived}")
print(f"📈 Overall Survival   : {survival_rate:.1f}%")
print(f"\n👩 Female Survival    : {female_survival:.1f}%")
print(f"👨 Male Survival      : {male_survival:.1f}%")
print(f"\n🥇 1st Class Survival : {class1_survival:.1f}%")
print(f"🥉 3rd Class Survival : {class3_survival:.1f}%")
print(f"\n🎂 Average Age        : {avg_age:.1f} years")

print("\n✅ EDA Complete! All 8 plots saved as PNG files.")
print("=" * 60)