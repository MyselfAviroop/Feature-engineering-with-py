# # Import necessary libraries
# import seaborn as sns               # for loading dataset
# import matplotlib.pyplot as plt     # for visualizing data

# # Load Titanic dataset directly from seaborn's built-in datasets
# df = sns.load_dataset('titanic')

# # Display first 5 rows of the dataset
# print("First 5 rows of the dataset:")
# print(df.head())

# # Check for missing values in each column
# print("\nMissing values in each column:")
# print(df.isnull().sum())  # shows how many nulls (NaNs) are present in each column

# # Show the shape (rows, columns) of the dataset before dropping missing data
# print("\nOriginal shape of the dataset:", df.shape)

# # Drop rows that contain any missing values and show the new shape
# print("Shape after dropping rows with missing values:", df.dropna().shape)

# # Drop columns that contain any missing values and show the new shape
# print("Shape after dropping columns with missing values:", df.dropna(axis=1).shape)

# # Plot the distribution of the 'age' column to check if it's normally distributed
# print("\nDisplaying age distribution plot...")
# sns.histplot(df['age'], kde=True)  # KDE = Kernel Density Estimation (smooth curve)
# plt.title("Age Distribution with KDE")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()

# # Fill missing values in 'age' column using the mean value
# df['Age_mean'] = df['age'].fillna(df['age'].mean())

# # Show comparison between original and mean-imputed 'age'
# print("\nOriginal 'age' vs Mean-Imputed 'Age_mean':")
# print(df[['age', 'Age_mean']].head(10))

# # Fill missing values in 'age' column using the median value
# df['age_median'] = df['age'].fillna(df['age'].median())

# # Show comparison between original, mean, and median imputed values
# print("\nOriginal 'age' vs Mean-Imputed vs Median-Imputed:")
# print(df[['age', 'Age_mean', 'age_median']].head(10))

# # Find rows where 'embarked' column has missing values
# print("\nRows where 'embarked' is missing:")
# print(df[df['embarked'].isnull()])

# # Show all unique values in the 'embarked' column
# print("\nUnique values in 'embarked' column:")
# print(df['embarked'].unique())

# # Calculate the most frequent (mode) value from non-null 'embarked' values
# mode_value = df[df['embarked'].notna()]['embarked'].mode()[0]

# # Fill missing values in 'embarked' column using mode
# df['embarked_mode'] = df['embarked'].fillna(mode_value)

# # Compare original and mode-imputed 'embarked' values
# print("\nOriginal 'embarked' vs Mode-Imputed 'embarked_mode':")
# print(df[['embarked', 'embarked_mode']].head(10))

# # Final null value check after imputation
# print("\nMissing values after imputation:")
# print("Original 'embarked' still missing:", df['embarked'].isnull().sum())
# print("Imputed 'embarked_mode' missing:", df['embarked_mode'].isnull().sum())


# # Import necessary libraries
# from sklearn.datasets import make_classification
# import pandas as pd
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE  # ✅ Correct import

# # Create an imbalanced dataset
# x, y = make_classification(
#     n_samples=1000,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     n_clusters_per_class=1,
#     weights=[0.9],      # 90% of class 0
#     flip_y=0,
#     random_state=12
# )

# # Convert to DataFrame
# df1 = pd.DataFrame(x, columns=['f1', 'f2'])
# df2 = pd.DataFrame(y, columns=['target'])
# final_df = pd.concat([df1, df2], axis=1)

# # Display initial info
# print("Before SMOTE:")
# print(final_df.head())
# print(final_df['target'].value_counts())

# # # Plot before SMOTE
# # plt.title("Before SMOTE - Imbalanced Classes")
# # plt.scatter(final_df['f1'], final_df['f2'], c=final_df['target'], cmap='coolwarm')
# # plt.xlabel("f1")
# # plt.ylabel("f2")
# # plt.colorbar(label="Class")
# # plt.show()

# # Apply SMOTE
# smote = SMOTE(random_state=42)
# x_resampled, y_resampled = smote.fit_resample(final_df[['f1', 'f2']], final_df['target'])

# # Convert resampled data to DataFrame
# df_resampled = pd.DataFrame(x_resampled, columns=['f1', 'f2'])
# df_resampled['target'] = y_resampled

# # Print class distribution after SMOTE
# print("\nAfter SMOTE:")
# print(df_resampled['target'].value_counts())  # Should now be balanced (e.g., 900:900)

# # Plot after SMOTE
# plt.title("After SMOTE - Balanced Classes")
# plt.scatter(df_resampled['f1'], df_resampled['f2'], c=df_resampled['target'], cmap='coolwarm')
# plt.xlabel("f1")
# plt.ylabel("f2")
# plt.colorbar(label="Class")
# plt.show()


# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


# # List of marks
# lst_marks = [45, 23, 56, 78, 90, 12, 34, 67, 89, 100,1000,-200]

# # Get minimum, Q1, median (Q2), Q3, and maximum
# minimum, Q1, median, Q3, maximum = np.quantile(lst_marks, [0, 0.25, 0.50, 0.75, 1])

# # Print the quartiles
# print("Minimum:", minimum)
# print("Q1 (25th percentile):", Q1)
# print("Median (50th percentile):", median)
# print("Q3 (75th percentile):", Q3)
# print("Maximum:", maximum)
# IQR = Q3 - Q1
# print("Interquartile Range (IQR):", IQR)
# lower_fence = Q1 - 1.5 * IQR
# upper_fence = Q3 + 1.5 * IQR
# print("Lower Fence:", lower_fence)
# print("Upper Fence:", upper_fence)
# import seaborn as sns
# sns.boxplot(x=lst_marks, color="skyblue")
# plt.title("Boxplot of Marks with Potential Outliers")
# plt.xlabel("Marks")
# plt.grid(True, axis='x', linestyle='--', alpha=0.5)
# plt.show()









# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# import pandas as pd

# # Sample data
# df = pd.DataFrame({
#     'color': ['red', 'blue', 'green', 'blue', 'red']
# })

# # --------------------
# # 🟩 One-Hot Encoding
# # --------------------
# encoder = OneHotEncoder()
# encoded = encoder.fit_transform(df[['color']]).toarray()
# encoder_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['color']))
# print("One-Hot Encoded DataFrame:")
# print(encoder_df)

# # --------------------
# # 🟨 Label Encoding
# # --------------------
# lbl_encoder = LabelEncoder()
# label_encoded = lbl_encoder.fit_transform(df['color'])
# print("\nLabel Encoded values:", label_encoded)

# single_transform = lbl_encoder.transform(['red'])
# print("\nLabel Encoding for 'red':", single_transform)




#ordinal encoding
# from sklearn.preprocessing import OrdinalEncoder
# import pandas as pd
# df=pd.DataFrame({
#     'size': ['small', 'medium', 'large', 'medium', 'small','large']})
# print(df)
# encoder= OrdinalEncoder(categories=[['small', 'medium', 'large']])
# encoded=encoder.fit_transform(df[['size']])
# print("Ordinal Encoded DataFrame:")
# print(encoded)


#Targeted guided ordinal encoding

import pandas as pd
df=pd.DataFrame({ 
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'price': [100, 200, 150, 300, 250]
})
print(df)
mean=df.groupby('city')['price'].mean().to_dict()
print("Mean prices by city:", mean)
# Step 2: Map the city names to their mean prices
df['city_encoded'] = df['city'].map(mean)
print(df)
