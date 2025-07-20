import pandas as pd
from io import StringIO

# Original data (df_original) - This is the DataFrame you want to modify
original_data_str = """ID,1s_bin,1s_tri,1s_pen,5s_bin,5s_tri,5s_pen
4173_A_1,0,2,0,1,2,0
4173_A_2,0,2,0,1,2,2
4173_A_3,0,2,0,0,2,0
4173_A_4,0,0,0,1,2,0
8411_A_3,0,0,0,0,2,0
8411_A_4,0,2,0,1,2,0
4621_A_4,0,2,0,1,2,0
3821_A_4,0,2,0,1,2,0
6516_A_1,0,0,0,0,1,0
6516_A_2,0,1,0,0,1,0
"""

# New data (df_new) - This contains the new values and uses the .npy ID format
new_data_str = """ID,1s_bin,1s_tri
4173_1.npy,0,2
4173_2.npy,0,2
4173_3.npy,0,2
4173_4.npy,0,1
8411_3.npy,0,2
8411_4.npy,0,1
"""

df_original = pd.read_csv('/home/hipp/MPDD/answer_Track1/submission_cor.csv')
df_new = pd.read_csv('/home/hipp/MPDD/answer_Track1/submission.csv')

print("Original DataFrame (df_original) before update:")
print(df_original)
print("\nNew DataFrame (df_new) with updated values:")
print(df_new)

# --- Step 1: Define a normalization function for a common merge key ---
def create_merge_key(input_id: str) -> str:
    """
    Creates a standardized ID key for merging from either 'XXXX_A_Y' or 'XXXX_Y.npy' to 'XXXX_Y'.
    """
    temp_id = input_id.replace('_A_', '_')
    temp_id = temp_id.replace('.npy', '')
    return temp_id

# --- Step 2: Create a temporary merge key in both DataFrames ---
df_original['_merge_key'] = df_original['ID'].apply(create_merge_key)
df_new['_merge_key'] = df_new['ID'].apply(create_merge_key)

# --- Step 3: Drop the old '1s_bin' and '1s_tri' columns from df_original ---
# We keep the _merge_key for now, as it's needed for the merge.
df_original.drop(columns=['1s_bin', '1s_tri'], inplace=True)

print("\nOriginal DataFrame after dropping 1s_bin and 1s_tri (before adding new):")
print(df_original)


# --- Step 4: Merge the df_original (now without 1s_bin/1s_tri) with df_new to add the new columns ---
# We perform a LEFT merge from df_original to df_new.
# This ensures that:
#   - All rows from df_original are kept.
#   - New '1s_bin' and '1s_tri' values are brought in where `_merge_key` matches.
#   - If an `_merge_key` from df_original doesn't exist in df_new, the new '1s_bin'
#     and '1s_tri' will be NaN.
final_df = pd.merge(df_original, df_new[['_merge_key', '1s_bin', '1s_tri']],
                    on='_merge_key',
                    how='left')

# --- Step 5: Clean up the temporary merge key column and handle NaNs for new columns ---
final_df.drop(columns=['_merge_key'], inplace=True)

# Important: If there are IDs in df_original that are *not* in df_new,
# their '1s_bin' and '1s_tri' columns will now be NaN.
# You need to decide how to handle these:
#   - Option A: Leave them as NaN (if that's acceptable for your next steps).
#   - Option B: Fill them with a default value (e.g., 0, or an average/mode).
#   - Option C: If you wanted to retain their *original* 1s_bin/1s_tri values if not updated,
#             then the previous `fillna(df_combined['1s_bin_old'])` approach
#             (which internally uses the merge suffixes) is what you need.
#
# Based on "replace... altogether", if an ID isn't in new data, its columns are effectively "unknown"
# if you literally dropped them and then added.
# For consistency and to preserve previous behavior of keeping original for unmatched,
# the `fillna` based on suffixes is usually better.

# However, if "replace altogether" means, "for IDs in the new data, use new data.
# For IDs NOT in the new data, their 1s_bin/1s_tri should remain as they were in df_original."
# Then the simpler method of `df_original.update(df_new_processed)` is best.

# Let's go with the most direct interpretation of "drop and add" for clarity,
# and then show how to handle NaNs if they appear (which they will if not all original
# IDs are in the new data).

# After the merge, if any 1s_bin/1s_tri are NaN, they indicate an ID in df_original
# that was NOT in df_new. You decide how to fill these.
# For example, fill with 0 or a specific default for that column:
# final_df['1s_bin'] = final_df['1s_bin'].fillna(0).astype(int)
# final_df['1s_tri'] = final_df['1s_tri'].fillna(0).astype(int)

# In this example, ALL original IDs are in df_new, so no NaNs will be introduced here.
# But for a real dataset, this is a critical consideration.

# --- Step 6: Reorder columns to match original structure (optional but good practice) ---
original_columns_order = ['ID', '1s_bin', '1s_tri', '1s_pen', '5s_bin', '5s_tri', '5s_pen']
final_df = final_df[original_columns_order]

print("\nFinal DataFrame after dropping old columns and adding new ones (ID order maintained):")
print(final_df)

# Verification
print("\nVerification of specific changes:")
print(f"4173_A_4 -> 1s_tri: {final_df[final_df['ID'] == '4173_A_4']['1s_tri'].iloc[0]}")
print(f"8411_A_3 -> 1s_tri: {final_df[final_df['ID'] == '8411_A_3']['1s_tri'].iloc[0]}")
print(f"8411_A_4 -> 1s_tri: {final_df[final_df['ID'] == '8411_A_4']['1s_tri'].iloc[0]}")

output_csv_filename = 'updated_submission.csv'

try:
    final_df.to_csv(output_csv_filename, index=False)
    print(f"\nDataFrame successfully saved to '{output_csv_filename}'")
except Exception as e:
    print(f"Error saving DataFrame to CSV: {e}")

# df_original = pd.read_csv('path/to/your/original_file.csv')
# df_new = pd.read_csv('path/to/your/new_data_file.csv')
# (then apply the steps above)
# df_original.to_csv('path/to/your/updated_original_file.csv', index=False)