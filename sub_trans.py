import pandas as pd
import re # Import regex for more robust ID splitting if needed

# Define the ID normalization function (same as before)
def normalize_id(input_id: str) -> str:
    """
    Transforms an ID from the format 'XXXX_Y.npy' to 'XXXX_A_Y'.
    Example: '7064_1.npy' -> '7064_A_1'

    Handles cases where .npy might be missing or _A_ already present.
    """
    # Remove .npy extension if present
    base_id = input_id.replace('.npy', '')

    # If it already contains '_A_', assume it's already normalized
    if '_A_' in base_id:
        return base_id

    # Split the base_id from the right by the last underscore to get 'XXXX' and 'Y'
    parts = base_id.rsplit('_', 1)

    if len(parts) == 2:
        # Reconstruct with '_A_'
        return f"{parts[0]}_A_{parts[1]}"
    else:
        # If it doesn't match the 'XXXX_Y' pattern, return as is.
        # This might indicate an unexpected ID format, but it prevents errors.
        return base_id

# --- Main script to transform your existing CSV ---

# Path to your initially generated submission CSV
input_csv_path = '/home/hipp/MPDD/answer_Track1/submission.csv' # Or whatever your file is named

# Path for the corrected submission CSV
output_csv_path = '/home/hipp/MPDD/answer_Track1/submission_cor.csv' # Or overwrite input_csv_path

try:
    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    print(f"Original IDs sample (first 5):")
    print(df['ID'].head())

    # Apply the transformation function to the 'ID' column
    df['ID'] = df['ID'].apply(normalize_id)

    print(f"\nCorrected IDs sample (first 5):")
    print(df['ID'].head())

    # Save the corrected DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)

    print(f"\nSuccessfully transformed IDs in '{input_csv_path}' and saved to '{output_csv_path}'.")

except FileNotFoundError:
    print(f"Error: The file '{input_csv_path}' was not found.")
except KeyError:
    print(f"Error: The CSV file does not contain an 'ID' column.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
