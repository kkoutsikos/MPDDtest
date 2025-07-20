import json
import os
from collections import Counter

print("--- Script Started ---") # <--- ADD THIS LINE AT THE VERY TOP

# --- Configuration (Adjust if necessary) ---
data_rootpath = "/mnt/c/MPDD_data/MPDD-Elderly"
label_count = 2 # Based on your last run, for 'bin_category'
# -------------------------------------------

# Construct the full path to your testing JSON file
testing_json_path = os.path.join(data_rootpath, 'Testing', 'labels', 'Testing_files.json')

# Determine the correct label key based on label_count
label_key_map = {
    2: "bin_category",
    3: "tri_category",
    5: "pen_category"
}
label_key = label_key_map.get(label_count)

if not label_key:
    print(f"Error: Invalid label_count '{label_count}'. Please use 2, 3, or 5.")
    print("--- Script Exited ---") # <--- ADD THIS
    exit()

if not os.path.exists(testing_json_path):
    print(f"Error: Testing JSON file not found at '{testing_json_path}'")
    print("Please verify your 'data_rootpath' and the path to 'Testing_files.json'.")
    print("--- Script Exited ---") # <--- ADD THIS
    exit()

print(f"Checking classes in: {testing_json_path}")
print(f"Using label key: '{label_key}'")

all_labels = []

try:
    with open(testing_json_path, 'r') as f:
        test_data = json.load(f)

    for item in test_data:
        if label_key in item:
            all_labels.append(item[label_key])
        else:
            print(f"Warning: '{label_key}' not found in item: {item.get('audio_feature_path', 'N/A')}")

    if all_labels:
        label_counts = Counter(all_labels)
        print("\n--- Class Distribution in Testing Dataset ---")
        for label, count in label_counts.items():
            print(f"Class '{label}': {count} samples")
        print(f"Total samples: {len(all_labels)}")

        unique_labels = sorted(label_counts.keys())
        print(f"\nUnique classes found: {unique_labels}")

        if len(unique_labels) < label_count:
            print(f"\nWarning: Expected {label_count} classes but found only {len(unique_labels)}.")
            print("This indicates a highly imbalanced or single-class testing dataset, which can lead to misleading evaluation metrics (like your previous confusion matrix).")

    else:
        print("No labels found in the testing dataset using the specified label key.")

except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {testing_json_path}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Script Finished ---") # <--- ADD THIS LINE AT THE VERY END