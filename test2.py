import numpy as np

# Create a 4x4 NumPy array
original_array = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]])

# Create a 2x2 replacement array
replacement_array = np.array([[99, 98],
                               [97, 96]])

# Specify the indices for the subarray you want to replace
row_start, row_end = 1, 3  # Rows 1 and 2
col_start, col_end = 1, 3  # Columns 1 and 2

# Replace the subarray with the replacement array
original_array[row_start:row_end, col_start:col_end] = replacement_array

# Print the updated array
print(original_array)