import numpy as np

# Mean
print("#MEAN")

X = [1, 2, 3, 4, 5]
print("Mean:", np.mean(X))  # Tính trung bình của mảng X

X = np.array([[1, 2], [3, 4]])
print("Mean X:", np.mean(X))  # Tính trung bình của mảng X
print("Mean X with axis = 0: ", np.mean(X, axis=0))  # Trung bình theo trục 0
print("Mean X with axis = 1: ", np.mean(X, axis=1))  # Trung bình theo trục 1

# Creating array 'a' with zeros
print("Creating array 'a' with zeros")

a = np.zeros((2, 512*512), dtype=np.float32)
a[0, :] = 1.0
a[1, :] = 0.1

print("a.shape: ", a.shape)  # In ra kích thước của mảng 'a'
print("mean a = ", np.mean(a))  # Tính trung bình của mảng 'a'

print("mean a = ", np.mean(a, dtype=np.float64))  # Tính trung bình của mảng 'a' với kiểu dữ liệu float64

# Median
print("#MEDIAN")

X = np.array([2, 5, 3, 1, 7])
Y = np.array([2, 1, 8, 5, 7, 9])

print("Median X:", np.median(X))  # Tính trung vị của mảng X
print("Median Y:", np.median(Y))  # Tính trung vị của mảng Y

arr = np.array([[7, 4, 2], [3, 9, 5]])
print("median arr (axis = 0) =", np.median(arr, axis=0))  # Tính trung vị theo trục 0
print("median arr (axis = 1) =", np.median(arr, axis=1))  # Tính trung vị theo trục 1

# Mean & Median with NaN
print("#MEAN & MEDIAN with NaN")

X = np.array([2, np.nan, 5, 9])
print("Mean X:", np.mean(X))  # Tính trung bình của mảng X với xử lý NaN
print("Median X:", np.median(X))  # Tính trung vị của mảng X với xử lý NaN
print("Mean = ", np.nanmean(X))  # Tính trung bình của mảng X với xử lý NaN
print("Median =", np.nanmedian(X))  # Tính trung vị của mảng X với xử lý NaN

# Variance & standard deviation
print("#VARIANCE & STANDARD DEVIATION")

X = [19, 33, 51, 22, 18, 13, 45, 24, 58, 11]
print("Variance X:", np.var(X))  # Tính phương sai của mảng X
print("Standard deviation X:", np.std(X))  # Tính độ lệch chuẩn của mảng X

# Variance & standard deviation with NaN
print("#VARIANCE & STANDARD DEVIATION with NaN")

A = np.array([1, np.nan, 3, 4])
print("var = ", np.var(A))  # Tính phương sai của mảng A với xử lý NaN
print("std = ", np.std(A))  # Tính độ lệch chuẩn của mảng A với xử lý NaN
print("nan var =", np.nanvar(A))  # Tính phương sai của mảng A với xử lý NaN
print("nan std = ", np.nanstd(A))  # Tính độ lệch chuẩn của mảng A với xử lý NaN

# Order statistics
print("#ORDER STATISTICS")

X = np.array([[14, 96],
              [46, 82],
              [80, 67],
              [77, 91],
              [99, 87]])
print("X= ", X)

print("min X = ", np.min(X))  # Tìm giá trị nhỏ nhất trong mảng X
print("max X = ", np.max(X))  # Tìm giá trị lớn nhất trong mảng X

print("Max: (axis =0) = ", np.max(X, axis=0))  # Tìm giá trị lớn nhất theo trục 0
print("Min: (axis = 1) = ", np.min(X, axis=1))  # Tìm giá trị nhỏ nhất theo trục 1

# Order statistics with NaN
print("#ORDER STATISTICS with NaN")

X = np.array([[14, 96],
              [np.nan, 82],
              [77, np.nan],
              [99, 87]])
print("X = ", X)

print("Max: ", np.nanmax(X))  # Tìm giá trị lớn nhất với xử lý NaN
print("Min: ", np.nanmin(X))  # Tìm giá trị nhỏ nhất với xử lý NaN

# Range
print("#RANGE")

X = np.array([[14, 96],
              [46, 82],
              [80, 67],
              [77, 91],
              [99, 87]])
print("x = ", X)

print("Range X = ", np.ptp(X))  # Tính phạm vi của mảng X
print("Range X (axis = 0) = ", np.ptp(X, axis=0))  # Tính phạm vi theo trục 0
print("Range X (axis = 1) = ", np.ptp(X, axis=1))  # Tính phạm vi theo trục 1

# Correlation
print("#CORRELATION")

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

correlation_matrix = np.corrcoef(arr1, arr2)  # Tính ma trận tương quan
correlation = correlation_matrix[0, 1]  # Lấy giá trị hệ số tương quan

print("Correlation:", correlation)  # In ra hệ số tương quan
