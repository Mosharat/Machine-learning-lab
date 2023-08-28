def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    # Create a 2D matrix to store the edit distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Calculate the edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

# Get input from the user
str1 = input("Enter the first string: ")
str2 = input("Enter the second string: ")

# Calculate the edit distance and display the result
distance = edit_distance(str1, str2)
print(f"The edit distance between '{str1}' and '{str2}' is: {distance}")
