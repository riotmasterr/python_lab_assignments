#Program - 1 : To count the number of vowels and consoants in the given string
string = input("enter a string: ")

vowels = 0
consonants = 0

for i in string: #checks if a character is vowel or not
    if (i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u' or
        i == 'A' or i == 'E' or i == 'I' or i == 'O' or i == 'U'):
        vowels = vowels + 1
    elif i.isalpha():   # count consonants only if it's a letter
        consonants = consonants + 1

print("the number of vowels is =", vowels)
print("the number of consonants is =", consonants)






#Program 2 : Multiplication of matrix

#final 2
MAX = 20 # maximum size of the matrix

def printmatrix(M, rowsize, colsize):
    for i in range(rowsize):
        for j in range(colsize):
            print(M[i][j], end=" ")
        print()

def multiplymatrix(row1, col1, A, row2, col2, B):
#conditions for the multiplication of  matrix
    if col1 != row2:
        return None

#resultat matrix
    C = [[0 for i in range(MAX)] for j in range(MAX)]

#logic for the matrix multiplication
    for i in range(row1):
        for j in range(col2):
            for k in range(col1):
                C[i][j] += A[i][k] * B[k][j]

    return C


A = [[0 for i in range(MAX)] for j in range(MAX)]
B = [[0 for i in range(MAX)] for j in range(MAX)]

row1 = int(input("enter the number of rows in matrix A: "))
col1 = int(input("enter the number of columns in matrix A: "))

print("enter the elements for the first matrix:")
for i in range(row1):
    for j in range(col1):
        A[i][j] = int(input())

row2 = int(input("enter the number of rows of second matrix: "))
col2 = int(input("enter the number of columns of second matrix: "))

print("enter the elements for the second matrix:")
for i in range(row2):
    for j in range(col2):
        B[i][j] = int(input())

print("first matrix:")
printmatrix(A, row1, col1)

print("second matrix:")
printmatrix(B, row2, col2)

result = multiplymatrix(row1, col1, A, row2, col2, B)

if result is None:
    print("matrix multiplication is not possible")
else:
    print("final matrix:")
    printmatrix(result, row1, col2)









#Program 3: Finding the common elements between two lists

list1 = [1,2,3,4,5,6]
list2 = [2,3,4,5,6,7]

common = []

for i in list1:
    if i in list2 and i not in common:
        common.append(i)

print("common elements:", common)










# Program 4: find the transpose of a matrix

def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    Tmat = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            Tmat[j][i] = matrix[i][j]

    return Tmat

rows = int(input("enter the number of rows: "))
cols = int(input("enter the number of columns: "))

matrix = []
print("enter the matrix elements:")
for i in range(rows):
    row = list(map(int, input().split()))
    matrix.append(row)

res = transpose(matrix)

print("transpose of the matrix:")
for row in res:
    for elem in row:
        print(elem, end=" ")
    print() 










#Program 5 : finding 100 random integers between the range of 100 - 150 and find there mean median mode
import random
import statistics

def calculate(numbers):
    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    mode = statistics.multimode(numbers)
    return mean, median, mode


no = 100
a = 100
b = 150

res = []

for i in range(no):
    res.append(random.randint(a, b))

print(res)

mean, median, mode = calculate(res)

print(mean)
print(median)
print(mode)