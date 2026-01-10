#1st question
def count_pairs(list1, target):
    count = 0

    for i in range(len(list1)):
        for j in range(i + 1, len(list1)):
            if list1[i] + list1[j] == target:
                count += 1

    return count  



list1 = [2, 7, 4, 1, 3, 6]
target = 10

count = count_pairs(list1, target)
print("Total number of pairs with sum 10:", count)


#2nd question
def diff(list2):
    
    if len(list2) < 3:
        return "Range determination not possible"

    
    min_value = list2[0]
    max_value = list2[0]

    
    for i in range(1, len(list2)):
        if list2[i] < min_value:
            min_value = list2[i]
        if list2[i] > max_value:
            max_value = list2[i]

   
    return max_value - min_value



values = [5, 3, 8, 1, 0, 4]
result = diff(values)

print("Range of the list:", result)

#3rd question
def matrix_multiply(A, B):
    n = len(A)
    C = []

    for i in range(n):
        row = []
        for j in range(n):
            sum_value = 0
            for k in range(n):
                sum_value += A[i][k] * B[k][j]
            row.append(sum_value)
        C.append(row)

    return C


def matrix_power(A, m):
    result = A

    for count in range(1, m):
        result = matrix_multiply(result, A)

    return result

n = int(input())
A = []

for i in range(n):
    row = []
    for j in range(n):
        row.append(int(input()))
    A.append(row)

m = int(input())

power_matrix = matrix_power(A, m)

for row in power_matrix:
    print(row)

#4th question
def max_occuring(str):
 ch={}
 n=len(str)
 ans=' '
 cnt=0
 for i in range(n):
  if str[i] in ch:
   ch[str[i]] +=1
  else:
   ch[str[i]]=1

  if cnt < ch[str[i]]:
     ans=str[i]
     cnt=ch[str[i]]
 return ans

str="hippopotamus"
print("Max occuring character is: ",max_occuring(str))
   


#5th question
import random
import statistics

def calc(numbers):
    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    mode = statistics.multimode(numbers)
    return mean, median, mode


num = 25
a = 1
b = 10

abc = []

for i in range(num):
    abc.append(random.randint(a, b))

print(abc)

mean, median, mode = calc(abc)

print(mean)
print(median)
print(mode)
#1st question
def count_pairs(list1, target):
    count = 0

    for i in range(len(list1)):
        for j in range(i + 1, len(list1)):
            if list1[i] + list1[j] == target:
                count += 1

    return count  



list1 = [2, 7, 4, 1, 3, 6]
target = 10

count = count_pairs(list1, target)
print("Total number of pairs with sum 10:", count)


#2nd question
def diff(list2):
    
    if len(list2) < 3:
        return "Range determination not possible"

    
    min_value = list2[0]
    max_value = list2[0]

    
    for i in range(1, len(list2)):
        if list2[i] < min_value:
            min_value = list2[i]
        if list2[i] > max_value:
            max_value = list2[i]

   
    return max_value - min_value



values = [5, 3, 8, 1, 0, 4]
result = diff(values)

print("Range of the list:", result)

#3rd question
def matrix_multiply(A, B):
    n = len(A)
    C = []

    for i in range(n):
        row = []
        for j in range(n):
            sum_value = 0
            for k in range(n):
                sum_value += A[i][k] * B[k][j]
            row.append(sum_value)
        C.append(row)

    return C


def matrix_power(A, m):
    result = A

    for count in range(1, m):
        result = matrix_multiply(result, A)

    return result

n = int(input())
A = []

for i in range(n):
    row = []
    for j in range(n):
        row.append(int(input()))
    A.append(row)

m = int(input())

power_matrix = matrix_power(A, m)

for row in power_matrix:
    print(row)

#4th question
def max_occuring(str):
 ch={}
 n=len(str)
 ans=' '
 cnt=0
 for i in range(n):
  if str[i] in ch:
   ch[str[i]] +=1
  else:
   ch[str[i]]=1

  if cnt < ch[str[i]]:
     ans=str[i]
     cnt=ch[str[i]]
 return ans

str="hippopotamus"
print("Max occuring character is: ",max_occuring(str))
   


#5th question
import random
import statistics

def calc(numbers):
    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    mode = statistics.multimode(numbers)
    return mean, median, mode


num = 25
a = 1
b = 10

abc = []

for i in range(num):
    abc.append(random.randint(a, b))

print(abc)

mean, median, mode = calc(abc)

print(mean)
print(median)
print(mode)
