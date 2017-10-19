import numpy as np
import time

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()

print('Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

z_1 = None
z_2 = None

################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
pythonStartTime = time.time()
a, b = 3, 5;
z_1 = [[0 for x in range(b)] for y in range(a)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3, 5))
numPyEndTime = time.time()

print(z_1)
print('Step 1 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(z_2)
print('Step 1 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
pythonStartTime = time.time()
z_1[0] = [7, 7, 7, 7, 7]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[0:1,:] = 7
numPyEndTime = time.time()

print(z_1)
print('Step 2 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(z_2)
print('Step 2 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
pythonStartTime = time.time()
for row in range(a):
    z_1[0][1] = 9
    z_1[1][1] = 9
    z_1[2][1] = 9
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[:, 1] = 9
numPyEndTime = time.time()

print(z_1)
print('Step 3 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(z_2)
print('Step 3 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[1:2,2:3] = 5
numPyEndTime = time.time()

print(z_1)
print('Step 4 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(z_2)
print('Step 4 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
pythonStartTime = time.time()
x_1 = []
for i in range(50, 100):
    x_1.append(i)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
x_2 = np.arange(50, 100, 1)
numPyEndTime = time.time()

print(x_1)
print('Step 5 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(x_2)
print('Step 5 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

##############

y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python

pythonStartTime = time.time()
def createMatrix(row, col, value):
    matrix = []

    for i in range(row):
        rows = []
        for j in range(col):
            rows.append(value[col * i + j])
        matrix.append(rows)

    return matrix

m = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y_1 = createMatrix(4, 4, m)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
y_2 = np.arange(16).reshape(4, 4)
numPyEndTime = time.time()

print(y_1)
print('Step 6 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(y_2)
print('Step 6 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

##############

tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
pythonStartTime = time.time()
m = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
tmp_1 = createMatrix(5, 5, m)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones((5, 5))
tmp_2[1:-1, 1:-1] = 0
numPyEndTime = time.time()

##############
print(tmp_1)
print('Step 7 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(tmp_2)
print('Step 7 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

##############


a_1 = None;
a_2 = None
b_1 = None;
b_2 = None
c_1 = None;
c_2 = None

#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
pythonStartTime = time.time()
a = []
for i in range(0, 5000):
    a.append(i)

print (a)

a_1 = createMatrix(50, 100, a)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
a_2 = np.arange(5000).reshape(50, 100)
numPyEndTime = time.time()

print(a_1)
print('Step 8 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(a_2)
print('Step 8 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
pythonStartTime = time.time()
b = []
for i in range(0, 20000):
    b.append(i)
b_1 = createMatrix(100, 200, b)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
b_2 = np.arange(20000).reshape(100, 200)
numPyEndTime = time.time()

print(b_1)
print('Step 9 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(b_2)
print('Step 9 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
pythonStartTime = time.time()
a, b = 50, 200;
c_1 = [[0 for x in range(b)] for y in range(a)]
for i in range(len(a_1)):
    for j in range(len(b_1[0])):
        for k in range(len(b_1)):
            c_1[i][j] += a_1[i][k] * b_1[k][j]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
c_2 = np.dot(a_2, b_2)
numPyEndTime = time.time()

print(c_1)
print('Step 10 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print(c_2)
print('Step 10 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))

d_1 = None;
d_2 = None

################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
pythonStartTime = time.time()
d_1 = np.random.random((3, 3))
print ("Original Python Matrix:")
print (d_1)
pythonEndTime = time.time()

def normalize(M):
    row_sums = d_1.sum(axis=1)
    return d_1 / row_sums

normalize(d_1)
print("Python Matrix after Normalization:")
print(d_1)

# NumPy
numPyStartTime = time.time()
d_2 = np.random.random((3, 3))
print("Original Numpy Matrix:")
print(d_2)
d_2_min = d_2.min()
d_2_max = d_2.max()
d_2 = (d_2 - d_2_min) / (d_2_max - d_2_min)
print("Numpy Matrix after Normalization:")
numPyEndTime = time.time()
print(d_2)

print('Step 11 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print('Step 11 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
pythonStartTime = time.time()

def mean(values):
    length = len(values)
    total_sum = 0
    for i in range(length):
        total_sum += values[i]
    total_sum = sum(values)
    average = total_sum/length
    return average

new_a_1 = []
for row in a_1:
    mean_row = mean(row)
    for i in range(100):
        row[i] = row[i] - mean_row
        new_a_1.append(row[i])

print(new_a_1)


pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
new_a_2 = a_2 - a_2.mean(axis=1).reshape(-1, 1)
print(new_a_2)
numPyEndTime = time.time()

print('Step 12 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print('Step 12 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))


###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
pythonStartTime = time.time()
newMatrix = []
x = 0
for i in range(100):
    mean = 9900
    for j in range(x, x+200):
        newMatrix.append((j-mean))
        mean += 1
    x = x + 200

b_1 = createMatrix(100, 200, newMatrix)
print(b_1)
pythonEndTime = time.time()

# NumPy

numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(axis=0)
print(b_2)
numPyEndTime = time.time()

print('Step 13 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print('Step 13 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))


################
print(np.sum(new_a_1 == new_a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None;
e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
pythonStartTime = time.time()
print ("Original c_1 matrix:")
for row in c_1:
 print(row)
print("\n")
c_1_transposed = list(zip(*c_1))

a, b = 200, 50;
matrix_5 = [[5 for x in range(b)] for y in range(a)]

print("\n")
print("Transposed c_1 matrix:")
for row in c_1_transposed:
    print(row)

print("\n")
e_1 = [[0 for x in range(b)] for y in range(a)]

for i in range(len(e_1)):
   for j in range(len(e_1[0])):
       e_1[i][j] = c_1_transposed[i][j] + matrix_5[i][j]

print ("Transposed c_1 matrix + 5 :")
for row in e_1:
       print(row)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
e_2 = np.transpose(c_2)
e_2 = e_2 + 5
print(e_2)
numPyEndTime = time.time()

print('Step 14 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print('Step 14 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))


##################
print(np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
pythonStartTime = time.time()
f_1 = []
for i in e_1:
    f_1.append(i)

print(f_1)
f_1_length = len(f_1)
f_1 = (f_1_length,)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
f_2 = e_2.flatten()
print(f_2)
f_2 = f_2.shape

numPyEndTime = time.time()

print(f_1)
print(f_2)

print('Step 15 Python time: {0} sec.'.format(pythonEndTime - pythonStartTime))
print('Step 15 NumPy time: {0} sec.'.format(numPyEndTime - numPyStartTime))
