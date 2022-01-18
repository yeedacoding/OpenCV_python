import numpy as np

mylist = [1, 2, 3]
print(type(mylist))

# make numpy array
myarray = np.array(mylist)
print(myarray)
print(type(myarray))

# set range of array
a = np.arange(0, 10, 2)
print(a)

# zero values array
b = np.zeros(shape=(10, 5))
print(b)

# one values array
c = np.ones(shape=(2, 4))
print(c)

# random array values
np.random.seed(101)
arr = np.random.randint(0, 100, 10)
print(arr)

arr2 = np.random.randint(0, 100, 10)
print(arr2)

# max, max index, min, min index
print(arr.max())
print(arr.argmax())
print(arr.min())
print(arr.argmin())
print(arr.mean())

# array reshape
arr = arr.reshape((2, 5))
print(arr)
arr = arr.reshape((5, 2))
print(arr)

# indexing
mat = np.arange(0, 100).reshape(10, 10)
print(mat)
print(mat.shape)

row = 0
col = 1

print(mat[row, col])
print(mat[4, 6])

# slicing
print(mat[:, 1])
print(mat[4, :])
print(mat[0:3, 0:3])
mat[0:3, 0:3] = 0
print(mat)

# copy
newmat = mat.copy()
newmat[:6, :] = 999
print(newmat)
