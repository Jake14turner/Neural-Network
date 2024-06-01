import numpy as np
#When you are creating a neural network,
#in this example i have an input vector and i am trying to see which weight vector is closest.
input_vector = [1, 2, 3]
weights1 = [1, 3, 4]
weights2 = [-1, -2, -3]
#varibale which will store closer one
closer = ""

#calculate the dot product of input and weight1:
result1 = abs(np.dot(input_vector, weights1))

#calculate the dot product of input and weight2:
result2 = abs(np.dot(input_vector, weights2))

#determine which one is closer:
if(result1 > result2):
    closer = "first weight"

if(result2 > result1):
    closer = "second weight"

#print("First: ", result1, "\n", "Second: ", result2, "\n")
print("Closer vector: ", closer)
