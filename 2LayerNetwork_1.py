import numpy as np

#This is predictive training. Going into this with the following presuppositions:
# if the inputs are 1.66 and 1.56, then the output should be a 1.
# if the inputs are 2 and 1.5, then the output should be a 0.
# if the predicted value is greater than .5 then it will be a one and if it is less than .5 then it will be a zero


#inputs
#input_vector = np.array([1.66, 1.56]) #the network correctly predicted this one
input_vector = np.array([2, 1.5]) #the network did not correctly predict this one
#weight vector
weights_1 = np.array([1.45, -0.66])

#bias
bias = np.array([0.0])

#sigmoid function will convert all the values passed into it, into a value between 0 and 1.
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#prediction function
def predict(input_vector, weight, bias):
    layer1 = np.dot(input_vector, weights_1) + bias
    layer2 = sigmoid(layer1)
    return layer2

predictedValue = predict(input_vector, weights_1, bias)

print("The prediction is: ", predictedValue, "\n")

#The network correctly guesses the output for one of the inputs but not the other
