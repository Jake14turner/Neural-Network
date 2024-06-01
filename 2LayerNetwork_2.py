#in the first file "2LayerNetwork_1.py", the network did not corrctly predict the outcome for the second input
#in this file, we will train the network to predict correctly based on the input. Same presuppositions as last time.

#in order to train the model we will use gradient descent as well as backpropogation algorithms to adjust the weights

#the first step is to compute the error (the difference between the expected response and the given response)
#we are able to do this because we are using supervised learning
















import numpy as np

#inputs
#input_vector = np.array([1.66, 1.56]) 
input_vector = np.array([2, 1.5]) 

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


#                       ///////////       Calculating the error       ///////////

#cost/loss function using Mean Standard Deviation
#compute the difference between expected output and given output and then square it.

#expected output
target = 0

mse = np.square(predictedValue - target)
print("Prediction: ", predictedValue, "\nError: ", mse)


#                       ///////////       Reducing the error       ///////////

#we are going to calculate the derivative in order to see whether we need to go up or down in value within the weights
derivative = 2 * (predictedValue - target)
print("\nThe derivative is: ", derivative)

#update the weights based on whether the derivative is negative or positive
weights_1 = weights_1 - derivative

predictedValue = predict(input_vector, weights_1, bias)

error = np.square(predictedValue - target)

print("\nThe predicted value is: ", predictedValue)
print("\nThe new error is: ", error)

#in this situation, the error went to almost 0




#                       ///////////       Backpropogation       ///////////

# we are going to use the chain rule to calculate the 

#take the derivative of the error function with respect to the bias. its a function composition
#error(prediction(layer1(bias)))

# /// derivative for the bias ///

#first derivative
derror_dprediction = 2 * (predictedValue - target)
layer_1 = np.dot(input_vector, weights_1) + bias
#second derivative
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))
dprediction_dlayer1 = sigmoid_deriv(layer_1)
#third derivative
dlayer1_dbias = 1

#multiply all the derivatives together for the final derivative
derror_dbias = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
)



# /// derivative for the weights ///

#first derivative 
derror_dprediction = 2 * (predictedValue - target)

#second derivative
dprediction_dlayer1 = sigmoid_deriv(layer_1)

#third derivative
dlayer1_dweights = (0 * weights_1) + (1 * input_vector)

#multiply all derivatives together

derror_dweights = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
)


