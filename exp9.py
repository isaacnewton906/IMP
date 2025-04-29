import numpy as np

# Define step activation function
def step_function(x):
    return 1 if x > 0 else 0

# Perceptron function
def perceptron(inputs, weights, bias):
    total = np.dot(inputs, weights) + bias
    return step_function(total)

# Define inputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Define weights and bias for AND gate
weights_and = np.array([1, 1])
bias_and = -1.5

print("AND Gate Output:")
for x in X:
    print(f"Input: {x} Output: {perceptron(x, weights_and, bias_and)}")

# Define weights and bias for OR gate
weights_or = np.array([1, 1])
bias_or = -0.5

print("\nOR Gate Output:")
for x in X:
    print(f"Input: {x} Output: {perceptron(x, weights_or, bias_or)}")
