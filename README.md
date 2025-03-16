# NAVY

## 01: Perceptron
### Task:
Generate random 2D points and predict if they are below or above the line defined by linear function *y=3x+2*. Visualize the solution.

### Solution:
Implemented single perceptor with two imputs and signum activation function. Imput weigts and bias were balanced using product of learning rate, error and input. Prediction was made by sum of products of weights and input, plus bias.

### Visualization:
![Graph](img/01.png)
 - Top left: Generated data and line defined by linear function
 - Top right: Perceptron predictions with decision boundart
 - Bottom left: Evolution of weights and bias in training epochs
 - Bottom right: Evolution of perceptron accuracy in training epochs

## 02: XOR Problem
### Task:
Use a neural network to solve the XOR problem.

### Solution:
Implemented fully connected neural network with sigmoid activation function in every neuron. Solution used for XOR problem contains one hidden layer with two neurons and output layer with one neuron. Prediction was made using forward propagation and using activation function. Learning included prediction for selected input, error calculation using MSE (Mean Square Error) and weights+bias correction. For this, backward propagation was used. It started from output layer and go back into input layer. Weight delta of n-layer neuron is calculated as sum of products n+1-layer weight deltas and weights. This is mutliplayed by derivation of activation function used on perceptron output. 

### Terminal output:
Network architecture and weights+biases before traning
```bash
--------------
Pre-training
--------------
MLP:
hidden:
hidden_p_0; w: [-0.5969314085842015, -0.6628871343538043]; b: 0.07596656650268119
hidden_p_1; w: [-0.3967756514531524, -0.6256754275930128]; b: 0.48861402801188514
output:
output_p_0; w: [0.20103082341774647, 0.39924267645104194]; b: 0.2895867483744847
----------------
```
Traing process with total error
```bash
---------------
Epoch 0, Total Error: 0.538405527622102
Epoch 20000, Total Error: 0.4854308550821145
Epoch 40000, Total Error: 0.1875466359357718
...
Epoch 180000, Total Error: 0.0022758752580269363
---------------
```
Network architecture and weights+biases after traning
```bash
-------------
Post-training
-------------
MLP:
hidden:
hidden_p_0; w: [-5.900735186059314, -5.906617292413846]; b: 2.332701429610526
hidden_p_1; w: [-4.462690174937236, -4.463962292771719]; b: 6.631605875946858
output:
output_p_0; w: [-8.95799152309827, 8.67224598628567]; b: -4.05190095227655
--------------
```
Evaluation of trained model
```bash
---------------------
Results:
Input: [0, 0] -> Predicted: [0.0277] | Expected: [0]
Input: [0, 1] -> Predicted: [0.9703] | Expected: [1]
Input: [1, 0] -> Predicted: [0.9703] | Expected: [1]
Input: [1, 1] -> Predicted: [0.037] | Expected: [0]
---------------------
```
### Visualization:
![Graph](img/02.png)
 - Top left: Evolution of total in training epochs
 - Top right: Evolution of predicted outputs in training epochs
 - Bottom left: Evolution of weights and biases in training epochs
 - Bottom right: Empty

## 03: Hopfield Networks
__TODO__
## 04: Q-learning
__TODO__