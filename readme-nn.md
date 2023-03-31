Neural Network
==============

Introduction
------------

An artificial neural network is an algorithm that learns to
solve complex problems by performing a gradient descent
optimization across a function graph. Neural networks may be
applied to solve a wide range of problems such as linear
regression, non-linear prediction, classification,
segmentation, noise removal, natural language translation,
interactive conversation and text-to-image synthesis. There
exists many different types of neural networks which have
been specifically designed to handle this wide range
problems. Some notable examples include fully connected
neural networks (FCNN), convolutional neural networks (CNN),
recurrent neural networks (RNN), long short term memory
(LSTM), variational autoencoder neural networks (VAE) and
generative adversarial networks (GAN).

[TODO - SUMMARY]

References

* [CS231n Winter 2016](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)

Function Graph
--------------

The function graph is an directed acyclic graph (DAG) that
consists of many nodes, each of which implements a function
in the form of Y = f(X,W) that can solve a fragment of the
larger problem. The inputs and outputs to the functions may
be multi-dimensional arrays known as tensors. The parameters
(W) are trained or learned via the gradient descent
optimization. The functions implemented by each node may be
specialized for solving particular tasks. Some example
function types include the perceptron, activation functions,
convolution and pooling.

Nodes are typically organized into layers of similar
functions where the output of one layer is fed into the
input of the next layer. Early neural network architectures
were fully connected such that every output of one layer
was connected to every input of the next layer. In 2012, a
major innovation was introduced by the AlexNet architecture
which demonstrated how to use sparsely connected layers of
convolutional nodes to greatly improve image classification
tasks. As the number of layers (e.g. the neural network
depth) and nodes increases, so does the capacity of a neural
network to solve more complicated problems.

The following function graph shows a simple neural network
with two inputs X = [x1,x2], two nodes in the first layer
[Node11,Node12], two nodes in the second layer
[Node21,Node22] and two outputs Y = [y1,y2]. The neural
network implements Y = f(X,W) in terms of the node functions
f = [f1,f2] and the parameters W = [W11,W12,W21,W22]. Each
parameter variable may represent an array with zero or more
elements.

![Neural Network Example](docs/nn.jpg?raw=true "Neural Network Example")

Forward Pass
------------

A forward pass is performed on the neural network to make a
prediction given some input and simply involves evaluating
functions in the function graph from the input to the
output.

Backpropagation
---------------

The backpropagation procedure implements the gradient
descent optimization to learn the function parameters by
minimizing a loss function with respect to the predicted
output (Y) and the desired training output (Yt).

	L(Y,Yt)

The gradient descent opmization states that the function
parameters may be updated to minimize the loss by
subtracting the gradient of the loss with respect to each
function parameter. The learning rate (gamma) is a
hyperparameter that is selected when designing the neural
network.

	wi -= gamma*dL/dwi

Recall that the loss function is defined in terms of the
predicted output and desired training output so it is not
possible to compute the desired gradient directly. As a
result, we must backpropagate the gradient from loss
function to each function parameter by repeatedly applying
the chain rule. The chain rule allows the desired gradient
to be computed by chaining the gradients of dependent
variables. For example, the chain rule may be applied to the
dependent variables x, y and z as follows.

	dz/dx = (dz/dy)*(dy/dx)

The following gradients may be computed during the forward
pass (i.e. the forward gradients) which will be cached for
use by the backpropagation procedure.

	dy/dxi = df(X,W)/dxi
	dy/dwi = df(X,W)/dwi

When a node is connected to more than one output node, we
must combine the backpropagated loss gradient.

	dL/dy = SUM(dLi/dy)

The update gradient may now be determined using the loss
gradient, the forward gradients and the chain rule.

	dL/dwi = (dL/dy)*(dy/dwi)

The backpropagated gradient may also be determined using the
loss gradient, the forward gradients and the chain rule.

	dL/dxi = (dL/dy)*(dy/dxi)

In summary, the backpropagation procedure may be applied by
repeating the following steps for each training pattern.

* Forward Pass
* Forward Gradients
* Compute Loss
* Combine Loss
* Update Parameters
* Backpropagate Loss

The following function graph shows the backpropagation
procedure using our example from earlier.

![Neural Network Backpropagation](docs/nn-backprop.jpg?raw=true "Neural Network Backpropagation")

References

* [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

Loss Function
-------------

The loss function is a hyperparameter that is selected when
designing the neural network and the choice of loss function
depends upon the type of problem that the neural network is
solving. The two main types of problems are regression and
classification. Regression problems consist of predicting a
real value quantity while classification problems consist of
classifying a pattern in terms of one or more classes.

The Mean Squared Error (MSE) and Mean Absolute Error (MAE)
are the most commonly used loss functions for regression
problems. The MSE is typically used unless the training data
has a large number of outliers. This is because the MSE is
highly sensitive to outliers due to the squared term.

	MSE
	L(Y,Yt) = (1/n)*SUM((yi - yti)^2)
	dL/dyi  = (2/n)*(yi - yti)

	MAE
	L(Y,Yt) = (1/n)*SUM(|yi - yti|)
	dL/dyi  = (1/n)*(yi - yti)/|yi - yti|

The Categorical Cross Entropy Loss is the most commonly used
loss function for classification problems. Additionally, the
Variational Autoencoder Loss is often used for autoencoder
neural networks.

References

* [Loss Functions and Their Use In Neural Networks](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)
* [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)
* [Derivative Calculator](https://www.derivative-calculator.net/)

Perceptron
----------

The perceptron is the main type of node which is used by
neural networks and implements a function that roughly
mimics biological neurons. This function consists of a
weighted sum of inputs plus a bias term followed by an
activation function.

	W      = [[w1,w2,...,wn],b]
	f(X,W) = fact(SUM(xi*wi) + b)

The weights (w) and the bias (b) are the parameters that
are learned by the neural network while the activation
function is a hyperparameter that is selected when
designing the neural network.

The following function graph shows the perceptron which can
most easily be visualized as a compound node.

![Perceptron](docs/nn-perceptron.jpg?raw=true "Perceptron")

The following function graph shows the backpropagation
procedure for the perceptron node. Note that the activation
function does not include any function parameters so the
update step may be skipped. The perceptron node
implementation may also choose to combine the compound node
into a single node by simply substituting equations.

![Perceptron Backpropagation](docs/nn-perceptron-backprop.jpg?raw=true "Perceptron Backpropagation")

To gain a better understanding of how the perceptron works
it is useful to compare the perceptron function with the
equation of a line. The perceptron weights are analogous to
the slope of the line while the bias is analogous to the
y-intercept.

	y = m*x + b

From the biological perspective, a neuron may activate at
different strength depending on if some threshold was
exceeded. The activation function is generally designed to
mimic this behavior, however, in practice the designer may
choose any function desired to achieve a particular effect.
For example, an activation function may be selected to model
a non-linear operation, to threshold outputs or to predict a
probability.

References

* [A Brief Introduction to Neural Networks](https://www.dkriesel.com/en/science/neural_networks)

Activation Functions
--------------------

The following activation functions and their derivatives
may be used depending on the situation. The hidden layers
typically use one activation function and while the output
layer may use a different activation function. For the
hidden layers it is recommended to use either ReLU or Tanh.
For the output layer it is recommended to use Linear, Tanh,
Sigmoid or Softmax (classification). The activation function
which should be selected for the output layer may depend on
the desired range of your output. For example, probability
outputs exist in the range 0.0 to 1.0 which makes the
logistic function a good choice.

ReLU (Rectified Linear Unit)

	f(x)  = max(0, x)
	df/dx = 0 for x < 0
	      = 1 for x >= 0

PReLU (Parametric Rectified Linear Unit or Leaky ReLU)

	f(x)  = max(a*x, x)
	df/dx = a for x < 0
	      = 1 for x >= 0

	a is typically 0.01

Tanh

	f(x)  = tanh(x) = 2/(1 + exp(-2*x)) - 1
	df/dx = 1 - f(x)

Logistic

	f(x)  = 1/(1 + exp(-x))
	df/dx = f(x)*(1 - f(x))

Linear (Identity)

	f(x)  = x
	df/dx = 1

References

* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

Gradient Descent
----------------

[TODO - Gradient Descent]
	- local minima
	- vanishing gradient
	- exploding gradient
	- nodes are differentiable
	- learning rate (variable)
	- overtraining/undertraining
	- regularization

[TODO - Regularization]

Several issues related to capacity will be discussed in the
regularization section. However, it is very difficult to
know the amount of capacity required to solve complex
problems. Too little capacity can lead to underfitting and
too much capacity can lead to overfitting. In general, we
wish to have more capacity than is required solve a problem
then rely on regularization techniques to address the
overfitting problem. Regularization techniques cause the
neural network to produce generalized solutions rather than
memorizing training patterns (e.g. patterns not observed in
test data) or learing of an embedded noise signal. In
practice, the capacity of a neural network may be limited by
physical computing resources.

Parameter Initialization
------------------------

The perceptron weights must be initalized correctly to
ensure that the gradient descent procedure works properly.
Incorrect weight initialization can lead to several problems
including the symmetry problem (e.g. weights initialized to
zero resulting in symmetric partial derivatives), slow
learning and divergence (e.g. the output grows to infinity).

The perceptron bias on the other hand are typically
initialized to zero as they are not impacted by the symmetry
problem.

The following perceptron weight initialization methods are
recommended depending on the desired activation function.

Xavier Method

	fact = tanh or logistic
	m    = number of inputs
	min  = -1/sqrt(m)
	max  = 1/sqrt(m)
	w    = randUniformDistribution(min, max)

Normalized Xavier Method

	fact = tanh or logistic
	m    = number of inputs
	n    = number of outputs
	min  = -sqrt(6)/sqrt(m + n)
	max  = sqrt(6)/sqrt(m + n)
	w    = randUniformDistribution(min, max)

He Method

	fact  = ReLU or PReLU
	m     = number of inputs
	mu    = 0.0
	sigma = sqrt(2/m)
	w     = randNormalDistribution(mu, sigma)

Other parameter types may exist within the neural network
however each may have its own unique initialization
requirements.

References

* [Weight Initialization for Deep Learning Neural Networks](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
* [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
* [Bias Initialization in a Neural Network](https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0)
* [3 Common Problems with Neural Network Initialization](https://towardsdatascience.com/3-common-problems-with-neural-network-initialisation-5e6cacfcd8e6)

L1/L2 Regularization
--------------------

[TODO - L1/L2 Regularization]

	- backpropagation

References

* [Chapter 8 Training Neural Networks Part 2](https://srdas.github.io/DLBook/ImprovingModelGeneralization.html)

Data Centering
--------------

[TODO - Data Centering]

	- zero mean
	- unit stddev
