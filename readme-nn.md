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
function types include the perceptron, non-linear activation
functions, convolution and pooling.

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

The following diagram shows a simple neural network with two
inputs X = [x1,x2], two nodes in the first layer
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
minimizing a loss function.

	L(Y, Yt)

To minimize the loss function we require the gradient with
respect to the function parameters. However, the loss
function is defined in terms of the predicted output (Y) and
desired training output (Yt). As a result, we must
backpropagate the gradient from loss function to each
function parameter by repeatedly applying the chain rule.
The chain rule allows the desired gradient to be computed by
chaining the partial derivatives of dependent variables. For
example, the chain rule may be applied to the dependent
variables x, y and z as follows.

	dz/dx = (dz/dy)*(dy/dx)

The function parameters may now be updated by applying the
gradient descent optimization where the learning rate
(gamma) is a hyperparameter that is selected when designing
the neural network.

	wi -= gamma*dL/dwi

In summary, the backpropagation procedure may be applied by
repeating the following steps for each training pattern.

1. Make a prediction using a forward pass
2. Evaluate the loss gradient
3. Update function parameters using gradient descent
4. Backpropagate the loss gradient using the chain rule

The following diagram demonstrates the backpropagation
procedure using our example from earlier.

![Neural Network Backpropagation Example](docs/nn-backprop.jpg?raw=true "Neural Network Backpropagation Example")

References

* [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

Gradient Descent
----------------

[TODO - Gradient Descent]
	- local minima
	- vanishing gradient
	- exploding gradient
	- nodes are differentiable
	- learning rate (variable)
	- overtraining/undertraining

Loss Function
-------------

The loss function is a hyperparameter that is selected when
designing the neural network and the choice of loss function
depends upon the type of problem that the neural network is
solving. The two main types of problems are regression and
classification. Regression problems consist of predicting a
real value quantity while classification problems consist of
classifying a pattern in terms of one or more classes.
Recall that the backpropagation procedure also requires the
gradient of the loss function with respect to the predicted
output.

The Mean Squared Error (MSE) and Mean Absolute Error (MAE)
are the most commonly used loss functions for regression
problems. The MSE is typically used unless the training data
has a large number of outliers. This is because the MSE is
highly sensitive to outliers due to the squared term.

	MSE
	L      = (1/n)*SUM((yi - yti)^2)
	dL/dyi = (2/n)*(yi - yti)

	MAE
	L      = (1/n)*SUM(|yi - yti|)
	dL/dyi = (1/n)*(yi - yti)/|yi - yti|

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

Perceptron nodes roughly mimic the functional capabilities
of biological neurons, however, for our purposes it is
sufficient to describe the perceptron in terms of the node
function. As such, the perceptron function performs a
weighted sum of the inputs, plus an additional bias and is
followed by a non-linear activation function.

[TODO - forward pass]

The pseudo-code for the perceptron function is as follows.

	y = b;
	for(i = 0; i < n; ++i)
	{
		y += w[i]*x[i];
	}
	y = fact(y);

[TODO - backpropagation]
[TODO - composite node]

The weights (w) and the bias (b) are parameters that are
learned by the neural network. The activation function on
the other hand is a hyperparameter that is selected when
designing the neural network. Activation functions will be
described in more detail in the next section.

It is useful to note the similarity between the perceptron
and the equation of a line. The perceptron weights are
analogous to the slope of the line while the bias is
analogous to the y-intercept. However, the inclusion of the
activation function enables the perceptron to solve more
complicated non-linear problems.

	y = m*x + b

Many types of problems can be solved by simply connecting
multiple layers of perceptron nodes (e.g. a multi-layer
perceptron or MLP).

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
	f'(x) = 0 for x < 0
	      = 1 for x >= 0

PReLU (Parametric Rectified Linear Unit or Leaky ReLU)

	f(x)  = max(a*x, x)
	f'(x) = a for x < 0
	      = 1 for x >= 0

	a is typically 0.01

Tanh

	f(x)  = tanh(x) = 2/(1 + exp(-2*x)) - 1
	f'(x) = 1 - f(x)

Logistic

	f(x)  = 1/(1 + exp(-x))
	f'(x) = f(x)*(1 - f(x))

Linear (Identity)

	f(x)  = x
	f'(x) = 1

References

* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

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

Regularization
--------------

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
