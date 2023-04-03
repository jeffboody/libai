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
interactive conversation and text-to-image synthesis. Many
different types of neural networks exist such as the
following.

* Fully Connected Neural Networks (FCNN)
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Long Short Term Memory (LSTM)
* Variational Autoencoder Neural Networks (VAE)
* Generative Adversarial Networks (GAN)

[TODO - SUMMARY]

References

* [CS231n Winter 2016](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)

Gradient Descent
----------------

[TODO - Gradient Descent/Stocastic Gradient Descent - SGD]
	- mountain analogy
	- iterative
	- differentiable functions
	- multi-dimensional
	- learning rate (variable)
	---
	- zig-zag pattern
	- local minima
	- saddle point
	- slow convergence
	- mini batch (reduces variance)
	- Centering and Scaling Data
	- Momentum, RMSProp, Adam, etc.
	---
	- Batch Normalization
	- vanishing/exploding gradient
	---
	- overfitting/underfitting
	- regularization
	- weight decay
	- capacity

Several issues related to capacity will be discussed in the
regularization section. However, it's very difficult to know
the amount of capacity required to solve complex problems.
Too little capacity can lead to underfitting and too much
capacity can lead to overfitting. In general, we wish to
have more capacity than is required solve a problem then
rely on regularization techniques to address the overfitting
problem. Regularization techniques cause the neural network
to produce generalized solutions rather than memorizing
training patterns (e.g. patterns not observed in test data)
or learing of an embedded noise signal. In practice, the
capacity of a neural network may be limited by physical
computing resources.

Function Graph
--------------

The function graph is an directed acyclic graph (DAG) that
consists of many nodes, each of which implements a function
in the form of Y = f(X,W) that can solve a fragment of the
larger problem. The inputs and outputs to the functions may
be multi-dimensional arrays known as tensors. The parameters
(W) are trained or learned via the gradient descent
optimization. The functions implemented by each node may be
specialized for solving particular tasks.

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
network to solve more complicated problems. Example layers
include the following.

* Perceptron
* Activation
* Batch Normalization
* Convolution
* Pooling
* Flattening

The following function graph shows a simple neural network
with two inputs X = [x1,x2] (e.g. input layer), two nodes in
the first layer [Node11,Node12] (e.g. hidden layer), two
nodes in the second layer [Node21,Node22] (e.g. output
layer) and two outputs Y = [y1,y2]. The neural network
implements Y = f(X,W) in terms of the node functions
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
predicted output and desired training output so it's not
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
it's useful to compare the perceptron function with the
equation of a line. The perceptron weights are analogous to
the slope of the line while the bias is analogous to the
y-intercept.

	y = m*x + b

The weighted average is also analogous to the dot product
operation between the two vectors which is maximized when
the vectors point in the same direction.

	y = W.X + b = |W|*|X|*cos(theta) + b

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
hidden layers it's recommended to use either ReLU or Tanh.
For the output layer it's recommended to use Linear, Tanh,
Sigmoid or Softmax (classification). The activation function
which should be selected for the output layer may depend on
the desired range of your output. For example, probability
outputs exist in the range 0.0 to 1.0 which makes the
logistic function a good choice.

ReLU (Rectified Linear Unit)

	f(x)  = max(0, x)
	df/dx = 0 for x < 0
	      = 1 for x >= 0
	range = [0,infinity]

PReLU (Parametric Rectified Linear Unit or Leaky ReLU)

	f(x)  = max(a*x, x)
	df/dx = a for x < 0
	      = 1 for x >= 0
	range = [-infinity,infinity]

	a is typically 0.01

Tanh

	f(x)  = tanh(x) = 2/(1 + exp(-2*x)) - 1
	df/dx = 1 - f(x)
	range = [-1,1]

Logistic

	f(x)  = 1/(1 + exp(-x))
	df/dx = f(x)*(1 - f(x))
	range = [0,1]

Linear (Identity)

	f(x)  = x
	df/dx = 1
	range = [-infinity,infinity]

References

* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

Data Centering and Scaling
--------------------------

Data centering and scaling should be performed on the input
layer on a per-channel (i) basis to normalize the data to
zero mean and unit variance. When the input layer contains
images it's common to perform the zero mean but skip the
unit variance. It may also be beneficial to perform data
centering and scaling on a per-image basis rather than
per-channel (e.g. face recognition). Data whitening may also
be applied by performing PCA and transforming the covariance
matrix to the identity matrix.

	Yi = (Xi - Mean(Xi))/StdDev(Xi)

Add a small epsilon to avoid divide-by-zero problems.

This transformation improves the learing/convergence rate by
avoiding the well known zig-zag pattern where the gradient
descent trajectory oscilates back and forth along one
dimension.

References

* [Batch Norm Explained Visually - How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
* [CS231n Winter 2016: Lecture 5: Neural Networks Part 2](https://www.youtube.com/watch?v=gYpoJMlgyXA&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=5)

Parameter Initialization
------------------------

The perceptron weights must be initalized correctly to
ensure that the gradient descent procedure works properly.
Incorrect weight initialization can lead to several problems
including the symmetry problem (e.g. weights initialized to
zero resulting in symmetric partial derivatives), slow
learning and divergence (e.g. the output grows to infinity).

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

The perceptron bias on the other hand are typically
initialized to zero as they are not impacted by the symmetry
problem.

Other parameter types may exist within the neural network
however each may have its own unique initialization
requirements.

References

* [Weight Initialization for Deep Learning Neural Networks](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
* [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
* [Bias Initialization in a Neural Network](https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0)
* [3 Common Problems with Neural Network Initialization](https://towardsdatascience.com/3-common-problems-with-neural-network-initialisation-5e6cacfcd8e6)

Batch Size
----------

The batch size refers to the number of training patterns
that are processed in the forward pass before the loss is
backpropagated. In practice, two approaches are used to
select the batch size including Stochastic Gradient Descent
(SGD) and Mini-Batch Gradient Descent (MBGD). The SGD method
performs a forward pass and backpropagation for each
training pattern. MBGD performs a forward pass using
multiple training patterns followed by a single
backpropagation using the average loss and the average
forward gradients. The advantages of each approach includes
the following.

Stochastic Gradient Descent

* Simplest to implement
* Less memory is required

Mini-Batch Gradient Descent

* Mini-batch is typically the preferred method
* Backpropagation is amortized across the mini-batch
* Smoother gradients results in more stable convergence
* Implementations may vectorize code across mini-batches
* Batch Normalization may further improve convergence

The batch size is a hyperparameter that is selected when
designing the neural network. It was suggested that a good
default mini-batch size is 32. It may also be possible to
increase the batch size over time as this reduces the
variance in the gradients when approaching a minimal
solution.

References

* [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
* [Variable batch-size in mini-batch gradient descent](https://www.reddit.com/r/MachineLearning/comments/481f2v/variable_batchsize_in_minibatch_gradient_descent/)

Batch Normalization
-------------------

Batch Normalization is a neural network layer that is
closely related to data centering and scaling. The main
difference is that Batch Normalization includes a pair of
learnable parameters to scale (gamma) and offset (beta) data
samples. This function is applied on a per-channel or
per-filter basis. It is important to note that the function
is differentiable as is required for backpropagation.

	Yi = gammai*(Xi - Mean(Xi))/StdDev(Xi) + betai

Add a small epsilon to avoid divide-by-zero problems.

The mean and standard deviation are calculated during
training from the mini batch. Running averages of these
values are also calculated during the training which are
subsequently used when making predictions. The exponential
average momentium (e.g. 0.99) is a hyperparameter that is
selected when designing the neural network.

	avg_mean   = avg_mean*momentum + batch_mean*(1 - momentum)
	avg_stddev = avg_stddev*momentum + batch_stddev*(1 - momentum)

Why are the batch mean and standard deviation used during
training rather than the running averages?

Note that the neural network may learn the identity
operation (e.g. beta is the mean and gamma is the inverse of
standard deviation) should this be optimal.

There is some discussion as to the best place for the Batch
Normalization layer. The original paper placed this layer
between the perceptron weighted average and the activation
function, however, more recent results suggest that it's
better to place after the activation function. When placed
per the original paper, the perceptron bias is redundant
with the beta offset.

It was also mentioned that Batch Normalization can be
performed on the input layer in place of data centering and
scaling. However, it's unclear if the the mean and standard
deviation should be used from the training set or mini batch
in this case.

Weight Normalization and Layer Normalization are additional
related techniques however these won't be covered at this
time since it's unclear when or if these techniques are
better.

References

* [Batch Norm Explained Visually - How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
* [Batch normalization: What it is and how to use it](https://www.youtube.com/watch?v=yXOMHOpbon8)
* [CS231n Winter 2016: Lecture 5: Neural Networks Part 2](https://www.youtube.com/watch?v=gYpoJMlgyXA&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=5)
* [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/pdf/1706.05350.pdf)
* [Moving average in Batch Normalization](https://jiafulow.github.io/blog/2021/01/29/moving-average-in-batch-normalization/)

L1/L2 Regularization
--------------------

L1/L2 regularization is a technique that was originally
designed to reduce data overfitting by adding a term to the
loss function which penalizes the variance of the function
parameters. However, it's important to note that L2
regularization (presumably L1 as well) has no regularization
effect when combined with normalization techniques such as
Batch Normalization. However, if L2 regularization is not
used, then the norm of the weights tends to increase over
time. As a result, the effective learning rate decreases.
While this may be a desirable property, it can be difficult
to control and may interfere with explicit attempts to
control the backpropagation learning rate. As such, it seems
best to combine L2 regularization with a normalization
technique.

Loss Function with L1 Regularization

	LR(lambda,W,Y,Yt) = L(Y,Yt) + lambda*SUM(|wi|)
	dLR/dwi = dL/dwi + lambda*wi/|wi|

Loss Function with L2 Regularization

	LR(lambda,W,Y,Yt) = L(Y,Yt) + lambda*SUM(wi^2)
	dLR/dwi = dL/dwi + 2*lambda*wi

Keep in mind that the regularization term affects the
backpropagation procedure by adding an additional term to
the update parameter step since dL/dwi is replaced by
dLR/dwi. Some results also suggest that the regularization
term is not required for the perceptron bias parameter since
it does not seem to impact the final result. The lambda
term is a regularization hyperparameter (0 to 1) that is
selected when designing the neural network.

To explain how regularization works in the absense of
normalization, lets consider how the L2 regularization term
affects the following example.

	X  = [1,1,1,1]
	W1 = [1,0,0,0]
	W2 = [0.25,0.25,0.25,0.25]

The perceptron output is the same for each parameter vector
however the regularization term prefers W2.

	SUM(xi*w1i) == SUM(xi*w2i) = 1.0
	SUM(w1i^2)  = 1.0
	SUM(w2i^2)  = 0.25

The W2 parameters are prefered since they are more
generalized across inputs and therefore reduce the variance
caused by a single outlier.

References

* [CS231n Winter 2016: Lecture 3: Linear Classification 2, Optimization](https://www.youtube.com/watch?v=qlLChbHhbg4&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=3)
* [Regularization in a Neural Network | Dealing with overfitting](https://www.youtube.com/watch?v=EehRcPo1M-Q)
* [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/pdf/1706.05350.pdf)
* [Chapter 8 Training Neural Networks Part 2](https://srdas.github.io/DLBook/ImprovingModelGeneralization.html)

Dropout
-------

Dropout is a regularization technique that may be applied to
a neural network layer where a subset of nodes may be
randomly disabled. Regularization is achieved through the
reduction in capacity which forces the neural network to
increase generalization. The dropout procedure may also be
viewed as selecting a random layer from a large ensemble of
layers that share parameters.

Nodes which have been dropped will not contribute to the
loss during the forward pass and therefore will also not
participate in backpropagation. The dropout probability is a
hyperparameter that is selected when designing the neural
network. However, as a general rule, layers with many
parameters may benefit more from a higher dropout
probability. At least one node must be active.

During training, the output of the layer is attenuated due
to the dropped out nodes. However, during prediction, the
entire set of nodes are used. This results in a change of
scale for the layer output which causes problems for
subsequent layers. The scale can be adjusted during training
by applying the Inverted Dropout procedure where output
nodes are scaled as follows.

	scale = total/active

References

* [CS231n Winter 2016: Lecture 6: Neural Networks Part 3 / Intro to ConvNets](https://www.youtube.com/watch?v=hd_KFJ5ktUc&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=6)

Momentum/RMSProp/Adam
---------------------

[TODO - Momentum]
[TODO - RMSProp]
[TODO - Adam]

References

* [Intro to optimization in deep learning: Momentum, RMSProp and Adam](https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/)

Data Augmentation
-----------------

The size of a training set may be artifically increased
through the use of various data augmentation techniques to
help reduce overfitting and increase generalization.

For example, facial recognition may benefit by flipping
faces horizontally when they are lit from different
directions.

References

* [A Complete Guide to Data Augmentation](https://www.datacamp.com/tutorial/complete-guide-data-augmentation)

Cross Validation
----------------

[TODO - Cross Validation]
