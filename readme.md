About
=====

This library includes support for various AI algorithms.

Multilayer Perceptron (MLP)
===========================

The MLP implementation is based primarily on the e-book
"A Brief Introduction to Neural Networks" and includes
support for M-P-Q-N MLPs. There are M inputs, N outputs
and P/Q optional hidden layers. When P/Q are zero then the
neural network is simply a single layer perceptron (SLP).

![MLP](docs/mlp.jpg?raw=true "MLP")

The following diagram shows the MLP backpropagation update
step.

![MLP Update](docs/mlp_update.jpg?raw=true "MLP Update")

Regression Test
---------------

A sample regression test is included which demonstrates how
to use MLPs to approximate a simple nonlinear function
(e.g. y=x*x).

The following diagram shows the output of y=x*x.

![Output of y=x*x](docs/mlp_xx_output.jpg?raw=true "Output of y=x*x")

The following diagram shows the error of y=x*x.

![Error of y=x*x](docs/mlp_xx_error.jpg?raw=true "Error of y=x*x")

The following diagram shows the 1-1-0-1 MLP Diagram of y=x*x.

![1-1-0-1 MLP Diagram of y=x*x](docs/mlp_xx_diagram.jpg?raw=true "1-1-0-1 MLP Diagram of y=x*x")

References
==========

* [A Brief Introduction to Neural Networks](https://www.dkriesel.com/en/science/neural_networks)
* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [Loss Functions and Their Use In Neural Networks](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)
* [Training Hidden Units: The Generalized Delta Rule](https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf)
* [Filters In Convolutional Neural Networks](https://blog.paperspace.com/filters-in-convolutional-neural-networks/)
* [Introduction to Autoencoders](https://www.jeremyjordan.me/autoencoders/)
* [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

License
=======

The AI software was implemented by
[Jeff Boody](mailto:jeffboody@gmail.com)
under The MIT License.

	Copyright (c) 2023 Jeff Boody

	Permission is hereby granted, free of charge, to any person obtaining a
	copy of this software and associated documentation files (the "Software"),
	to deal in the Software without restriction, including without limitation
	the rights to use, copy, modify, merge, publish, distribute, sublicense,
	and/or sell copies of the Software, and to permit persons to whom the
	Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included
	in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
