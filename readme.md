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

Convolutional Neural Networks (CNN)
===================================

Convolution is a well known technique in computer vision
where carefully designed convolution filters are used to
extract an intuitive sets of features. See the Appendix for
examples of traditional convolution filters used by
computer vision. The traditional convolution filters are
fixed and have been carefully designed to select features
to solve specific kinds of problems (e.g. edge detection).
The CNN filters, on the other hand, are learned during the
training algorithm to extract an ideal set of features for
the data set. These learned convolution filters may resemble
traditional filters, but they will more likely consist of
completly novel convolution filters.

Same vs Valid Padding with Convolution
--------------------------------------

The convolution operation processes an input image with
a convolution filter and return an output image. This
output image will have a reduced size due to padding
required for the convolution filter. Given an WxH input
image and an NxM mask the output will be an
(W - N + 1)x(H - M + 1) image.

Optionally the input image may be padded (e.g. zeros or
clamp-to-edge) such that the output image size remains the
same as the input image. The amount of padding required is
p = ((N - 1)/2, (M -1)/2).

Padded images are said to use "same" padding while unpadded
images are said to use "valid" or "no" padding.

Dilated Convolution
-------------------

The dilation technique can also be applied to filters to
increase their receptive field and/or increase performance
of large filters. This is achieved by introducing a
dilation rate between filter samples. For example a 3x3
filter with a dilation rate of 2 will cover the same area
as a 5x5 filter.

1x1 Convolution (Network-in-Network)
------------------------------------

The 1x1 convolution operation is simply a convolution
operation that operates on the depth component of an WxHxD
matrix. The size of the convolution filter is actually
1x1xD. The result of an indivitual 1x1 convolution
operation is a WxHx1 matrix. A collection of C 1x1xD
filters can be applied to a WxHxD matrix which results in
a WxHxC matrix. As a result, the 1x1 convolution can be
used for dimensionality reduction (e.g. C < D) and for
dimensionality expansion (e.g. C > D). The dimensionality
reduction is a useful technique to improve performance of
subsequent operations and can significantly reduce the
number of parameters in the model.

One trivial example of a dimensionality reduction is the
conversion of an RGB image to a grayscale image where an
WxHx3 image is converted to a WxHx1 image with a 1x1x3
averaging filter (1/3, 1/3, 1/3).

A "Bottle-Neck layer" is one which includes a sequence of
1x1, MxN and 1x1 convolution filters. The initial 1x1
convolution filter reduces the depth of the WxHxD matrix
prior to performing a computationally expensive operation.
The final 1x1 convolution filter can be used to expand the
output image depth back to WxHxD.

ReLU Activation Function
------------------------

The ReLU activation function is typically applied on a
per-element basis following the convolution operation.

According to the AlexNet CNN implementation, the ReLU
activation function can be used to train deep CNNs much
faster than saturating activation functions (e.g. tanh or
sigmoid).

It is also useful to note that many convolution operations
are designed to have a positive and negative response to
their input (e.g. Sobel Edge Filter). By discarding the
negative response with the ReLU activation function we can
eliminate the redundant "ghost edges" associated with the
negative response.

Pooling Layer
-------------

A pooling layer may be used to reduce dimensionality in the
W/H dimensions and also introduce a slight amount of local
translational invariance. Much like convolution, the pooling
operation slides a window across the input matrix and
outputs a value for each step. For example, a pooling
operation consisting of a 2x2 window with a stride of 2
will reduce the width and height dimensions of the input
image by half. Typically the max pooling function is used
which simply outputs the max value of elements within the
pooling window. This max value has the highest activation to
the convolution filter which is desirable when trying to
identify features such as edges. Other pooling functions may
also include the average, min or other attention based
functions.

According to the AlexNex CNN implementation, an overlapping
pooling technique was used to reduce model overfitting by
selecting a larger sliding window size than the stride size.

Appendix
========

Traditional Convolution Filters
-------------------------------

The following are examples of traditional convolutional
filters used in computer vision.

Average Filter

	    1  1  1
	1/9 1  1  1
	    1  1  1

Gaussian Blur (coefficients selected from a normal distribution with desired standard deviation and mean)

	       1  4  7  4  1
	       4 16 26 16  4
	1/273  7 26 41 26  7
	       4 16 26 16  4
	       1  4  7  4  1

Sobel Edge Filter (horizontal/vertical)

	-1 0 1 |  1  2  1
	-2 0 2 |  0  0  0
	-1 0 1 | -1 -2 -1

Prewitt Edge Filter (horizontal/vertical)

	-1 0 1 |  1  1  1
	-1 0 1 |  0  0  0
	-1 0 1 | -1 -1 -1

Robinson Compass Masks (NW/N/NE/E, SE/S/SW/W)

	-2 -1  0 | -1 -2 -1 |  0 -1 -2 |  1  0 -1
	-1  0  1 |  0  0  0 |  1  0 -1 |  2  0 -2
	 0  1  2 |  1  2  1 |  2  1  0 |  1  0 -1

	 2  1  0 |  1  2  1 |  0  1  2 | -1  0  1
	 1  0 -1 |  0  0  0 | -1  0  1 | -2  0  2
	 0 -1 -2 | -1 -2 -1 | -2 -1  0 | -1  0  1

Krisch Compass Masks (NW/N/NE/E, SE/S/SW/W)

	-3 -3 -3 | -3 -3 -3 | -3 -3 -3 |  5 -3 -3
	-3  0  5 | -3  0 -3 |  5  0 -3 |  5  0 -3
	-3  5  5 |  5  0  5 |  5  5 -3 |  5 -3 -3

	 5  5 -3 |  5  5  5 | -3  5  5 | -3 -3  5
	 5  0 -3 | -3  0 -3 | -3  0  5 | -3  0  5
	-3 -3 -3 | -3 -3 -3 | -3 -3 -3 | -3 -3  5

Canny Edge Detector (edge filter with noise removal)

Laplacian Filter (second order derivative edge filter that highlights regions of rapid change)

	 0 -1  0
	-1  4 -1
	 0 -1  0

	-1 -1 -1
	-1  8 -1
	-1 -1 -1

Gabor Filters (frequency detection filters often used for texture analysis)

Separable filters are used to describe 2D filters as the
product of two 1D filters. The advantage of 1D filters over
their equivalent 2D filters is computation performance.
Some 2D filters may not be described by a product of 1D
filters. Below are some examples.

	Separable Average Filter

	    1                    1  1  1
    1/3 1 * 1/3 1 1 1 = 1/9  1  1  1
        1                    1  1  1

	Separable Smoothing Filter

	    1                     1  2  1
    1/4 2 * 1/4 1 2 1 = 1/16  2  4  2
        1                     1  2  1

	Separable Sobel Edge Filter

	    1             1  0 -1
        2 * 1 0 -1 =  2  0 -2
        1             1  0 -1

References
==========

ANN Introduction

* [A Brief Introduction to Neural Networks](https://www.dkriesel.com/en/science/neural_networks)

Bias Update Function

* [Training Hidden Units: The Generalized Delta Rule](https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf)

Activation Functions

* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

Loss Functions

* [Loss Functions and Their Use In Neural Networks](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)

CNN Filters

* [Filters In Convolutional Neural Networks](https://blog.paperspace.com/filters-in-convolutional-neural-networks/)
* [Convolution Filters / Filters in CNN](https://iq.opengenus.org/convolution-filters/)
* [Gabor Filter](https://en.wikipedia.org/wiki/Gabor_filter)
* [Normal Distribution or Gaussian Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
* [Separable Filters](https://en.wikipedia.org/wiki/Separable_filter)
* [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector)

Convolution Padding

* [How Padding helps in CNN](https://www.numpyninja.com/post/how-padding-helps-in-cnn)

1x1 Convolution

* [Networks in Networks and 1x1 Convolutions](https://www.coursera.org/lecture/convolutional-neural-networks/networks-in-networks-and-1x1-convolutions-ZTb8x)
* [Talented Mr. 1X1: Comprehensive look at 1X1 Convolution in Deep Learning](https://medium.com/analytics-vidhya/talented-mr-1x1-comprehensive-look-at-1x1-convolution-in-deep-learning-f6b355825578)

Pooling Layer

* [What is Pooling in a Convolutional Neural Network (CNN): Pooling Layers Explained](https://programmathically.com/what-is-pooling-in-a-convolutional-neural-network-cnn-pooling-layers-explained/)

Auto Encoders

* [Introduction to Autoencoders](https://www.jeremyjordan.me/autoencoders/)

Image Segmentation

* [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf)

Image Classification

* [Concept of AlexNet: Convolutional Neural Network](https://medium.com/analytics-vidhya/concept-of-alexnet-convolutional-neural-network-6e73b4f9ee30)
* [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

Image Inpainting

* [Generative Image Inpainting with Contextual Attention](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Generative_Image_Inpainting_CVPR_2018_paper.pdf)
* [Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)

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
