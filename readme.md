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

Convolution Filters
-------------------

The following filters can be applied in the convolution layers of a CNN.

Average Filter

	 1  1  1
	 1  1  1
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

The dilation technique can also be applied to filters to
increase their receptive field and/or increase performance
of large filters. This is achieved by introducing a
dilation stride between filter samples. For example a 3x3
filter with a dilation stride of 2 will cover the same area
as a 5x5 filter.

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

Auto Encoders

* [Introduction to Autoencoders](https://www.jeremyjordan.me/autoencoders/)

Image Segmentation

* [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf)

Image Classification

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
