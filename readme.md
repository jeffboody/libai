About
=====

This library includes support for various AI algorithms.

Multilayer Perceptron (MLP)
===========================

The following e-book provides an excellent introduction to
neural networks including their relation to biological
neural networks, the components of artificial neural
networks, fundamentals on training neural networks and the
perceptron. Perceptrons are the fundamental building block
that make up neural networks.

[A Brief Introduction to Neural Networks](https://www.dkriesel.com/en/science/neural_networks)

The following sections will expand upon the ideas presented
in the e-book where additional discussion is useful.

Implementation
--------------

This experimental MLP implementation supports M inputs, N
outputs and P/Q optional hidden layers. When P and Q are
zero then the neural network is simply a single layer
perceptron (SLP). The implementation and design are
expected to change significantly as I incorporate support
for more advanced neural network variations.

![MLP](docs/mlp.jpg?raw=true "MLP")

Backpropagation Update
----------------------

The MLP backpropagation update step is fundamental to
training the MLP however I found that the discussion in
section 5.4 of the e-book somewhat confusing with regards to
updating the bias term. This diagram helps to augment that
section by showing the complete set of updates required for
the hidden nodes as well as the output nodes.

![MLP Update](docs/mlp_update.jpg?raw=true "MLP Update")

Weight Initialization
---------------------

Weight initialization is a crucial step to perform
correctly in order to ensure that the MLP training
algorithm works properly. In section 5.7.4, the e-book
simply states that the weights should be initialized with
small randomly choosen values. This simplistic approach
avoids the symmetry problem which occurs when all weights
are initialized to zero and the resulting partial
deriviatives will be the same for every neuron during
backpropagation. The bias weights on the other hand are
typically initialized to zero as they are not impacted by
the symmetry problem. Additional problems which can occur
as the result of incorrect initialization including slow
learning and divergence (e.g. the output grows to infinity).

The following weight initialization methods are recommended
depending on the desired activation function.

Xavier Method

	fact   = tanh or logistic
	m      = number of inputs
	min    = -1/sqrt(m)
	max    = 1/sqrt(m)
	weight = randUniformDistribution(min, max)

Normalized Xavier Method

	fact   = tanh or logistic
	m      = number of inputs
	n      = number of outputs
	min    = -sqrt(6)/sqrt(m + n)
	max    = sqrt(6)/sqrt(m + n)
	weight = randUniformDistribution(min, max)

He Method

	fact   = ReLU or PReLU
	m      = number of inputs
	mu     = 0.0
	sigma  = sqrt(2/m)
	weight = randNormalDistribution(mu, sigma)

Note that the above equations have been altered such that
the number of inputs is m and the number of outputs is n to
match the conventions used elsewhere in this library.

Examples
--------

* [Regression Test](mlp/regression-test/readme.md)

Convolutional Neural Networks (CNN)
===================================

The data within a MLP is represented as a 1D vector. It is
possible to convert a WxHxD image to a 1D vector by
concatenating its pixels however this leads to a couple of
problems. The first problem is that the MLP will have a
untenable number of parameters due to the fact that it
uses a fully connected network. The second problem is that
the spatial relationships between pixels are lost. To
resolve these problems we will need to consider a different
data representation and connection scheme as follows.

The data within a CNN can generally be described as an
N-dimensional tensor (e.g. multi-dimensional arrays). When
the input consists of an image, the tensor will consist of
an WxHxD array. At the input layer the W and H values
correspond to the the image size while the D (depth) value
corresponds to the number of color channels. A CNN may
consist of multiple layers where each layer learns how to
recognize a different feature. Conceptually the early
layers might learn to identify low level features (e.g.
edges) while each subsequent layer might learn higher level
features (e.g. eyes/nose/mouth/face). Each CNN layer is
typically exposed to a larger receptive field which
facilitates learning of higher level features. In practice,
the precise function of each individual CNN layer cannot be
explained as they may be the product of millions of
parameters. The internal CNN layers may also have varying
sizes of tensors depending on the operations applied at
each layer.

The fully connected network can be replaced by sets of
small convolution filters whose values correspond to
trainable weights in the neural network. The convolutional
filters are windows which slide across the CNN
multi-dimensional arrays to compute per-pixel features. The
convolutional filter weights are re-used as the window
slides across the CNN multi-dimensional arrays. As a result
the number of trainable weights is reduced enormously. For
example, a 1000x1000x1 image will have 10^12 ((1000x1000)^2)
weights in a single layer of a fully connected network.
While a CNN layer consisting of 100 3x3x1 convolution
filters will only have 900 (100x3x3) weights. Note that
these calculations disregard the relatively small number of
bias weights.

The CNN multi-dimensional arrays and the convolutional
filters are locally connected which preserves spatial
relationships between pixels.

The next section will describe the convolution operation in
detail.

Convolution Operation
---------------------

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

To explain the convolution operation lets consider the
following 8x8x1 input array which contains a vertical line
feature.

	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0
	0 0 0 0 1 0 0 0

Lets also select a traditional 3x3x1 Sobel vertical edge
filter.

	    -1 0 1
	1/4 -2 0 2
	    -1 0 1

The convolution operation involves sliding the filter
window across the input array and computing the dot product
between the two at each step. The edges of the input array
present a problem with this operation since the filter
window must be centered on the array element to be
computed. To address this problem, either the input array
may be padded such that there are no overhanging elements
from the filter window or the output array may be cropped.

Padded images are said to use "same" padding while unpadded
images are said to use "valid" or "no" padding. Same
padding simply means the output array will have the same
dimensions as the input array. The amount of padding
required for same padding with an MxM filter is (M - 1)/2.
Alternatively, a WxH input array would be cropped to have
a (W - M + 1)x(H - M + 1) output array without padding.

The resulting output matrix with each padding scheme is as
follows.

	8x8x1 Output Array with Same Padding

	    0 0 0 3 0 -3 0 0
	    0 0 0 4 0 -4 0 0
	    0 0 0 4 0 -4 0 0
	    0 0 0 4 0 -4 0 0
	1/4 0 0 0 4 0 -4 0 0
	    0 0 0 4 0 -4 0 0
	    0 0 0 4 0 -4 0 0
	    0 0 0 3 0 -3 0 0

	6x6x1 Output Array with No/Valid Padding

	    0 0 4 0 -4 0
	    0 0 4 0 -4 0
	    0 0 4 0 -4 0
	1/4 0 0 4 0 -4 0
	    0 0 4 0 -4 0
	    0 0 4 0 -4 0

The filter must have the same depth as the input array in
order to compute the dot product acrosss channels.

The output array will have one channel per filter.

Receptive Field
---------------

The receptive field is the region of the sensory area
which can be affected by a stimuli. For the convolution
input array the receptive field is a single element however
the receptive field increases as the result of the
convolution operation (e.g. to 3x3 elements in the previous
example). Each subsequent layer of convolutions will also
increase the receptive field. For example, a sequence of
3x3-3x3-3x3 convolutions will increase the receptive field
from 1 to 3-5-7 elements.

It is advantagous to leverage this effect to increase the
receptive field size by using small convolution filters
because they are faster to compute and will have fewer
parameters to train. A WxH array with MxM filter will
require WxHxMxM operations to compute the dot product. A
7x7 filter will result in 49 operations per pixel while
three 3x3 filters will result in 27 operations per pixel.
This principle is similar to separable filters described in
the appendix.

The dilation technique can also be applied to filters to
increase their receptive field and/or increase performance
of large filters. This is achieved by introducing a
dilation rate (stride) between filter elements. For example
a 3x3 filter with a dilation rate of 2 will cover the same
area as a 5x5 filter.

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
W/H dimensions, increase the receptive field size and also
introduce a slight amount of local translational invariance.
Much like convolution, the pooling operation slides a
window across the input matrix and outputs a value for each
step. For example, a pooling operation consisting of a 2x2
window with a stride of 2 will reduce the width and height
dimensions of the input image by half. Typically the max
pooling function is used which simply outputs the max value
of elements within the pooling window. This max value has
the highest activation to the convolution filter which is
desirable when trying to identify features such as edges.
Other pooling functions may also include the average, min or
other attention based functions.

According to the AlexNex CNN implementation, an overlapping
pooling technique was used to reduce model overfitting by
selecting a larger sliding window size than the stride size.

Dropout
-------

Dropout is a regularization procedure that prevents deep
neural networks from overfitting. Overfitting occurs when
a neural network memorizes the training data and fails to
produce generalized solutions for new data. The fact that
deep neural networks have a very large number of parameters
can make them more susceptible to to the overfitting
problem. The dropout procedure consists of probabilistically
disabling hidden neurons such that they do not contribute to
the forward pass or participate in backpropagation. As a
result, the neural network activates a different subset of
neurons to learn each input which results in more
generalized solutions. The AlexNet CNN implementation uses
a probability of 0.5 for dropout neurons.

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

	     1  2  1 | -1 0 1
	1/4  0  0  0 | -2 0 2
	    -1 -2 -1 | -1 0 1

Prewitt Edge Filter (horizontal/vertical)

	     1  1  1 | -1 0 1
	1/3  0  0  0 | -1 0 1
	    -1 -1 -1 | -1 0 1

Robinson Compass Masks (NW/N/NE/E, SE/S/SW/W)

	    -2 -1  0 | -1 -2 -1 |  0 -1 -2 |  1  0 -1
	1/4 -1  0  1 |  0  0  0 |  1  0 -1 |  2  0 -2
	     0  1  2 |  1  2  1 |  2  1  0 |  1  0 -1

	    2  1  0 |  1  2  1 |  0  1  2 | -1  0  1
	1/4 1  0 -1 |  0  0  0 | -1  0  1 | -2  0  2
	    0 -1 -2 | -1 -2 -1 | -2 -1  0 | -1  0  1

Krisch Compass Masks (NW/N/NE/E, SE/S/SW/W)

	     -3 -3 -3 | -3 -3 -3 | -3 -3 -3 |  5 -3 -3
	1/15 -3  0  5 | -3  0 -3 |  5  0 -3 |  5  0 -3
	     -3  5  5 |  5  0  5 |  5  5 -3 |  5 -3 -3

	      5  5 -3 |  5  5  5 | -3  5  5 | -3 -3  5
	1/15  5  0 -3 | -3  0 -3 | -3  0  5 | -3  0  5
	     -3 -3 -3 | -3 -3 -3 | -3 -3 -3 | -3 -3  5

Canny Edge Detector (edge filter with noise removal)

Laplacian Filter (second order derivative edge filter that highlights regions of rapid change)

	     0 -1  0
	1/4 -1  4 -1
	     0 -1  0

	    -1 -1 -1
	1/8 -1  8 -1
	    -1 -1 -1

Gabor Filters (frequency detection filters often used for texture analysis)

Separable filters are used to describe 2D filters as the
product of two 1D filters. The advantage of 1D filters over
their equivalent 2D filters is computation performance.
Some 2D filters may not be described by a product of 1D
filters. Separable filters are rank-1 matrices. The
rank and 1D filters can be determined from a matrix by
performing singular value decomposition (SVD).

Below are some examples.

	Separable Average Filter

	    1                    1  1  1
	1/3 1 * 1/3 1 1 1 = 1/9  1  1  1
	    1                    1  1  1

	Separable Smoothing Filter

	    1                     1  2  1
	1/4 2 * 1/4 1 2 1 = 1/16  2  4  2
	    1                     1  2  1

	Separable Sobel Edge Filter

	    1                 1  0 -1
	1/4 2 * 1 0 -1 =  1/4 2  0 -2
	    1                 1  0 -1

References
==========

ANN Introduction

* [A Brief Introduction to Neural Networks](https://www.dkriesel.com/en/science/neural_networks)

Bias Update Function

* [Training Hidden Units: The Generalized Delta Rule](https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf)

Initializing Weights

* [Weight Initialization for Deep Learning Neural Networks](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
* [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
* [Bias Initialization in a Neural Network](https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0)
* [3 Common Problems with Neural Network Initialization](https://towardsdatascience.com/3-common-problems-with-neural-network-initialisation-5e6cacfcd8e6)

Backpropagation

* [Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)
* [Backpropagation in a Convolutional Layer](https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509)

Deep Learning

* [Deep Learning](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning)

Activation Functions

* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

Loss Functions

* [Loss Functions and Their Use In Neural Networks](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)

Convolution Filters

* [Filters In Convolutional Neural Networks](https://blog.paperspace.com/filters-in-convolutional-neural-networks/)
* [Convolution Filters / Filters in CNN](https://iq.opengenus.org/convolution-filters/)
* [Gabor Filter](https://en.wikipedia.org/wiki/Gabor_filter)
* [Normal Distribution or Gaussian Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
* [Separable Filters](https://en.wikipedia.org/wiki/Separable_filter)
* [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
* [Separable convolution: Part 2](https://blogs.mathworks.com/steve/2006/11/28/separable-convolution-part-2/)
* [How can I determine if my convolution is separable](https://stackoverflow.com/questions/5886529/how-can-i-determine-if-my-convolution-is-separable)

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
