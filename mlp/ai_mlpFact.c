/*
 * Copyright (c) 2023 Jeff Boody
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include <math.h>
#include <stdlib.h>

#define LOG_TAG "ai"
#include "../../libcc/cc_log.h"
#include "ai_mlpFact.h"

/***********************************************************
* public                                                   *
***********************************************************/

float ai_mlpFact_linear(float x)
{
	return x;
}

float ai_mlpFact_logistic(float x)
{
	return 1.0f/(1.0f + exp(-x));
}

float ai_mlpFact_ReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.0f;
	}

	return x;
}

float ai_mlpFact_PReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.01f*x;
	}

	return x;
}

float ai_mlpFact_tanh(float x)
{
	return tanhf(x);
}

float ai_mlpFact_dlinear(float x)
{
	return 1.0f;
}

float ai_mlpFact_dlogistic(float x)
{
	float fx = ai_mlpFact_logistic(x);
	return fx*(1.0f - fx);
}

float ai_mlpFact_dReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.0f;
	}

	return 1.0f;
}

float ai_mlpFact_dPReLU(float x)
{
	if(x < 0.0f)
	{
		return 0.01f;
	}

	return 1.0f;
}

float ai_mlpFact_dtanh(float x)
{
	float tanhfx = tanhf(x);
	return 1.0f - tanhfx*tanhfx;
}
