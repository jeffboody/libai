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
#include <string.h>

#define LOG_TAG "ai"
#include "../../libcc/cc_log.h"
#include "ai_mlpFact.h"

const char* AI_MLP_FACT_STRING_CUSTOM    = "custom";
const char* AI_MLP_FACT_STRING_LINEAR    = "linear";
const char* AI_MLP_FACT_STRING_LOGISTIC  = "logistic";
const char* AI_MLP_FACT_STRING_RELU      = "ReLU";
const char* AI_MLP_FACT_STRING_PRELU     = "PReLU";
const char* AI_MLP_FACT_STRING_TANH      = "tanh";
const char* AI_MLP_FACT_STRING_DLINEAR   = "dlinear";
const char* AI_MLP_FACT_STRING_DLOGISTIC = "dlogistic";
const char* AI_MLP_FACT_STRING_DRELU     = "dReLU";
const char* AI_MLP_FACT_STRING_DPRELU    = "dPReLU";
const char* AI_MLP_FACT_STRING_DTANH     = "dtanh";

/***********************************************************
* public                                                   *
***********************************************************/

const char* ai_mlpFact_string(ai_mlpFact_fn fact)
{
	// fact may be NULL

	if(fact == ai_mlpFact_linear)
	{
		return AI_MLP_FACT_STRING_LINEAR;
	}
	else if(fact == ai_mlpFact_logistic)
	{
		return AI_MLP_FACT_STRING_LOGISTIC;
	}
	else if(fact == ai_mlpFact_ReLU)
	{
		return AI_MLP_FACT_STRING_RELU;
	}
	else if(fact == ai_mlpFact_PReLU)
	{
		return AI_MLP_FACT_STRING_PRELU;
	}
	else if(fact == ai_mlpFact_tanh)
	{
		return AI_MLP_FACT_STRING_TANH;
	}
	else if(fact == ai_mlpFact_dlinear)
	{
		return AI_MLP_FACT_STRING_DLINEAR;
	}
	else if(fact == ai_mlpFact_dlogistic)
	{
		return AI_MLP_FACT_STRING_DLOGISTIC;
	}
	else if(fact == ai_mlpFact_dReLU)
	{
		return AI_MLP_FACT_STRING_DRELU;
	}
	else if(fact == ai_mlpFact_dPReLU)
	{
		return AI_MLP_FACT_STRING_DPRELU;
	}
	else if(fact == ai_mlpFact_dtanh)
	{
		return AI_MLP_FACT_STRING_DTANH;
	}

	return AI_MLP_FACT_STRING_CUSTOM;
}

ai_mlpFact_fn ai_mlpFact_function(const char* str)
{
	ASSERT(str);

	if(strcmp(str, AI_MLP_FACT_STRING_LINEAR) == 0)
	{
		return ai_mlpFact_linear;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_LOGISTIC) == 0)
	{
		return ai_mlpFact_logistic;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_RELU) == 0)
	{
		return ai_mlpFact_ReLU;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_PRELU) == 0)
	{
		return ai_mlpFact_PReLU;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_TANH) == 0)
	{
		return ai_mlpFact_tanh;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_DLINEAR) == 0)
	{
		return ai_mlpFact_dlinear;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_DLOGISTIC) == 0)
	{
		return ai_mlpFact_dlogistic;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_DRELU) == 0)
	{
		return ai_mlpFact_dReLU;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_DPRELU) == 0)
	{
		return ai_mlpFact_dPReLU;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_DTANH) == 0)
	{
		return ai_mlpFact_dtanh;
	}
	else if(strcmp(str, AI_MLP_FACT_STRING_CUSTOM) == 0)
	{
		return NULL;
	}

	LOGE("invalid %s", str);
	return NULL;
}

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
