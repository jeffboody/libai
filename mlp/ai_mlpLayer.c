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
#include "../../libcc/rng/cc_rngNormal.h"
#include "../../libcc/rng/cc_rngUniform.h"
#include "../../libcc/cc_log.h"
#include "../../libcc/cc_memory.h"
#include "ai_mlpLayer.h"

/***********************************************************
* private                                                  *
***********************************************************/

static void
ai_mlpLayer_initXavierWeights(ai_mlpLayer_t* self)
{
	ASSERT(self);

	float min = -1.0/sqrt((double) self->m);
	float max = 1.0/sqrt((double) self->m);

	cc_rngUniform_t rng;
	cc_rngUniform_init(&rng);

	int    m;
	int    n;
	float* w;
	for(n = 0; n < self->n; ++n)
	{
		self->b[n] = cc_rngUniform_rand2F(&rng, min, max);

		w = ai_mlpLayer_weight(self, n);
		for(m = 0; m < self->m; ++m)
		{
			w[m] = cc_rngUniform_rand2F(&rng, min, max);
		}
	}
}

static void
ai_mlpLayer_initHeWeights(ai_mlpLayer_t* self)
{
	ASSERT(self);

	double mu    = 0.0;
	double sigma = sqrt(2.0/((double) self->m));

	cc_rngNormal_t rng;
	cc_rngNormal_init(&rng, mu, sigma);

	int    m;
	int    n;
	float* w;
	for(n = 0; n < self->n; ++n)
	{
		self->b[n] = cc_rngNormal_rand1F(&rng);

		w = ai_mlpLayer_weight(self, n);
		for(m = 0; m < self->m; ++m)
		{
			w[m] = cc_rngNormal_rand1F(&rng);
		}
	}
}

/***********************************************************
* public                                                   *
***********************************************************/

ai_mlpLayer_t* ai_mlpLayer_new(int m, int n,
                               ai_mlpFact_fn fact,
                               ai_mlpFact_fn dfact)
{
	ai_mlpLayer_t* self;
	self = (ai_mlpLayer_t*)
	       CALLOC(1, sizeof(ai_mlpLayer_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->m     = m;
	self->n     = n;
	self->fact  = fact;
	self->dfact = dfact;

	self->b = (float*) CALLOC(n, sizeof(float));
	if(self->b == NULL)
	{
		goto fail_b;
	}

	self->w = (float*) CALLOC(n*m, sizeof(float));
	if(self->w == NULL)
	{
		goto fail_w;
	}

	self->net = (float*) CALLOC(n, sizeof(float));
	if(self->net == NULL)
	{
		goto fail_net;
	}

	self->delta = (float*) CALLOC(n, sizeof(float));
	if(self->delta == NULL)
	{
		goto fail_delta;
	}

	self->o = (float*) CALLOC(n, sizeof(float));
	if(self->o == NULL)
	{
		goto fail_o;
	}

	if((self->fact == ai_mlpFact_ReLU) ||
	   (self->fact == ai_mlpFact_PReLU))
	{
		ai_mlpLayer_initHeWeights(self);
	}
	else
	{
		ai_mlpLayer_initXavierWeights(self);
	}

	// success
	return self;

	// failure
	fail_o:
		FREE(self->delta);
	fail_delta:
		FREE(self->net);
	fail_net:
		FREE(self->w);
	fail_w:
		FREE(self->b);
	fail_b:
		FREE(self);
	return NULL;
}

void ai_mlpLayer_delete(ai_mlpLayer_t** _self)
{
	ASSERT(_self);

	ai_mlpLayer_t* self = *_self;
	if(self)
	{
		FREE(self->b);
		FREE(self->w);
		FREE(self->net);
		FREE(self->delta);
		FREE(self->o);
		FREE(self);
		*_self = NULL;
	}
}

float*
ai_mlpLayer_solve(ai_mlpLayer_t* self, float* in)
{
	ASSERT(self);
	ASSERT(in);

	// update net and activation output
	int    m;
	int    n;
	float* w;
	ai_mlpFact_fn fact = self->fact;
	for(n = 0; n < self->n; ++n)
	{
		self->net[n] = self->b[n];

		w = ai_mlpLayer_weight(self, n);
		for(m = 0; m < self->m; ++m)
		{
			self->net[n] += w[m]*in[m];
		}

		self->o[n] = (*fact)(self->net[n]);
	}

	return self->o;
}

void ai_mlpLayer_updateOmega(float rate,
                             ai_mlpLayer_t* h,
                             ai_mlpLayer_t* Omega,
                             float* out)
{
	ASSERT(h);
	ASSERT(Omega);
	ASSERT(out);

	// h > Omega > out

	int m;
	int n;
	float* w;
	ai_mlpFact_fn dfact = Omega->dfact;
	for(n = 0; n < Omega->n; ++n)
	{
		// update delta for Omega[n]
		Omega->delta[n] = (*dfact)(Omega->net[n])*
		                  (out[n] - Omega->o[n]);

		// update weights for Omega[n]
		// note: h->n == Omega->m
		w = ai_mlpLayer_weight(Omega, n);
		for(m = 0; m < Omega->m; ++m)
		{
			w[m] += rate*h->o[m]*Omega->delta[n];
		}
		Omega->b[n] += rate*Omega->delta[n];
	}
}

void ai_mlpLayer_updateH2(float rate,
                          ai_mlpLayer_t* h1,
                          ai_mlpLayer_t* h2,
                          ai_mlpLayer_t* Omega)
{
	ASSERT(h1);
	ASSERT(h2);
	ASSERT(Omega);

	// h1 > h2 > Omega

	int l;
	int m;
	int n;
	float* w;
	ai_mlpFact_fn dfact = h2->dfact;
	float dfact_net;
	float suml;
	for(n = 0; n < h2->n; ++n)
	{
		// update delta for h2[n]
		h2->delta[n] = 0.0f;
		dfact_net    = (*dfact)(h2->net[n]);
		if(dfact_net != 0.0f)
		{
			// weighted sum of all weights from h2 to Omega
			// which are weighted by the Omega delta
			// note: h2->n == Omega->m
			suml = 0.0f;
			for(l = 0; l < Omega->n; ++l)
			{
				w = ai_mlpLayer_weight(Omega, l);
				suml += Omega->delta[l]*w[n];
			}
			h2->delta[n] = dfact_net*suml;
		}

		// update weights for h2[n]
		// note: h1->n == h2->m
		w = ai_mlpLayer_weight(h2, n);
		for(m = 0; m < h2->m; ++m)
		{
			w[m] += rate*h1->o[m]*h2->delta[n];
		}
		h2->b[n] += rate*h2->delta[n];
	}
}

void ai_mlpLayer_updateH1(float rate,
                          float* in,
                          ai_mlpLayer_t* h1,
                          ai_mlpLayer_t* h)
{
	ASSERT(in);
	ASSERT(h1);
	ASSERT(h);

	// in > h1 > h

	int l;
	int m;
	int n;
	float* w;
	ai_mlpFact_fn dfact = h1->dfact;
	float dfact_net;
	float suml;
	for(n = 0; n < h1->n; ++n)
	{
		// update delta for h1[n]
		h1->delta[n] = 0.0f;
		dfact_net    = (*dfact)(h1->net[n]);
		if(dfact_net != 0.0f)
		{
			// weighted sum of all weights from h1 to h
			// which are weighted by the h delta
			// note: h1->n == h->m
			suml = 0.0f;
			for(l = 0; l < h->n; ++l)
			{
				w = ai_mlpLayer_weight(h, l);
				suml += h->delta[l]*w[n];
			}
			h1->delta[n] = dfact_net*suml;
		}

		// update weights for h1[n]
		// note: in->n == h1->m
		w = ai_mlpLayer_weight(h1, n);
		for(m = 0; m < h1->m; ++m)
		{
			w[m] += rate*in[m]*h1->delta[n];
		}
		h1->b[n] += rate*h1->delta[n];
	}
}

void ai_mlpLayer_updateSLP(float rate,
                           float* in,
                           ai_mlpLayer_t* Omega,
                           float* out)
{
	ASSERT(in);
	ASSERT(Omega);
	ASSERT(out);

	// in > Omega > out

	int m;
	int n;
	float* w;
	ai_mlpFact_fn dfact = Omega->dfact;
	for(n = 0; n < Omega->n; ++n)
	{
		// update delta for Omega[n]
		Omega->delta[n] = (*dfact)(Omega->net[n])*
		                  (out[n] - Omega->o[n]);

		// update weights for Omega[n]
		w = ai_mlpLayer_weight(Omega, n);
		for(m = 0; m < Omega->m; ++m)
		{
			w[m] += rate*in[m]*Omega->delta[n];
		}
		Omega->b[n] += rate*Omega->delta[n];
	}
}

float* ai_mlpLayer_weight(ai_mlpLayer_t* self, int n)
{
	ASSERT(self);

	return &self->w[n*self->m];
}
