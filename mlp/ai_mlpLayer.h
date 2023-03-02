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

#ifndef ai_mlpLayer_H
#define ai_mlpLayer_H

#include "../../libcc/rng/cc_rngNormal.h"
#include "../../libcc/rng/cc_rngUniform.h"
#include "ai_mlpFact.h"

typedef struct ai_mlpLayer_s
{
	int m;
	int n;

	float* b;     // n
	float* w;     // nxm (stride is m)
	float* net;   // n
	float* delta; // n
	float* o;     // n

	ai_mlpFact_fn  fact;
	ai_mlpFact_fn  dfact;
} ai_mlpLayer_t;

ai_mlpLayer_t* ai_mlpLayer_new(int m, int n,
                               cc_rngUniform_t* rng_uniform,
                               cc_rngNormal_t* rng_normal,
                               ai_mlpFact_fn fact,
                               ai_mlpFact_fn dfact);
void           ai_mlpLayer_delete(ai_mlpLayer_t** _self);
float*         ai_mlpLayer_solve(ai_mlpLayer_t* self,
                                 float* in);
void           ai_mlpLayer_updateOmega(float rate,
                                       ai_mlpLayer_t* h,
                                       ai_mlpLayer_t* Omega,
                                       float* out);
void           ai_mlpLayer_updateH2(float rate,
                                    ai_mlpLayer_t* h1,
                                    ai_mlpLayer_t* h2,
                                    ai_mlpLayer_t* Omega);
void           ai_mlpLayer_updateH1(float rate,
                                    float* in,
                                    ai_mlpLayer_t* h1,
                                    ai_mlpLayer_t* h);
void           ai_mlpLayer_updateSLP(float rate,
                                     float* in,
                                     ai_mlpLayer_t* Omega,
                                     float* out);
float*         ai_mlpLayer_weight(ai_mlpLayer_t* self,
                                  int n);

#endif
