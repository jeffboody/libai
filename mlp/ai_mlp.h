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

#ifndef ai_mlp_H
#define ai_mlp_H

#include "ai_mlpFact.h"

typedef struct ai_mlpLayer_s ai_mlpLayer_t;

#define AI_MLP_STATE_INIT  0
#define AI_MLP_STATE_TRAIN 1
#define AI_MLP_STATE_SOLVE 2
#define AI_MLP_STATE_COUNT 3

typedef struct
{
	int state;

	int m;
	int p;
	int q;
	int n;

	// learning rate
	float rate;

	// in layer
	float* in; // m

	// hidden layer 1 (p > 0)
	ai_mlpLayer_t* h1;

	// hidden layer 2 (p > 0) and (q > 0)
	ai_mlpLayer_t* h2;

	// Omega layer
	ai_mlpLayer_t* Omega;
} ai_mlp_t;

ai_mlp_t* ai_mlp_new(int m, int p, int q, int n,
                     float rate,
                     ai_mlpFact_fn facth,
                     ai_mlpFact_fn dfacth,
                     ai_mlpFact_fn facto,
                     ai_mlpFact_fn dfacto);
void       ai_mlp_delete(ai_mlp_t** _self);
ai_mlp_t*  ai_mlp_import(const char* json,
                         ai_mlpFact_fn default_facth,
                         ai_mlpFact_fn default_dfacth,
                         ai_mlpFact_fn default_facto,
                         ai_mlpFact_fn default_dfacto);
char*      ai_mlp_export(ai_mlp_t* self);
void       ai_mlp_train(ai_mlp_t* self,
                        float* in, float* out);
float*     ai_mlp_solve(ai_mlp_t* self, float* in);
int        ai_mlp_graph(ai_mlp_t* self,
                        const char* label,
                        const char* fname);

#endif
