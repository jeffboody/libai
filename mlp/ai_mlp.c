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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LOG_TAG "ai"
#include "../../libcc/cc_log.h"
#include "../../libcc/cc_memory.h"
#include "ai_mlpLayer.h"
#include "ai_mlp.h"

/***********************************************************
* public                                                   *
***********************************************************/

ai_mlp_t*
ai_mlp_new(int m, int p, int q, int n, float rate,
           ai_mlpFact_fn facth,
           ai_mlpFact_fn dfacth,
           ai_mlpFact_fn facto,
           ai_mlpFact_fn dfacto)
{
	ai_mlp_t* self;
	self = (ai_mlp_t*) CALLOC(1, sizeof(ai_mlp_t));
	if(self == NULL)
	{
		LOGE("CALLOC failed");
		return NULL;
	}

	self->m    = m;
	self->p    = p;
	self->q    = q;
	self->n    = n;
	self->rate = rate;

	// initialize seed for rand
	srand(time(NULL));

	if(p && q)
	{
		self->h1 = ai_mlpLayer_new(m, p, facth, dfacth);
		if(self->h1 == NULL)
		{
			goto fail_h1;
		}

		self->h2 = ai_mlpLayer_new(p, q, facth, dfacth);
		if(self->h2 == NULL)
		{
			goto fail_h2;
		}

		self->Omega = ai_mlpLayer_new(q, n, facto, dfacto);
		if(self->Omega == NULL)
		{
			goto fail_Omega;
		}
	}
	else if(p)
	{
		self->h1 = ai_mlpLayer_new(m, p, facth, dfacth);
		if(self->h1 == NULL)
		{
			goto fail_h1;
		}

		self->Omega = ai_mlpLayer_new(p, n, facto, dfacto);
		if(self->Omega == NULL)
		{
			goto fail_Omega;
		}
	}
	else
	{
		self->Omega = ai_mlpLayer_new(m, n, facto, dfacto);
		if(self->Omega == NULL)
		{
			goto fail_Omega;
		}
	}

	// success
	return self;

	// failure
	fail_Omega:
		ai_mlpLayer_delete(&self->h2);
	fail_h2:
		ai_mlpLayer_delete(&self->h1);
	fail_h1:
		FREE(self);
	return NULL;
}

void ai_mlp_delete(ai_mlp_t** _self)
{
	ASSERT(_self);

	ai_mlp_t* self = *_self;
	if(self)
	{
		ai_mlpLayer_delete(&self->Omega);
		ai_mlpLayer_delete(&self->h2);
		ai_mlpLayer_delete(&self->h1);
		FREE(self);
		*_self = NULL;
	}
}

void ai_mlp_train(ai_mlp_t* self,
                  float* in, float* out)
{
	ASSERT(self);
	ASSERT(in);
	ASSERT(out);

	// initialize solution
	ai_mlp_solve(self, in);

	// back-propagation update
	if(self->h1 && self->h2)
	{
		ai_mlpLayer_updateOmega(self->rate, self->h2,
		                        self->Omega, out);
		ai_mlpLayer_updateH2(self->rate, self->h1,
		                     self->h2, self->Omega);
		ai_mlpLayer_updateH1(self->rate, in, self->h1,
		                     self->h2);
	}
	else if(self->h1)
	{
		ai_mlpLayer_updateOmega(self->rate, self->h1,
		                        self->Omega, out);
		ai_mlpLayer_updateH1(self->rate, in, self->h1,
		                     self->Omega);
	}
	else
	{
		ai_mlpLayer_updateSLP(self->rate, in,
		                      self->Omega, out);
	}
}

float* ai_mlp_solve(ai_mlp_t* self, float* in)
{
	ASSERT(self);
	ASSERT(in);

	float* o1;
	float* o2;
	float* oo = NULL;
	if(self->h1 && self->h2)
	{
		o1 = ai_mlpLayer_solve(self->h1, in);
		o2 = ai_mlpLayer_solve(self->h2, o1);
		oo = ai_mlpLayer_solve(self->Omega, o2);
	}
	else if(self->h1)
	{
		o1 = ai_mlpLayer_solve(self->h1, in);
		oo = ai_mlpLayer_solve(self->Omega, o1);
	}
	else
	{
		oo = ai_mlpLayer_solve(self->Omega, in);
	}

	return oo;
}

int ai_mlp_graph(ai_mlp_t* self, float* in,
                 const char* label, const char* fname)
{
	ASSERT(self);
	ASSERT(label);
	ASSERT(fname);

	FILE* f = fopen(fname, "w");
	if(f == NULL)
	{
		LOGE("fopen failed");
		return 0;
	}

	// header
	fprintf(f, "// sudo apt-get install graphviz\n");
	fprintf(f, "// xdot %s\n", fname);
	fprintf(f, "%s\n", "digraph MLP");
	fprintf(f, "%s\n", "{");
	fprintf(f, "\tlabel=\"%s\";\n", label);
	fprintf(f, "\t%s\n", "fontsize=20;");
	fprintf(f, "\t%s\n", "size=\"2,1\";");
	fprintf(f, "\t%s\n", "ratio=fill;");

	ai_mlpLayer_t* h1    = self->h1;
	ai_mlpLayer_t* h2    = self->h2;
	ai_mlpLayer_t* Omega = self->Omega;

	// in nodes
	int m;
	int n;
	for(m = 0; m < self->m; ++m)
	{
		fprintf(f, "\tin%i [label=\"in%i\\n%0.2f\"];\n",
		        m, m, in[m]);
	}

	// h1 nodes
	if(h1)
	{
		for(n = 0; n < h1->n; ++n)
		{
			fprintf(f, "\th1%i [label=\"h1%i\\nnet=%0.2f\\ndelta=%0.2f\\nb=%0.2f\\no=%0.2f\"];\n",
			        n, n, h1->net[n], h1->delta[n],
			        h1->b[n], h1->o[n]);
		}
	}

	// h2 nodes
	if(h2)
	{
		for(n = 0; n < h2->n; ++n)
		{
			fprintf(f, "\th2%i [label=\"h2%i\\nnet=%0.2f\\ndelta=%0.2f\\nb=%0.2f\\no=%0.2f\"];\n",
			        n, n, h2->net[n], h2->delta[n],
			        h2->b[n], h2->o[n]);
		}
	}

	// Omega nodes
	for(n = 0; n < Omega->n; ++n)
	{
		fprintf(f, "\tOmega%i [label=\"Omega%i\\nnet=%0.2f\\ndelta=%0.2f\\nb=%0.2f\\no=%0.2f\"];\n",
		        n, n, Omega->net[n], Omega->delta[n],
		        Omega->b[n], Omega->o[n]);
	}

	// weights
	float* w;
	if(h1 && h2)
	{
		// in > h1 weights
		for(n = 0; n < h1->n; ++n)
		{
			w = ai_mlpLayer_weight(h1, n);
			for(m = 0; m < h1->m; ++m)
			{
				fprintf(f, "\tin%i -> h1%i [label=\"%0.2f\"];\n",
				        m, n, w[m]);
			}
		}

		// h1 > h2 weights
		for(n = 0; n < h2->n; ++n)
		{
			w = ai_mlpLayer_weight(h2, n);
			for(m = 0; m < h2->m; ++m)
			{
				fprintf(f, "\th1%i -> h2%i [label=\"%0.2f\"];\n",
				        m, n, w[m]);
			}
		}

		// h2 > Omega weights
		for(n = 0; n < Omega->n; ++n)
		{
			w = ai_mlpLayer_weight(Omega, n);
			for(m = 0; m < Omega->m; ++m)
			{
				fprintf(f, "\th2%i -> Omega%i [label=\"%0.2f\"];\n",
				        m, n, w[m]);
			}
		}
	}
	else if(h1)
	{
		// in > h1 weights
		for(n = 0; n < h1->n; ++n)
		{
			w = ai_mlpLayer_weight(h1, n);
			for(m = 0; m < h1->m; ++m)
			{
				fprintf(f, "\tin%i -> h1%i [label=\"%0.2f\"];\n",
				        m, n, w[m]);
			}
		}

		// h1 > Omega weights
		for(n = 0; n < Omega->n; ++n)
		{
			w = ai_mlpLayer_weight(Omega, n);
			for(m = 0; m < Omega->m; ++m)
			{
				fprintf(f, "\th1%i -> Omega%i [label=\"%0.2f\"];\n",
				        m, n, w[m]);
			}
		}
	}
	else
	{
		// in > Omega weights
		for(n = 0; n < Omega->n; ++n)
		{
			w = ai_mlpLayer_weight(Omega, n);
			for(m = 0; m < Omega->m; ++m)
			{
				fprintf(f, "\tin%i -> Omega%i [label=\"%0.2f\"];\n",
				        m, n, w[m]);
			}
		}
	}

	// footer
	fprintf(f, "%s\n", "}");

	fclose(f);

	// success
	return 1;
}
