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
#include <string.h>
#include <time.h>

#define LOG_TAG "ai"
#include "../../libcc/cc_log.h"
#include "../../libcc/cc_memory.h"
#include "../../jsmn/wrapper/jsmn_wrapper.h"
#include "ai_mlpLayer.h"
#include "ai_mlp.h"

/***********************************************************
* private                                                  *
***********************************************************/

static int ai_mlp_cat(char** _a, char* b)
{
	ASSERT(_a);
	ASSERT(b);

	char* a = *_a;

	// compute size/length
	size_t size1 = MEMSIZEPTR(a);
	size_t lena  = 0;
	size_t lenb  = strlen(b);
	if(a)
	{
		lena = strlen(a);
	}
	size_t size2 = lena + lenb + 1;

	// resize buffer
	if(size1 < size2)
	{
		size_t size = size1;
		if(size1 == 0)
		{
			size = 256;
		}

		while(size < size2)
		{
			size *= 2;
		}

		char* tmp = (char*) REALLOC(a, size2);
		if(tmp == NULL)
		{
			LOGE("REALLOC failed");
			return 0;
		}

		a   = tmp;
		*_a = a;
	}

	// copy buffer
	int i;
	for(i = 0; i < lenb; ++i)
	{
		a[i + lena] = b[i];
	}
	a[lena + lenb] = '\0';

	return 1;
}

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
	ASSERT(facth);
	ASSERT(dfacth);
	ASSERT(facto);
	ASSERT(dfacto);

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

	self->in = (float*) CALLOC(m, sizeof(float));
	if(self->in == NULL)
	{
		goto fail_in;
	}

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
		FREE(self->in);
	fail_in:
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
		FREE(self->in);
		FREE(self);
		*_self = NULL;
	}
}

ai_mlp_t* ai_mlp_import(size_t size, const char* json,
                        ai_mlpFact_fn default_facth,
                        ai_mlpFact_fn default_dfacth,
                        ai_mlpFact_fn default_facto,
                        ai_mlpFact_fn default_dfacto)
{
	ASSERT(json);
	ASSERT(default_facth);
	ASSERT(default_dfacth);
	ASSERT(default_facto);
	ASSERT(default_dfacto);

	// see ai_mlp_export for file format

	jsmn_val_t* root = jsmn_val_new(json, size);
	if(root == NULL)
	{
		return NULL;
	}

	if(root->type != JSMN_TYPE_OBJECT)
	{
		LOGE("invalid type=%i", root->type);
		goto fail_type;
	}

	int            m      = 0;
	int            p      = 0;
	int            q      = 0;
	int            n      = 0;
	float          rate   = 0.1f;
	ai_mlpFact_fn  facth  = NULL;
	ai_mlpFact_fn  dfacth = NULL;
	ai_mlpFact_fn  facto  = NULL;
	ai_mlpFact_fn  dfacto = NULL;
	jsmn_array_t*  h1b    = NULL;
	jsmn_array_t*  h1w    = NULL;
	jsmn_array_t*  h2b    = NULL;
	jsmn_array_t*  h2w    = NULL;
	jsmn_array_t*  Omegab = NULL;
	jsmn_array_t*  Omegaw = NULL;

	// parse kv pairs
	jsmn_keyval_t* kv;
	cc_listIter_t* iter = cc_list_head(root->obj->list);
	while(iter)
	{
		kv = (jsmn_keyval_t*) cc_list_peekIter(iter);
		if(kv->val->type == JSMN_TYPE_STRING)
		{
			if(strcmp(kv->key, "m") == 0)
			{
				m = (int) strtol(kv->val->data, NULL, 0);
			}
			else if(strcmp(kv->key, "p") == 0)
			{
				p = (int) strtol(kv->val->data, NULL, 0);
			}
			else if(strcmp(kv->key, "q") == 0)
			{
				q = (int) strtol(kv->val->data, NULL, 0);
			}
			else if(strcmp(kv->key, "n") == 0)
			{
				n = (int) strtol(kv->val->data, NULL, 0);
			}
			else if(strcmp(kv->key, "rate") == 0)
			{
				rate = strtof(kv->val->data, NULL);
			}
			else if(strcmp(kv->key, "facth") == 0)
			{
				facth = ai_mlpFact_function(kv->val->data);
			}
			else if(strcmp(kv->key, "dfacth") == 0)
			{
				dfacth = ai_mlpFact_function(kv->val->data);
			}
			else if(strcmp(kv->key, "facto") == 0)
			{
				facto = ai_mlpFact_function(kv->val->data);
			}
			else if(strcmp(kv->key, "dfacto") == 0)
			{
				dfacto = ai_mlpFact_function(kv->val->data);
			}
			else
			{
				LOGW("unknown key=%s", kv->key);
			}
		}
		else if(kv->val->type == JSMN_TYPE_ARRAY)
		{
			if(strcmp(kv->key, "h1b") == 0)
			{
				h1b = kv->val->array;
			}
			else if(strcmp(kv->key, "h1w") == 0)
			{
				h1w = kv->val->array;
			}
			else if(strcmp(kv->key, "h2b") == 0)
			{
				h2b = kv->val->array;
			}
			else if(strcmp(kv->key, "h2w") == 0)
			{
				h2w = kv->val->array;
			}
			else if(strcmp(kv->key, "Omegab") == 0)
			{
				Omegab = kv->val->array;
			}
			else if(strcmp(kv->key, "Omegaw") == 0)
			{
				Omegaw = kv->val->array;
			}
			else
			{
				LOGW("unknown key=%s", kv->key);
			}
		}
		else
		{
			LOGW("invalid key=%s, type=%i",
			     kv->key, kv->val->type);
		}

		iter = cc_list_next(iter);
	}

	// replace custom fact
	if(facth == NULL)
	{
		facth = default_facth;
	}
	if(dfacth == NULL)
	{
		dfacth = default_dfacth;
	}
	if(facto == NULL)
	{
		facto = default_facto;
	}
	if(dfacto == NULL)
	{
		dfacto = default_dfacto;
	}

	// validate inputs
	if((m < 1) || (p < 0) || (q < 0) || (n < 1) ||
	   (rate <= 0.0f) ||
	   (facth == NULL) || (dfacth == NULL) ||
	   (facto == NULL) || (dfacto == NULL))
	{
		LOGE("invalid m=%i, p=%i, q=%i, n=%i, rate=%f, facth=%p, dfacth=%p, facto=%p, dfacto=%p",
		     m, p, q, n, rate, facth, dfacth, facto, dfacto);
		goto fail_validate;
	}

	ai_mlp_t* self;
	self = ai_mlp_new(m, p, q, n, rate,
	                  facth, dfacth, facto, dfacto);
	if(self == NULL)
	{
		goto fail_mlp;
	}

	// h1 bias and weights
	int            idx;
	jsmn_val_t*    val;
	ai_mlpLayer_t* h1 = self->h1;
	if(h1)
	{
		if((h1b == NULL) || (h1w == NULL) ||
		   (cc_list_size(h1b->list) != h1->n) ||
		   (cc_list_size(h1w->list) != (h1->n*h1->m)))
		{
			LOGE("invalid h1b=%p, h1w=%p", h1b, h1w);
			goto fail_h1;
		}

		idx  = 0;
		iter = cc_list_head(h1b->list);
		while(iter)
		{
			val = (jsmn_val_t*) cc_list_peekIter(iter);
			if(val->type != JSMN_TYPE_PRIMITIVE)
			{
				LOGE("invalid type=%i", val->type);
				goto fail_h1;
			}

			h1->b[idx++] = strtof(val->data, NULL);
			iter = cc_list_next(iter);
		}

		idx  = 0;
		iter = cc_list_head(h1w->list);
		while(iter)
		{
			val = (jsmn_val_t*) cc_list_peekIter(iter);
			if(val->type != JSMN_TYPE_PRIMITIVE)
			{
				LOGE("invalid type=%i", val->type);
				goto fail_h1;
			}

			h1->w[idx++] = strtof(val->data, NULL);
			iter = cc_list_next(iter);
		}
	}

	// h2 bias and weights
	ai_mlpLayer_t* h2 = self->h2;
	if(h2)
	{
		if((h2b == NULL) || (h2w == NULL) ||
		   (cc_list_size(h2b->list) != h2->n) ||
		   (cc_list_size(h2w->list) != (h2->n*h2->m)))
		{
			goto fail_h2;
		}

		idx  = 0;
		iter = cc_list_head(h2b->list);
		while(iter)
		{
			val = (jsmn_val_t*) cc_list_peekIter(iter);
			if(val->type != JSMN_TYPE_PRIMITIVE)
			{
				LOGE("invalid type=%i", val->type);
				goto fail_h2;
			}

			h2->b[idx++] = strtof(val->data, NULL);
			iter = cc_list_next(iter);
		}

		idx  = 0;
		iter = cc_list_head(h2w->list);
		while(iter)
		{
			val = (jsmn_val_t*) cc_list_peekIter(iter);
			if(val->type != JSMN_TYPE_PRIMITIVE)
			{
				LOGE("invalid type=%i", val->type);
				goto fail_h2;
			}

			h2->w[idx++] = strtof(val->data, NULL);
			iter = cc_list_next(iter);
		}
	}

	// Omega bias and weights
	ai_mlpLayer_t* Omega = self->Omega;
	{
		if((Omegab == NULL) || (Omegaw == NULL) ||
		   (cc_list_size(Omegab->list) != Omega->n) ||
		   (cc_list_size(Omegaw->list) != (Omega->n*Omega->m)))
		{
			goto fail_Omega;
		}

		idx  = 0;
		iter = cc_list_head(Omegab->list);
		while(iter)
		{
			val = (jsmn_val_t*) cc_list_peekIter(iter);
			if(val->type != JSMN_TYPE_PRIMITIVE)
			{
				LOGE("invalid type=%i", val->type);
				goto fail_Omega;
			}

			Omega->b[idx++] = strtof(val->data, NULL);
			iter = cc_list_next(iter);
		}

		idx  = 0;
		iter = cc_list_head(Omegaw->list);
		while(iter)
		{
			val = (jsmn_val_t*) cc_list_peekIter(iter);
			if(val->type != JSMN_TYPE_PRIMITIVE)
			{
				LOGE("invalid type=%i", val->type);
				goto fail_Omega;
			}

			Omega->w[idx++] = strtof(val->data, NULL);
			iter = cc_list_next(iter);
		}
	}

	jsmn_val_delete(&root);

	// success
	return self;

	// failure
	fail_Omega:
	fail_h2:
	fail_h1:
		ai_mlp_delete(&self);
	fail_mlp:
	fail_validate:
	fail_type:
		jsmn_val_delete(&root);
	return NULL;
}

char* ai_mlp_export(ai_mlp_t* self)
{
	ASSERT(self);

	ai_mlpLayer_t* h1    = self->h1;
	ai_mlpLayer_t* h2    = self->h2;
	ai_mlpLayer_t* Omega = self->Omega;

	// json file format
	// {"m":"M","p":"P","q":"Q","n":"N",
	//  "rate":"RATE",
	//  "facth":"FACTH","facto":"FACTO",
	//  "dfacth":"DFACTH","dfacto":"DFACTO",
	//  "h1b":[1,2,3],"h1w":[1,2,3],
	//  "h2b":[1,2,3],"h2w":[1,2,3],
	//  "Omegab":[1,2,3],"Omegaw":[1,2,3]}

	// mpqn, rate, fact, dfact
	char mpqn[256];
	char rate[256];
	char fact[256];
	char dfact[256];
	snprintf(mpqn, 256, "\"m\":\"%i\",\"p\":\"%i\",\"q\":\"%i\",\"n\":\"%i\",",
	         self->m, self->p, self->q, self->n);
	snprintf(rate, 256, "\"rate\":\"%f\",",
	         self->rate);
	if(h1)
	{
		snprintf(fact, 256, "\"facth\":\"%s\",\"facto\":\"%s\",",
		         ai_mlpFact_string(h1->fact),
		         ai_mlpFact_string(Omega->fact));
		snprintf(dfact, 256, "\"dfacth\":\"%s\",\"dfacto\":\"%s\",",
		         ai_mlpFact_string(h1->dfact),
		         ai_mlpFact_string(Omega->dfact));
	}
	else
	{
		snprintf(fact, 256, "\"facto\":\"%s\",",
		         ai_mlpFact_string(Omega->fact));
		snprintf(fact, 256, "\"dfacto\":\"%s\",",
		         ai_mlpFact_string(Omega->dfact));
	}

	char* buf = NULL;
	int   ret = ai_mlp_cat(&buf, "{");
	ret &= ai_mlp_cat(&buf, mpqn);
	ret &= ai_mlp_cat(&buf, rate);
	ret &= ai_mlp_cat(&buf, fact);
	ret &= ai_mlp_cat(&buf, dfact);

	// h1 bias and weights
	int i;
	if(h1)
	{
		char h1b[256];
		ret &= ai_mlp_cat(&buf, "\"h1b\":[");
		for(i = 0; i < h1->n; ++i)
		{
			if(i)
			{
				snprintf(h1b, 256, ",%f", h1->b[i]);
			}
			else
			{
				snprintf(h1b, 256, "%f", h1->b[i]);
			}
			ret &= ai_mlp_cat(&buf, h1b);
		}
		ret &= ai_mlp_cat(&buf, "],");

		char h1w[256];
		ret &= ai_mlp_cat(&buf, "\"h1w\":[");
		for(i = 0; i < h1->n*h1->m; ++i)
		{
			if(i)
			{
				snprintf(h1w, 256, ",%f", h1->w[i]);
			}
			else
			{
				snprintf(h1w, 256, "%f", h1->w[i]);
			}
			ret &= ai_mlp_cat(&buf, h1w);
		}
		ret &= ai_mlp_cat(&buf, "],");
	}

	// h2 bias and weights
	if(h2)
	{
		char h2b[256];
		ret &= ai_mlp_cat(&buf, "\"h2b\":[");
		for(i = 0; i < h2->n; ++i)
		{
			if(i)
			{
				snprintf(h2b, 256, ",%f", h2->b[i]);
			}
			else
			{
				snprintf(h2b, 256, "%f", h2->b[i]);
			}
			ret &= ai_mlp_cat(&buf, h2b);
		}
		ret &= ai_mlp_cat(&buf, "],");

		char h2w[256];
		ret &= ai_mlp_cat(&buf, "\"h2w\":[");
		for(i = 0; i < h2->n*h2->m; ++i)
		{
			if(i)
			{
				snprintf(h2w, 256, ",%f", h2->w[i]);
			}
			else
			{
				snprintf(h2w, 256, "%f", h2->w[i]);
			}
			ret &= ai_mlp_cat(&buf, h2w);
		}
		ret &= ai_mlp_cat(&buf, "],");
	}

	// Omega bias and weights
	char Omegab[256];
	ret &= ai_mlp_cat(&buf, "\"Omegab\":[");
	for(i = 0; i < Omega->n; ++i)
	{
		if(i)
		{
			snprintf(Omegab, 256, ",%f", Omega->b[i]);
		}
		else
		{
			snprintf(Omegab, 256, "%f", Omega->b[i]);
		}
		ret &= ai_mlp_cat(&buf, Omegab);
	}
	ret &= ai_mlp_cat(&buf, "],");

	char Omegaw[256];
	ret &= ai_mlp_cat(&buf, "\"Omegaw\":[");
	for(i = 0; i < Omega->n*Omega->m; ++i)
	{
		if(i)
		{
			snprintf(Omegaw, 256, ",%f", Omega->w[i]);
		}
		else
		{
			snprintf(Omegaw, 256, "%f", Omega->w[i]);
		}
		ret &= ai_mlp_cat(&buf, Omegaw);
	}
	ret &= ai_mlp_cat(&buf, "]}");

	// check for failures
	if(ret == 0)
	{
		FREE(buf);
		buf = NULL;
	}

	return buf;
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

	self->state = AI_MLP_STATE_TRAIN;
}

float* ai_mlp_solve(ai_mlp_t* self, float* in)
{
	ASSERT(self);
	ASSERT(in);

	memcpy(self->in, in, self->m*sizeof(float));

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

	self->state = AI_MLP_STATE_SOLVE;

	return oo;
}

int ai_mlp_graph(ai_mlp_t* self, const char* label,
                 const char* fname)
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

	const char* h1_fact    = NULL;
	const char* h2_fact    = NULL;
	const char* Omega_fact = NULL;
	if(h1)
	{
		h1_fact = ai_mlpFact_string(h1->fact);
	}
	if(h2)
	{
		h2_fact = ai_mlpFact_string(h2->fact);
	}
	Omega_fact = ai_mlpFact_string(Omega->fact);

	int m;
	int n;
	if(self->state == AI_MLP_STATE_TRAIN)
	{
		// in nodes
		for(m = 0; m < self->m; ++m)
		{
			fprintf(f, "\tin%i [label=\"in%i\\n%0.2f\"];\n",
			        m, m, self->in[m]);
		}

		// h1 nodes
		if(h1)
		{
			for(n = 0; n < h1->n; ++n)
			{
				fprintf(f, "\th1%i [label=\"h1%i\\nfact=%s\\nb=%0.2f\\nnet=%0.2f\\ndelta=%0.2f\\no=%0.2f\"];\n",
				        n, n, h1_fact, h1->b[n], h1->net[n],
				        h1->delta[n], h1->o[n]);
			}
		}

		// h2 nodes
		if(h2)
		{
			for(n = 0; n < h2->n; ++n)
			{
				fprintf(f, "\th2%i [label=\"h2%i\\nfact=%s\\nb=%0.2f\\nnet=%0.2f\\ndelta=%0.2f\\no=%0.2f\"];\n",
				        n, n, h2_fact, h2->b[n], h2->net[n],
				        h2->delta[n], h2->o[n]);
			}
		}

		// Omega nodes
		for(n = 0; n < Omega->n; ++n)
		{
			fprintf(f, "\tOmega%i [label=\"Omega%i\\nfact=%s\\nb=%0.2f\\nnet=%0.2f\\ndelta=%0.2f\\no=%0.2f\"];\n",
			        n, n, Omega_fact, Omega->b[n], Omega->net[n],
			        Omega->delta[n], Omega->o[n]);
		}
	}
	else if(self->state == AI_MLP_STATE_SOLVE)
	{
		// in nodes
		for(m = 0; m < self->m; ++m)
		{
			fprintf(f, "\tin%i [label=\"in%i\\n%0.2f\"];\n",
			        m, m, self->in[m]);
		}

		// h1 nodes
		if(h1)
		{
			for(n = 0; n < h1->n; ++n)
			{
				fprintf(f, "\th1%i [label=\"h1%i\\nfact=%s\\nb=%0.2f\\nnet=%0.2f\\no=%0.2f\"];\n",
				        n, n, h1_fact, h1->b[n], h1->net[n], h1->o[n]);
			}
		}

		// h2 nodes
		if(h2)
		{
			for(n = 0; n < h2->n; ++n)
			{
				fprintf(f, "\th2%i [label=\"h2%i\\nfact=%s\\nb=%0.2f\\nnet=%0.2f\\no=%0.2f\"];\n",
				        n, n, h2_fact, h2->b[n], h2->net[n], h2->o[n]);
			}
		}

		// Omega nodes
		for(n = 0; n < Omega->n; ++n)
		{
			fprintf(f, "\tOmega%i [label=\"Omega%i\\nfact=%s\\nb=%0.2f\\nnet=%0.2f\\no=%0.2f\"];\n",
			        n, n, Omega_fact, Omega->b[n], Omega->net[n], Omega->o[n]);
		}
	}
	else
	{
		// in nodes
		for(m = 0; m < self->m; ++m)
		{
			fprintf(f, "\tin%i [label=\"in%i\"];\n", m, m);
		}

		// h1 nodes
		if(h1)
		{
			for(n = 0; n < h1->n; ++n)
			{
				fprintf(f, "\th1%i [label=\"h1%i\\nfact=%s\\nb=%0.2f\"];\n",
				        n, n, h1_fact, h1->b[n]);
			}
		}

		// h2 nodes
		if(h2)
		{
			for(n = 0; n < h2->n; ++n)
			{
				fprintf(f, "\th2%i [label=\"h2%i\\nfact=%s\\nb=%0.2f\"];\n",
				        n, n, h2_fact, h2->b[n]);
			}
		}

		// Omega nodes
		for(n = 0; n < Omega->n; ++n)
		{
			fprintf(f, "\tOmega%i [label=\"Omega%i\\nfact=%s\\nb=%0.2f\"];\n",
			        n, n, Omega_fact, Omega->b[n]);
		}
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
