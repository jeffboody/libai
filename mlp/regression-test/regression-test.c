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
#include <stdio.h>
#include <stdlib.h>
#include "libai/mlp/ai_mlpFact.h"
#include "libai/mlp/ai_mlpLayer.h"
#include "libai/mlp/ai_mlp.h"

#define LOG_TAG "ai"
#include "libcc/cc_log.h"

#define REGRESSION_TEST_XSQUARED

/***********************************************************
* public                                                   *
***********************************************************/

int main(int argc, char** argv)
{
	ai_mlp_t* mlp;
	#ifdef REGRESSION_TEST_XSQUARED
	mlp = ai_mlp_new(1, 1, 0, 1, 0.1f,
	                 ai_mlpFact_tanh,
	                 ai_mlpFact_dtanh,
	                 ai_mlpFact_linear,
	                 ai_mlpFact_dlinear);
	#else
	mlp = ai_mlp_new(1, 0, 0, 1, 0.1f,
                 ai_mlpFact_ReLU,
                 ai_mlpFact_ReLU,
                 ai_mlpFact_linear,
                 ai_mlpFact_dlinear);
	#endif
	if(mlp == NULL)
	{
		return EXIT_FAILURE;
	}

	float in0 = 0;
	char fname[256];
	char label[256];
	snprintf(label, 256, "%s", "Init");
	snprintf(fname, 256, "%s", "out/init.dot");
	ai_mlp_graph(mlp, &in0, label, fname);

	FILE* fdat;
	fdat = fopen("out/output.dat", "w");
	if(fdat == NULL)
	{
		goto fail_fdat;
	}

	FILE* ferr;
	ferr = fopen("out/error.dat", "w");
	if(ferr == NULL)
	{
		goto fail_ferr;
	}

	// training
	int i;
	float in;
	float out;
	float err;
	int   count = 20000;
	for(i = 0; i < count; ++i)
	{
		in  = 1.0f*((float) (rand()%(count + 1)))/
		      ((float) count);
		#ifdef REGRESSION_TEST_XSQUARED
		out = in*in;
		#else
		out = 2.0f*in + 1.0f;
		#endif

		ai_mlp_train(mlp, &in, &out);

		if((i%200) == 0)
		{
			snprintf(label, 256, "Train%i", i);
			snprintf(fname, 256, "out/train%i.dot", i);
			ai_mlp_graph(mlp, &in, label, fname);

			err = fabs(out - mlp->Omega->o[0]);
			fprintf(ferr, "%i %f\n", i, err);

			LOGD("TRAIN i=%i, in=%f, out=%f, o=%f, err=%f",
			     i, in, out, mlp->Omega->o[0], err);
		}
	}

	// solving
	count = 20;
	for(i = 0; i < count; ++i)
	{
		in  = 1.0f*((float) i)/((float) count);
		#ifdef REGRESSION_TEST_XSQUARED
		out = in*in;
		#else
		out = 2.0f*in + 1.0f;
		#endif

		ai_mlp_solve(mlp, &in);

		err = fabs(out - mlp->Omega->o[0]);
		fprintf(fdat, "%f %f %f\n", in, out, mlp->Omega->o[0]);

		LOGD("SOLVE i=%i, in=%f, out=%f, o=%f, err=%f",
		     i, in, out, mlp->Omega->o[0], err);
	}

	fclose(ferr);
	fclose(fdat);
	ai_mlp_delete(&mlp);

	// success
	return EXIT_SUCCESS;

	// failure
	fail_ferr:
		fclose(fdat);
	fail_fdat:
		ai_mlp_delete(&mlp);
	return EXIT_FAILURE;
}
