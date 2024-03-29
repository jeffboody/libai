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

#ifndef ai_mlpFact_H
#define ai_mlpFact_H

typedef float (*ai_mlpFact_fn)(float x);

// string/function conversions
const char*   ai_mlpFact_string(ai_mlpFact_fn fact);
ai_mlpFact_fn ai_mlpFact_function(const char* str);

// activation functions
float ai_mlpFact_linear(float x);
float ai_mlpFact_logistic(float x);
float ai_mlpFact_ReLU(float x);
float ai_mlpFact_PReLU(float x);
float ai_mlpFact_tanh(float x);

// activation function derivatives
float ai_mlpFact_dlinear(float x);
float ai_mlpFact_dlogistic(float x);
float ai_mlpFact_dReLU(float x);
float ai_mlpFact_dPReLU(float x);
float ai_mlpFact_dtanh(float x);

#endif
