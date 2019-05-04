#pragma once
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include "util.h"


struct LSTMcell {
	int batch_size;
	int feature_size;
	int unit_size;

	float *ht_1;
	float *ct_1;
	float *kernel;
	float *rec_kernel;
	float *bias;

	void init(int batch_size, int feature_size, int unit_size)
	{
		this->batch_size = batch_size;
		this->unit_size = unit_size;
		this->feature_size = feature_size;

		ht_1 = (float *)malloc(sizeof(float) *batch_size * unit_size);
		ct_1 = (float *)malloc(sizeof(float) *batch_size * unit_size);
		kernel = (float *)malloc(sizeof(float) * feature_size * unit_size * 4);
		rec_kernel = (float *)malloc(sizeof(float) * unit_size * unit_size * 4);
		bias = (float *)malloc(sizeof(float)*unit_size * 4);

		memset(ht_1, 0, sizeof(float) *batch_size * unit_size);
		memset(ct_1, 0, sizeof(float) *batch_size * unit_size);
		memset(rec_kernel, 0, sizeof(float) *batch_size * unit_size);
		memset(kernel, 0, sizeof(float) *batch_size * feature_size);
		memset(bias, 0, sizeof(float) * unit_size);
	}

	void weight_transfer(float *kernel_in, float *rec_kernel_in, float *bias_in)
	{
		memcpy(kernel, kernel_in, sizeof(float) * feature_size *unit_size * 4);
		memcpy(rec_kernel, rec_kernel_in, sizeof(float) * unit_size * unit_size * 4);
		memcpy(bias, bias_in, sizeof(float) * unit_size * 4);
	}

	// x : float[sequence][batch_size][feature_size]
	// out : float[sequence][batch_size][unit_size]
	// h_init , c_init : float[sequence][batch_size][unit_size]

	void io_transfer(float *x,float *out, float *h_init, float *c_init, int total_sequence)
	{
		for (int seq = 0; seq < total_sequence; seq++)
		{
			printf("-----------step %d-----------\n", seq);
			// call calculation, check x
			print_2d_array("input : ", (float *)&x[batch_size * feature_size], batch_size, feature_size);
			step(&x[batch_size * feature_size], &out[batch_size * unit_size], h_init, c_init, seq);
			print_2d_array("output : ", (float *)&out[batch_size * unit_size], batch_size, unit_size);

		}
	}

	// input : 
	// x float(batch, feature) 
	// h_init float(batch, unit) 
	// c_init float(batch, unit) 
	// int sequence
	void step(float *x, float *out, float *h_init, float *c_init, int sequence)
	{
		if (sequence == 0)
		{
			memcpy(ht_1, h_init, sizeof(float)* batch_size * unit_size);
			memcpy(ct_1, c_init, sizeof(float)* batch_size * unit_size);
		}

		// ifco
		float *ifco = (float *)malloc(sizeof(float) *batch_size * unit_size * 4);
		memset(ifco, 0, sizeof(float) *batch_size * unit_size * 4);

		// A
		matmul(x, kernel, ifco, batch_size, feature_size, unit_size * 4);
		// B
		matmul(ht_1, rec_kernel, ifco, batch_size, unit_size, unit_size * 4);
		// C
		biasadd(ifco, bias, batch_size, unit_size * 4);

		// i f c o each (batch, unit)
		sigmoid(&ifco[0], batch_size, unit_size);
		sigmoid(&ifco[unit_size * batch_size], batch_size, unit_size);
		sigmoid(&ifco[unit_size  * batch_size * 2], batch_size, unit_size);
		sigmoid(&ifco[unit_size  * batch_size * 3], batch_size, unit_size);

		float *i = &ifco[0];
		float *f = &ifco[unit_size * batch_size];
		float *c = &ifco[unit_size * batch_size * 2];
		float *o = &ifco[unit_size * batch_size * 3];

		float *fct_1 = (float *)malloc(sizeof(float) *batch_size * unit_size);
		memset(fct_1, 0, sizeof(float) *batch_size * unit_size);
		float *io = (float *)malloc(sizeof(float) *batch_size * unit_size);
		memset(io, 0, sizeof(float) *batch_size * unit_size);
		float *ct = (float *)malloc(sizeof(float) *batch_size * unit_size);
		memset(ct, 0, sizeof(float) *batch_size * unit_size);
		float *ht = (float *)malloc(sizeof(float) *batch_size * unit_size);
		memset(ht, 0, sizeof(float) *batch_size * unit_size);

		// calculate ct
		elem_multiply(ct_1, f, fct_1, batch_size, unit_size); // ct-1 x f
		elem_multiply(i, c, io, batch_size, unit_size); // i x c
		elem_add(fct_1, io, ct, batch_size, unit_size); // i x c + ct-1 x f
		// store ct
		memcpy(ct_1, ct, sizeof(float)*batch_size * unit_size);

		// calculate ht
		sigmoid(ct, batch_size, unit_size);
		elem_multiply(ct, o, ht, batch_size, unit_size);
		// store ht
		memcpy(ht_1, ht, sizeof(float)*batch_size * unit_size);
		// copy ht to the output buffer
		memcpy(out, ht_1, sizeof(float)*batch_size * unit_size);

		free(ifco);
		free(fct_1);
		free(io);
		free(ct);
		free(ht);
	}
};