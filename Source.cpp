#include <malloc.h>
#include <string.h>
#include "util.cpp"

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
		kernel = (float *)malloc(sizeof(float) *batch_size * feature_size);
		rec_kernel = (float *)malloc(sizeof(float) *batch_size * unit_size);
		bias = (float *)malloc(sizeof(float)*unit_size);

		memset(ht_1, 0, sizeof(float) *batch_size * unit_size);
		memset(ct_1, 0, sizeof(float) *batch_size * unit_size);
		memset(rec_kernel, 0, sizeof(float) *batch_size * unit_size);
		memset(kernel, 0, sizeof(float) *batch_size * feature_size);
		memset(bias, 0, sizeof(float) * unit_size);
	}

	void weight_transfer(float *kernel_in, float *rec_kernel_in, float *bias_in)
	{
		memcpy(kernel, kernel_in, sizeof(float) *batch_size * feature_size);
		memcpy(rec_kernel, rec_kernel_in, sizeof(float) *batch_size * unit_size);
		memcpy(bias, bias_in, sizeof(float) * unit_size);
	}


	// input : 
	// x float(batch, feature) 
	// h_init float(batch, unit) 
	// c_init float(batch, unit) 
	// int sequence
	void step(float *x, float *h_init, float *c_init, int sequence)
	{
		if (sequence == 0)
		{
			memcpy(ht_1, h_init, sizeof(float)* batch_size * unit_size);
			memcpy(ct_1, c_init, sizeof(float)* batch_size * unit_size);
		}

		// i f 
		float *i = (float *)malloc( sizeof(float) *batch_size * unit_size);
		memset(i, 0, sizeof(float) *batch_size * unit_size);
		float *f = (float *)malloc(sizeof(float) *batch_size * unit_size);
		memset(f, 0, sizeof(float) *batch_size * unit_size);

		matmul()
	}
};

int main(void)
{

}