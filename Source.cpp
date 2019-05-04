#include "util.h"
#include "model.h"

int main(void)
{
	int batch_size = 2;
	int feature_size = 3;
	int unit_size = 2;
	int sequence_length = 3;
	
	// ct_1 , ht_1 initial value
	float ct_1_init[2][2] = { {1, 1} , {1, 1} };
	float ht_1_init[2][2] = { {1, 1} , {1, 1} };
	// kernel 
	float kernel[3][8] = { {1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1} };
	// rec_kernel
	float rec_kernel[2][8] = { {1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1} };
	// bias
	float bias[8] = { 1,1,1,1,1,1,1,1 };
	// input
	// sequence_length, batch_size, feature_size
	float x[3][2][3] = { {{1,1,1} , {1,1,1}}, {{1,1,1} , {1,1,1}}, {{1,1,1} , {1,1,1}} };
	float out[3][2][2];
	memset(out, 0, sizeof(out));

	struct LSTMcell testcell;
	testcell.init(batch_size, feature_size, unit_size);
	testcell.weight_transfer((float *)kernel, (float *)rec_kernel, (float *)bias);
	testcell.io_transfer((float *)x, (float *)out, (float *)ct_1_init, (float *)ht_1_init, sequence_length);
}