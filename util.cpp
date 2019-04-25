
// arr1 : (n, k)
// arr2 : (k, m)
// result : (n, m)
void matmul(float *arr1, float *arr2, float *result, int n, int k, int m)
{
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
			for (int i = 0; i < k; i++)
				result[y*m + x] += arr1[y*k + i] * arr2[i*m + x];
}

void elem_add(float *arr1, float *arr2, float*result, int n, int m)
{
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
			result[y*m + x] += arr1[y*m + x] + arr2[y*m + x];
}

void elem_add(float *arr1, float *arr2, float*result, int n, int m)
{
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
			result[y*m + x] += arr1[y*m + x] * arr2[y*m + x];
}

void sigmoid(float *result, int n, int m)
{
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
		{
			float t = result[y*m + x];
			result[y*m + x]  = t  / (1 + t);
		}
}