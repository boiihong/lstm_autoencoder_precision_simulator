#pragma once

void biasadd(float *result, float *bias, int n, int m);
void sigmoid(float *result, int n, int m);
void elem_multiply(float *arr1, float *arr2, float*result, int n, int m);
void elem_add(float *arr1, float *arr2, float*result, int n, int m);
void matmul(float *arr1, float *arr2, float *result, int n, int k, int m);

