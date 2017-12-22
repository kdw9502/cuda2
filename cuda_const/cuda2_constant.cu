#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <time.h>
#include <Windows.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <assert.h>

/******************************
Device: GTX 1070
===============================
N_ELEMS: 2^21
Cutoff count: 1,000

******************************/
#define ELEM_PER_POINT				(1 << 7)
#define N_ELEM_LOG					(22)

#define N_ELEMS						(1 << N_ELEM_LOG)
#define N_POINTS					(N_ELEMS / ELEM_PER_POINT)

#define ARRAY_2D_WIDTH				1024
#define ARRAY_2D_HEIGHT				(N_POINTS/ARRAY_2D_WIDTH)
#define BLOCK_WIDTH					1025
#define BLOCK_HEIGHT				1

//#define MAX_SHARED_MEM_PER_BLOCK	(3 << 14)	// from GTX 680
//#define MAX_SHARED_MEM_PER_SM		(3 << 14)	// same with 'per block'

//#define SHARED_AOS_BLOCK_HEIGHT		((ELEM_PER_POINT >> 4) ? 3 : BLOCK_HEIGHT)
//#define SHARED_AOS_BLOCK_WIDTH		((ELEM_PER_POINT >> 4) ? ((MAX_SHARED_MEM_PER_BLOCK / (ELEM_PER_POINT * sizeof(float)) ) / SHARED_AOS_BLOCK_HEIGHT) : BLOCK_WIDTH)

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))

#define IN
#define INOUT
#define OUT


typedef struct {
	float elem[ ELEM_PER_POINT ];
} POINT_ELEMENT;

typedef struct {
	float *elem[ ELEM_PER_POINT ];
} POINTS_SOA;

__constant__ float constantBuffer[ 1000 ];
//extern __shared__ float sharedBuffer[ ];

__global__ void TransformAOSKernel( INOUT POINT_ELEMENT *A, IN int m )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int id = gridDim.x*blockDim.x*row + col;
	float tmp;
	int i,j;
	for (j = 2; j <= m; j++) {
		tmp = 1.0f / (float)j;
		for(i=0;i<ELEM_PER_POINT;i++){
			A[id].elem[i] += tmp*A[id].elem[i];
		}
	}

}

__global__ void TransformSOAKernel( INOUT POINTS_SOA A, IN int m )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int id = gridDim.x*blockDim.x*row + col;
		float tmp;
	int i,j;
	for(i=0;i<ELEM_PER_POINT;i++){
		for (j = 2; j <= m; j++) {
			tmp = 1.0f / (float)j;
		
			A.elem[i][id] += tmp*A.elem[i][id];
		}
	}

}

__global__ void TransformAOSwithConstantMemKernel( INOUT POINT_ELEMENT *A, IN int m )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int id = gridDim.x*blockDim.x*row + col;
	int i,j;

	for ( i = 0; i < ELEM_PER_POINT; ++i ) {
		for ( j = 2; j <= m; j++ ) {
			A[id].elem[ i ] += constantBuffer[j-1]*A[id].elem[ i ];
		}
	}
	
}

__global__ void TransformSOAwithConstantMemKernel( INOUT POINTS_SOA A, IN int m )
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int id = gridDim.x*blockDim.x*row + col;
	int i, j;

	for ( i = 0; i < ELEM_PER_POINT; ++i ) {
		for ( j = 2; j <= m; j++ ) {
			A.elem[ i ][ id ] += constantBuffer[j-1]*A.elem[ i ][ id ];
		}
	}
	
}

void transform_points_AOS( INOUT POINT_ELEMENT *p_AOS, IN int n_points, IN int m )
{
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	

//.............
	POINT_ELEMENT *d_pAOS;
	size_t size = N_POINTS * sizeof(POINT_ELEMENT);
	cudaMalloc(&d_pAOS, size);
	cudaMemcpy(d_pAOS, p_AOS, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 dimGrid(ARRAY_2D_WIDTH /dimBlock.x, ARRAY_2D_HEIGHT/dimBlock.y);

		cudaEventRecord( start, 0 );
	TransformAOSKernel <<< dimGrid, dimBlock >>>(d_pAOS, m);
		cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaMemcpy(p_AOS, d_pAOS, size, cudaMemcpyDeviceToHost);
	cudaFree(d_pAOS);


	//cudaDeviceSynchronize(); //It may stall the GPU pipeline.

	//
//.............



	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("*** GPU1 - E:%d N:%d B:(%d,%d) M:global,AOS : GPU Time taken = %.3fms\n", ELEM_PER_POINT, N_ELEM_LOG, BLOCK_HEIGHT, BLOCK_WIDTH, elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
}

void transform_points_SOA( INOUT POINTS_SOA p_SOA, IN int n_points, IN int m )
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	

//.............
	POINTS_SOA d_pSOA;
	size_t size = N_POINTS * sizeof(float);
	for(int i=0;i<ELEM_PER_POINT;i++){
		cudaMalloc(&d_pSOA.elem[i], size);
		cudaMemcpy(d_pSOA.elem[i], p_SOA.elem[i], size, cudaMemcpyHostToDevice);
	}
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 dimGrid(ARRAY_2D_WIDTH/dimBlock.x, ARRAY_2D_HEIGHT/dimBlock.y);
	cudaEventRecord( start, 0 );
	TransformSOAKernel <<< dimGrid, dimBlock >>> (d_pSOA, m);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	for(int i=0;i<ELEM_PER_POINT;i++){
		cudaMemcpy(p_SOA.elem[i], d_pSOA.elem[i], size, cudaMemcpyDeviceToHost);

	}
	for(int i=0;i<ELEM_PER_POINT;i++){
		cudaFree(d_pSOA.elem[i]);
	}
	//cudaDeviceSynchronize(); //It may stall the GPU pipeline.

	//
//.............



	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("*** GPU2 - E:%d N:%d B:(%d,%d) M:global,SOA : GPU Time taken = %.3fms\n", ELEM_PER_POINT, N_ELEM_LOG, BLOCK_HEIGHT, BLOCK_WIDTH, elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

void transform_points_AOS_with_constant( INOUT POINT_ELEMENT *p_AOS, IN int n_points, IN int m )
{	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	


//.............
	POINT_ELEMENT *d_pAOS;
	size_t size = N_POINTS * sizeof(POINT_ELEMENT);
	cudaMalloc(&d_pAOS, size);
	cudaMemcpy(d_pAOS, p_AOS, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 dimGrid(ARRAY_2D_WIDTH /dimBlock.x, ARRAY_2D_HEIGHT/dimBlock.y);
		cudaEventRecord( start, 0 );
	
	TransformAOSwithConstantMemKernel <<< dimGrid, dimBlock >>>(d_pAOS, m);
		cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaMemcpy(p_AOS, d_pAOS, size, cudaMemcpyDeviceToHost);
	cudaFree(d_pAOS);

		//cudaDeviceSynchronize(); //It may stall the GPU pipeline.

	//
//.............



	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("*** GPU3 - E:%d N:%d B:(%d,%d) M:constant,AOS : GPU Time taken = %.3fms\n", ELEM_PER_POINT, N_ELEM_LOG, BLOCK_HEIGHT, BLOCK_WIDTH, elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

void transform_points_SOA_with_constant( INOUT POINTS_SOA p_SOA, IN int n_points, IN int m )
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	

//.............
	POINTS_SOA d_pSOA;
	size_t size = N_POINTS * sizeof(float);
	for(int i=0;i<ELEM_PER_POINT;i++){
		cudaMalloc(&d_pSOA.elem[i], size);
		cudaMemcpy(d_pSOA.elem[i], p_SOA.elem[i], size, cudaMemcpyHostToDevice);
	}
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 dimGrid(ARRAY_2D_WIDTH/dimBlock.x, ARRAY_2D_HEIGHT/dimBlock.y);
		cudaEventRecord( start, 0 );
	TransformSOAwithConstantMemKernel <<< dimGrid, dimBlock >>> (d_pSOA, m);
		cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	for(int i=0;i<ELEM_PER_POINT;i++){
		cudaMemcpy(p_SOA.elem[i], d_pSOA.elem[i], size, cudaMemcpyDeviceToHost);
		cudaFree(d_pSOA.elem[i]);
	}

		//cudaDeviceSynchronize(); //It may stall the GPU pipeline.

	//
//.............



	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("*** GPU4 - E:%d N:%d B:(%d,%d) M:constant,SOA : GPU Time taken = %.3fms\n", ELEM_PER_POINT, N_ELEM_LOG, BLOCK_HEIGHT, BLOCK_WIDTH, elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


void generate_point_data( OUT POINT_ELEMENT *p_AOS, OUT POINTS_SOA *p_SOA, IN int n )
{
	int i;

	srand( ( unsigned int )time( NULL ) );

	for( i = 0; i < n; i++ )
	{
		for( int j = 0; j < ELEM_PER_POINT; ++j )
		{
			p_AOS[ i ].elem[ j ] = p_SOA->elem[ j ][ i ] = 100.0f*( ( float )rand() ) / RAND_MAX;
		}
	}
}

void generate_constant_data( IN int m )
{
	float *p_constant = new float[ m ];

	p_constant[ 0 ] = 0; // not used
	for( int i = 2; i <= m; ++i )
	{
		p_constant[ i-1 ] = 1.0f / ( float )i;
	}

	cudaMemcpyToSymbol( constantBuffer, p_constant, sizeof( float )* m );

	delete[] p_constant;
}

int main(void){

	float compute_time;
	int n_points, cutoff;
	POINT_ELEMENT *Points_AOS;
	POINTS_SOA Points_SOA;

	printf("ELEMENT PER POINT = %d\n\n", ELEM_PER_POINT);

	n_points = N_POINTS;
	cutoff = 1000;
	Points_AOS = new POINT_ELEMENT[N_POINTS];
	for( int i = 0; i < ELEM_PER_POINT; ++i )
	{
		Points_SOA.elem[ i ] = new float[ N_POINTS ];
	}
	generate_point_data(Points_AOS, &Points_SOA, n_points);
	generate_constant_data(cutoff);

	transform_points_AOS(Points_AOS, n_points, cutoff);

	transform_points_SOA(Points_SOA, n_points, cutoff);


	printf("\n");
	printf("--- AOS.10.x = %e, SOA.10.x = %e\n", Points_AOS[10].elem[0], Points_SOA.elem[0][10]);
	printf("--- AOS.20.y = %e, SOA.20.y = %e\n", Points_AOS[20].elem[1], Points_SOA.elem[1][20]);
	printf("\n");

	printf("\n///////////////////Second round///////////////////////\n\n");
	generate_point_data(Points_AOS, &Points_SOA, n_points);
	generate_constant_data(cutoff);
	transform_points_AOS(Points_AOS, n_points, cutoff);

	transform_points_SOA(Points_SOA, n_points, cutoff);

	printf("\n");
	printf("--- AOS.10.x = %e, SOA.10.x = %e\n", Points_AOS[10].elem[0], Points_SOA.elem[0][10]);
	printf("--- AOS.20.y = %e, SOA.20.y = %e\n", Points_AOS[20].elem[1], Points_SOA.elem[1][20]);
	printf("\n");

	printf( "\n///////////////////Constant first round///////////////////////\n\n" );
	generate_point_data(Points_AOS, &Points_SOA, n_points);
	generate_constant_data(cutoff);
	transform_points_AOS_with_constant(Points_AOS, n_points, cutoff);
	transform_points_SOA_with_constant(Points_SOA, n_points, cutoff);
	printf( "\n" );
	printf( "--- AOS.10.x = %e, SOA.10.x = %e\n", Points_AOS[ 10 ].elem[ 0 ], Points_SOA.elem[ 0 ][ 10 ] );
	printf( "--- AOS.20.y = %e, SOA.20.y = %e\n", Points_AOS[ 20 ].elem[ 1 ], Points_SOA.elem[ 1 ][ 20 ] );
	printf( "\n" );

	printf( "\n///////////////////Constant second round///////////////////////\n\n" );
	generate_point_data(Points_AOS, &Points_SOA, n_points);
	generate_constant_data(cutoff);
	transform_points_AOS_with_constant(Points_AOS, n_points, cutoff);
	transform_points_SOA_with_constant(Points_SOA, n_points, cutoff);
	
	printf( "\n" );
	printf( "--- AOS.10.x = %e, SOA.10.x = %e\n", Points_AOS[ 10 ].elem[ 0 ], Points_SOA.elem[ 0 ][ 10 ] );
	printf( "--- AOS.20.y = %e, SOA.20.y = %e\n", Points_AOS[ 20 ].elem[ 1 ], Points_SOA.elem[ 1 ][ 20 ] );
	printf( "\n" );


	return 0;
}