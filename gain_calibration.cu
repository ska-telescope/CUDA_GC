
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "gain_calibration.h"

//IMPORTANT: Modify configuration for target GPU and DFT
void init_config(Config *config)
{
	// Toggle whether right ascension should be enabled (observation dependant)
	config->enable_right_ascension = true;

	// Number of visibilities per source
	config->num_visibilities = 2000000;

	config->grid_size = 18000;

	// Source of Visibilities
	config->vis_src_file    = "../el82-70.txt";

	config->vis_dest_file 	= "../el82-70_output.txt";

	// Dimension of Fourier domain grid
	config->grid_size = 18000;

	// Fourier domain grid cell size in radians
	config->cell_size = 6.39708380288950e-6;

	// Frequency of visibility uvw terms
	config->frequency_hz = 100e6;

	// Scalar for visibility coordinates
	config->uv_scale = config->grid_size * config->cell_size;

	// Number of CUDA threads per block - this is GPU specific
	config->gpu_max_threads_per_block = 1024;

	config->num_recievers = 512;

	config->max_calibration_cycles = 10;
}


void calculate_receiver_pairs(Config *config, int2 *receiver_pairs)
{
	int a = 0;
	int b = 1;
	int baselines = (config->num_recievers*(config->num_recievers-1))/2;

	for(int i=0;i<baselines;++i)
	{
		//printf(">>>> CREATING RECEIVER PAIR (%d,%d) \n",a,b);
		receiver_pairs[i].x = a;
		receiver_pairs[i].y = b;

		b++;
		if(b>=config->num_recievers)
		{
			a++;
			b = a+1;
		}
	}

}

void execute_gain_calibration(Config *config, Complex *vis_measured, Complex *vis_predicted, 
								Complex *gains_array, int2 *receiver_pairs)
{

	int baselines = (config->num_recievers*(config->num_recievers-1))/2;

	printf("UPDATE >>> EXECUTING GAIN CALIBRATION for %d recievers with total baselines of %d...\n\n",config->num_recievers, baselines);

	PRECISION2 *device_predicted;
	PRECISION2 *device_measured;
	PRECISION2 *device_gains;
	int2 *device_receiver_pairs;


	//COPY VIS and GAINS To GPU!
	// CUDA_CHECK_RETURN(cudaMalloc(&device_visibilities,  sizeof(PRECISION3) * baselines));
	// CUDA_CHECK_RETURN(cudaMemcpy(device_visibilities, visibilities, baselines * sizeof(PRECISION3), cudaMemcpyHostToDevice));
	// cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&device_measured,  sizeof(PRECISION2) * baselines));
	CUDA_CHECK_RETURN(cudaMemcpy(device_measured, vis_measured, baselines * sizeof(PRECISION2), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&device_predicted,  sizeof(PRECISION2) * baselines));
	CUDA_CHECK_RETURN(cudaMemcpy(device_predicted, vis_predicted, baselines * sizeof(PRECISION2), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&device_gains,  sizeof(PRECISION2) * config->num_recievers));
	CUDA_CHECK_RETURN(cudaMemcpy(device_gains, gains_array, config->num_recievers * sizeof(PRECISION2), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&device_receiver_pairs,  sizeof(int2) * baselines));
	CUDA_CHECK_RETURN(cudaMemcpy(device_receiver_pairs, receiver_pairs, baselines * sizeof(int2), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	PRECISION *Q_array;
	PRECISION *A_array;

	//intit Q and A Array for reuse;
	CUDA_CHECK_RETURN(cudaMalloc(&Q_array,  sizeof(PRECISION) * 2 * config->num_recievers));
	CUDA_CHECK_RETURN(cudaMalloc(&A_array,  sizeof(PRECISION) * 2 * config->num_recievers * 2 * config->num_recievers));


	//SET CUDA WORK PLAN:
	int max_threads_per_block = min(config->gpu_max_threads_per_block, baselines);
	int num_blocks = (int) ceil((double) baselines / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);


	for(int i=0;i<config->max_calibration_cycles;++i)
	{

 		//EXECUTE CUDA KERNEL UPDATING Q AND A array (NEED ATOMIC ACCESS!)
		update_gain_calibration<<<kernel_blocks, kernel_threads>>>(
				device_measured,
				device_predicted,
				device_gains,
				device_receiver_pairs,
				A_array,
				Q_array,
				config->num_recievers,
				baselines
		);
		cudaDeviceSynchronize();	

		//NOW DO THE SVD and Gauss-Newton to calculate Delat and update Gains array (G = G+DELTA)
		execute_calibration_SVD(config, A_array);

		//RESET Q AND A ARRRAY FOR NEXT CYCLE
		CUDA_CHECK_RETURN(cudaMemset(Q_array, 0, sizeof(PRECISION) * 2 * config->num_recievers));
		CUDA_CHECK_RETURN(cudaMemset(A_array, 0, sizeof(PRECISION) * 2 * config->num_recievers * 2 * config->num_recievers));
	}

	CUDA_CHECK_RETURN(cudaFree(Q_array));
	CUDA_CHECK_RETURN(cudaFree(A_array));


	//Should freee other memory for now!
	CUDA_CHECK_RETURN(cudaFree(device_predicted));
	CUDA_CHECK_RETURN(cudaFree(device_measured));
	CUDA_CHECK_RETURN(cudaFree(device_gains));
	CUDA_CHECK_RETURN(cudaFree(device_receiver_pairs));
	//CUDA_CHECK_RETURN(cudaFree(device_visibilities));
}

void execute_SVD(Config *config, PRECISION *d_A)
{
   	cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;

    const int m = 2 * config->num_recievers;
    const int n = 2 * config->num_recievers;
    const int lda = m;

    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    double *d_work = NULL; /* devie workspace for gesvdj */
    int info = 0;        /* host copy of error info */

	/* configuration of gesvdj  */
    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 0; /* econ = 1 for economy size */

    /* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;

    /* step 1: create cusolver handle, bind a stream */
    CUDA_SOLVER_CHECK_RETURN(cusolverDnCreate(&cusolverH));
    CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_SOLVER_CHECK_RETURN(cusolverDnSetStream(cusolverH, stream));

	/* step 2: configuration of gesvdj */
    CUDA_CHECK_RETURN(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    /* default value of tolerance is machine zero */
    CUDA_CHECK_RETURN(cusolverDnXgesvdjSetTolerance(gesvdj_params,tol));

	/* default value of max. sweeps is 100 */
    CUDA_CHECK_RETURN(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params,max_sweeps));

    // Allocate device memory for U, V, S
	PRECISION *d_U = NULL;
	PRECISION *d_V = NULL;
	PRECISION *d_S = NULL;

	CUDA_CHECK_RETURN(cudaMalloc(&d_U, sizeof(PRECISION) * m * n));
	CUDA_CHECK_RETURN(cudaMalloc(&d_V, sizeof(PRECISION) * m * n));
	CUDA_CHECK_RETURN(cudaMalloc(&d_S, sizeof(PRECISION) * m));
	CUDA_CHECK_RETURN(cudaMalloc(&d_info, sizeof(int)));

	/* step 4: query workspace of SVD */
	CUDA_CHECK_RETURN(cusolverDnDgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m,    /* nubmer of rows of A, 0 <= m */
        n,    /* number of columns of A, 0 <= n  */
        d_A,  /* m-by-n */
        lda,  /* leading dimension of A */
        d_S,  /* min(m,n) */
              /* the singular values in descending order */
        d_U,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        lda,  /* leading dimension of U, ldu >= max(1,m) */
        d_V,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        lda,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork,
        gesvdj_params));

	/* step 5: compute SVD */
	CUDA_CHECK_RETURN(cusolverDnDgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_A,   /* m-by-n */
        lda,   /* leading dimension of A */
        d_S,   /* min(m,n)  */
               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */
               /* m-by-min(m,n) if econ = 1 */
        lda,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */
               /* n-by-min(m,n) if econ = 1  */
        lda,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	if ( 0 == info ){
        printf("gesvdj converges \n");
    }else if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }else{
        printf("WARNING: info = %d : gesvdj does not converge \n", info );
    }

    CUDA_CHECK_RETURN(cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps));
    CUDA_CHECK_RETURN(cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual));

    printf("residual |A - U*S*V**H|_F = %E \n", residual);
    printf("number of executed sweeps = %d \n", executed_sweeps);

    /*  free resources  */
    if (d_S    ) CUDA_CHECK_RETURN(cudaFree(d_S));
    if (d_U    ) CUDA_CHECK_RETURN(cudaFree(d_U));
    if (d_V    ) CUDA_CHECK_RETURN(cudaFree(d_V));
    if (d_info ) CUDA_CHECK_RETURN(cudaFree(d_info));
    if (d_work ) CUDA_CHECK_RETURN(cudaFree(d_work));

    if (cusolverH    ) CUDA_SOLVER_CHECK_RETURN(cusolverDnDestroy(cusolverH));
    if (stream       ) CUDA_CHECK_RETURN(cudaStreamDestroy(stream));
    if (gesvdj_params) CUDA_SOLVER_CHECK_RETURN(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

__global__ void update_gain_calibration(const PRECISION2 *vis_measured_array, const PRECISION2 *vis_predicted_array, 
	const PRECISION2 *gains_array, const int2 *receiver_pairs, PRECISION *A_array, PRECISION *Q_array, const int num_recievers, const int num_baselines)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index >= num_baselines)
		return;

	PRECISION2 vis_measured = vis_measured_array[index];
	PRECISION2 vis_predicted = vis_predicted_array[index];

	int2 antennae = receiver_pairs[index];

	PRECISION2 gainA = gains_array[antennae.x];
	//only need gainB as conjugate??
	PRECISION2 gainB_conjugate = complex_conjugate(gains_array[antennae.y]);

	//NOTE do not treat residual as a COMPLEX!!!!! (2 reals)
	PRECISION2 residual = complex_subtract(vis_measured, complex_multiply(vis_predicted,complex_multiply(gainA, gainB_conjugate)));

	//CALCULATE Partial Derivatives

	PRECISION2 part_respect_to_real_gain_a = complex_multiply(vis_predicted, gainB_conjugate);

	PRECISION2 part_respect_to_imag_gain_a = flip_for_i(complex_multiply(vis_predicted, gainB_conjugate));

	PRECISION2 part_respect_to_real_gain_b = complex_multiply(vis_predicted,gainA);

	PRECISION2 part_respect_to_imag_gain_b = flip_for_neg_i(complex_multiply(vis_predicted, gainA));

	//Calculate Q[2a],Q[2a+1],Q[2b],Q[2b+1] arrays - In this order... NEED ATOMIC UPDATE 
	double qValue = part_respect_to_real_gain_a.x * residual.x 
					+ part_respect_to_real_gain_a.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.x]), qValue);

	qValue = part_respect_to_imag_gain_a.x * residual.x 
					+ part_respect_to_imag_gain_a.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.x+1]), qValue);

	qValue = part_respect_to_real_gain_b.x * residual.x 
					+ part_respect_to_real_gain_b.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.y]), qValue);

	qValue = part_respect_to_imag_gain_b.x * residual.x 
					+ part_respect_to_imag_gain_b.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.y+1]), qValue);

	//CALCULATE JAcobian product on A matrix... 2a2a, 2a2a+1, 2a2b, 2a2b+1
	//2a2a
	double aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_a.x + 
					part_respect_to_real_gain_a.y * part_respect_to_real_gain_a.y; 
	
	int aIndex = (2 *  antennae.x * num_recievers) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2a+1,
	aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_a.x + 
					part_respect_to_real_gain_a.y * part_respect_to_imag_gain_a.y; 

	aIndex = (2 *  antennae.x * num_recievers) + (2 * antennae.x + 1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2b
	aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_b.x + 
					part_respect_to_real_gain_a.y * part_respect_to_real_gain_b.y;

	aIndex = (2 *  antennae.x * num_recievers) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2b+1
	aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_real_gain_a.y * part_respect_to_imag_gain_b.y;

	aIndex = (2 *  antennae.x * num_recievers) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);
	//CACLUATE JAcobian product on A matrix... [2a+1,2a], [2a+1,2a+1], [2a+1,2b], [2a+1,2b+1]
	//2a+1,2a
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_a.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_real_gain_a.y; 

	aIndex = ((2 *  antennae.x+1) * num_recievers) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2a+1
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_a.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_a.y; 

	aIndex = ((2 *  antennae.x+1) * num_recievers) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2b
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_b.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_real_gain_b.y;

	aIndex = ((2 *  antennae.x+1) * num_recievers) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2b+1
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_b.y;

	aIndex = ((2 *  antennae.x+1) * num_recievers) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);


	//CACLUATE JAcobian product on A matrix... [2b,2a], [2b,2a+1], [2b,2b], [2b,2b+1]
	//2b,2a
	aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_a.x + 
					part_respect_to_real_gain_b.y * part_respect_to_real_gain_a.y; 
	
	aIndex = (2 *  antennae.y * num_recievers) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2b,2a+1
	aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_a.x + 
			 		part_respect_to_real_gain_b.y * part_respect_to_imag_gain_a.y;

	aIndex = (2 *  antennae.y * num_recievers) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);
	
	//2b,2b
	aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_b.x + 
					part_respect_to_real_gain_b.y * part_respect_to_real_gain_b.y;
	
	aIndex = (2 *  antennae.y * num_recievers) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b, 2b+1
	aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_real_gain_b.y * part_respect_to_imag_gain_b.y;

	aIndex = (2 *  antennae.y * num_recievers) + (2 * antennae.y + 1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//CALCULATE JAcobian product on A matrix... [2b+1,2a], [2b+1,2a+1], [2b+1,2b], [2b+1,2b+1]
	//2b+1,2a
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_a.x + 
					part_respect_to_imag_gain_b.y * part_respect_to_real_gain_a.y; 

	aIndex = ((2 *  antennae.y+1) * num_recievers) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1,2a+1
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_a.x+ 
					 part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_a.y;

	aIndex = ((2 *  antennae.y+1) * num_recievers) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1,2b
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_b.x+ 
					part_respect_to_imag_gain_b.y * part_respect_to_real_gain_b.y;

	aIndex = ((2 *  antennae.y+1) * num_recievers) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1, 2b+1

	aValue = part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_b.y; 

	aIndex = ((2 *  antennae.y+1) * num_recievers) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);

}

__device__ PRECISION2 flip_for_i(const PRECISION2 z)
{
	return MAKE_PRECISION2(-z.y, z.x);
}

__device__ PRECISION2 flip_for_neg_i(const PRECISION2 z)
{
	return MAKE_PRECISION2(z.y, -z.x);
}


// __device__ PRECISION2 partial_respect_to_real_gain_a(const PRECISION2 v, const PRECISION2 g)
// {
// 	return complex_multiply()

// }

__device__ PRECISION2 complex_multiply(const PRECISION2 z1, const PRECISION2 z2)
{
    return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}

// http://mathworld.wolfram.com/ComplexDivision.html
__device__ PRECISION2 complex_divide(const PRECISION2 z1, const PRECISION2 z2)
{
    PRECISION a = z1.x * z2.x + z1.y * z2.y;
    PRECISION b = z1.y * z2.x - z1.x * z2.y;
    PRECISION c = z2.x * z2.x + z2.y * z2.y;
    return MAKE_PRECISION2(a / c, b / c);
}

__device__ PRECISION2 complex_subtract(const PRECISION2 z1, const PRECISION2 z2)
{
    return MAKE_PRECISION2(z1.x - z2.x, z1.y - z2.y);
}

__device__ PRECISION2 complex_conjugate(const PRECISION2 z1)
{
    return MAKE_PRECISION2(z1.x, -z1.y);
}

void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity)
{
		
	printf(">>> UPDATE: Using Visibilities from file...\n\n");

	FILE *file = fopen(config->vis_src_file, "r");
	if(file == NULL)
	{
		printf(">>> ERROR: Unable to locate visibilities file...\n\n");
		return;
	}

	// Reading in the counter for number of visibilities
	fscanf(file, "%d\n", &(config->num_visibilities));

	*visibilities = (Visibility*) calloc(config->num_visibilities, sizeof(Visibility));
	*vis_intensity =  (Complex*) calloc(config->num_visibilities, sizeof(Complex));

	// File found, but was memory allocated?
	if(*visibilities == NULL || *vis_intensity == NULL)
	{
		printf(">>> ERROR: Unable to allocate memory for visibilities...\n\n");
		if(file) fclose(file);
		if(*visibilities) free(*visibilities);
		if(*vis_intensity) free(*vis_intensity);
		return;
	}

	PRECISION u = 0.0;
	PRECISION v = 0.0;
	PRECISION w = 0.0;
	Complex brightness;
	PRECISION intensity = 0.0;

	// Used to scale visibility coordinates from wavelengths
	// to meters
	PRECISION wavelength_to_meters = config->frequency_hz / C;
	PRECISION right_asc_factor = (config->enable_right_ascension) ? -1.0 : 1.0;

	// Read in n number of visibilities
	for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
	{
		// Read in provided visibility attributes
		// u, v, w, brightness (real), brightness (imag), intensity
#if SINGLE_PRECISION
		fscanf(file, "%f %f %f %f %f %f\n", &u, &v, &w, 
			&(brightness.real), &(brightness.imaginary), &intensity);
#else
		fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &u, &v, &w, 
			&(brightness.real), &(brightness.imaginary), &intensity);
#endif

		u *=  right_asc_factor;
		w *=  right_asc_factor;

		(*visibilities)[vis_indx] = (Visibility) {
			.u = u * wavelength_to_meters,
			.v = v * wavelength_to_meters,
			.w = w * wavelength_to_meters
		};
	}

	// Clean up
	fclose(file);
		printf(">>> UPDATE: Successfully loaded %d visibilities from file...\n\n",config->num_visibilities);
}

void save_visibilities(Config *config, Visibility *visibilities, Complex *vis_intensity)
{
	// Save visibilities to file
	FILE *file = fopen(config->vis_dest_file, "w");
	// Unable to open file
	if(file == NULL)
	{
		printf(">>> ERROR: Unable to save visibilities to file...\n\n");
		return;
	}

	if(config->enable_messages)
		printf(">>> UPDATE: Writing visibilities to file...\n\n");

	// Record number of visibilities
	fprintf(file, "%d\n", config->num_visibilities);
	
	// Used to scale visibility coordinates from meters to
	// wavelengths (useful for gridding, inverse DFT etc.)
	PRECISION meters_to_wavelengths = config->frequency_hz / C;

	// Record individual visibilities
	for(int vis_indx = 0; vis_indx < config->num_visibilities; ++vis_indx)
	{

		visibilities[vis_indx].u /= meters_to_wavelengths;
		visibilities[vis_indx].v /= meters_to_wavelengths;
		visibilities[vis_indx].w /= meters_to_wavelengths;

		if(config->enable_right_ascension)
		{
			visibilities[vis_indx].u *= -1.0;
			visibilities[vis_indx].w *= -1.0;
		}

		// u, v, w, real, imag, weight (intensity)
#if SINGLE_PRECISION
		fprintf(file, "%f %f %f %f %f %f\n", visibilities[vis_indx].u,
			visibilities[vis_indx].v, visibilities[vis_indx].w,
			vis_intensity[vis_indx].real, vis_intensity[vis_indx].imaginary, 1.0);
#else
		fprintf(file, "%lf %lf %lf %lf %lf %lf\n", visibilities[vis_indx].u,
			visibilities[vis_indx].v, visibilities[vis_indx].w,
			vis_intensity[vis_indx].real, vis_intensity[vis_indx].imaginary, 1.0);
#endif
	}

	// Clean up
	fclose(file);
	if(config->enable_messages)
		printf(">>> UPDATE: Completed writing of visibilities to file...\n\n");
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

void check_cuda_solver_error_aux(const char *file, unsigned line, const char *statement, cusolverStatus_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}
