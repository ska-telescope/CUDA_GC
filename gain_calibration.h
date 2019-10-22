
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GAIN_CALIBRATION_H_
#define GAIN_CALIBRATION_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Speed of light
#ifndef C
	#define C 299792458.0
#endif

#ifndef SINGLE_PRECISION
	#define SINGLE_PRECISION 1
#endif

#if SINGLE_PRECISION
	#define PRECISION float
	#define PRECISION2 float2
	#define PRECISION3 float3
	#define PRECISION4 float4
	#define PI ((float) 3.141592654)
#else
	#define PRECISION double
	#define PRECISION2 double2
	#define PRECISION3 double3
	#define PRECISION4 double4
	#define PI ((double) 3.1415926535897931)
#endif

#if SINGLE_PRECISION
	#define SIN(x) sinf(x)
	#define COS(x) cosf(x)
	#define SINCOS(x, y, z) sincosf(x, y, z)
	#define ABS(x) fabs(x)
	#define SQRT(x) sqrtf(x)
	#define ROUND(x) roundf(x)
	#define CEIL(x) ceilf(x)
	#define LOG(x) logf(x)
	#define POW(x, y) powf(x, y)
	#define MAKE_PRECISION2(x,y) make_float2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)
#else
	#define SIN(x) sin(x)
	#define COS(x) cos(x)
	#define SINCOS(x, y, z) sincos(x, y, z)
	#define ABS(x) abs(x)
	#define SQRT(x) sqrt(x)
	#define ROUND(x) round(x)
	#define CEIL(x) ceil(x)
	#define LOG(x) log(x)
	#define POW(x, y) pow(x, y)
	#define MAKE_PRECISION2(x,y) make_double2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)
#endif

#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

typedef struct Config {
	int num_visibilities;
	const char *vis_src_file;
	const char *vis_dest_file;
	bool enable_right_ascension;
	int grid_size;
	double cell_size;
	double uv_scale;
	double frequency_hz;
	int gpu_max_threads_per_block;
	bool enable_messages;
	unsigned int num_recievers;
	unsigned int max_calibration_cycles;
} Config;

typedef struct Complex {
	PRECISION real;
	PRECISION imaginary;
} Complex;


typedef struct Visibility {
	PRECISION u;
	PRECISION v;
	PRECISION w;
} Visibility;

void init_config (Config *config);

void load_visibilities(Config *config, Visibility **visibilities, Complex **vis_intensity);

void execute_gain_calibration(Config *config, Complex *vis_measured, Complex *vis_predicted, Complex *gains_array, int2 *receiver_pairs);

void calculate_receiver_pairs(Config *config, int2 *receiver_pairs);

__global__ void update_gain_calibration(const PRECISION2 *vis_measured_array, const PRECISION2 *vis_predicted_array, 
	const PRECISION2 *gains_array, const int2 *receiver_pairs, PRECISION *A_array, PRECISION *Q_array, const int num_recievers, const int num_baselines);


__device__ PRECISION2 flip_for_i(const PRECISION2 z);

__device__ PRECISION2 flip_for_neg_i(const PRECISION2 z);

__device__ PRECISION2 complex_multiply(const PRECISION2 z1, const PRECISION2 z2);

__device__ PRECISION2 complex_divide(const PRECISION2 z1, const PRECISION2 z2);

__device__ PRECISION2 complex_subtract(const PRECISION2 z1, const PRECISION2 z2);

__device__ PRECISION2 complex_conjugate(const PRECISION2 z1);

static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

#endif /* GAIN_CALIBRATION_H_ */

#ifdef __cplusplus
}
#endif