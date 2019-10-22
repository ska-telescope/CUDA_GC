
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

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/time.h>

#include "gain_calibration.h"

int main(int argc, char **argv)
{
	printf("====================================================================\n");
	printf(">>> AUT HPC Research Laboratory - GAIN GALIBRATION (GPU VERSION) <<<\n");
	printf("====================================================================\n\n");

	if(SINGLE_PRECISION)
		printf(">>> INFO: Executing GC using single precision...\n\n");
	else
		printf(">>> INFO: Executing GC using double precision...\n\n");

	// Seed random from time
	srand(time(NULL));

	Config config;
	init_config(&config);

	Visibility *visibilities = NULL;
	Complex *vis_measured = NULL;
	load_visibilities(&config, &visibilities, &vis_measured);

	if(visibilities == NULL || vis_measured == NULL)
	{	
		printf(">>> ERROR: Visibility memory was unable to be allocated...\n\n");
		if(visibilities)       free(visibilities);
		if(vis_measured)       free(vis_measured);
		return EXIT_FAILURE;
	}
	//create gains_array and predicted
	Complex *gains_array = (Complex*) calloc(config.num_recievers, sizeof(Complex));

	for(int i=0;i<config.num_recievers;++i)
	{
		gains_array[i] = (Complex){ .real = 1.0, .imaginary = 0.0 };
	}

	//create function to load in predicted!
	Complex *vis_predicted = (Complex*) calloc(config.num_visibilities, sizeof(Complex));

	//SHALL WE STORE BASELINES IN CONFIG????
	int baselines = (config.num_recievers*(config.num_recievers-1))/2;

	int2 *receiver_pairs = (int2*) calloc(baselines, sizeof(int2));

	calculate_receiver_pairs(&config, receiver_pairs);



	execute_gain_calibration(&config, vis_measured, vis_predicted, gains_array, receiver_pairs);

	printf(">>> INFO: GAIN GALIBRATION operations complete, exiting...\n\n");

	return EXIT_SUCCESS;
}
