//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(stream<featuresSdCh> &inStream, stream<featuresSdCh> &outStream, unsigned int &max_size)
{

	//Set the HLS stream
	#pragma HLS interface axis port=inStream
	#pragma HLS interface axis port=outStream
	#pragma HLS interface s_axilite port=max_size bundle=CTRL_BUS
	// Set function interface to be controlled via zynq processing system
	#pragma HLS interface s_axilite port=return bundle=CTRL_BUS

	// input_layer_array_size will be given in myproject.h
	const unsigned int row_length = N_INPUT_1_1;

	// MAX Size of stream. Put in size according to requirements
	max_size = 0xFFFFFFFF;

	// Default if none specified

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<Dense_weight_t, 4>(w2, "w2.txt");
        nnet::load_weights_from_txt<Dense_bias_t, 2>(b2, "b2.txt");
        nnet::load_weights_from_txt<Dense_weight_t, 10>(w4, "w4.txt");
        nnet::load_weights_from_txt<Dense_bias_t, 5>(b4, "b4.txt");
        nnet::load_weights_from_txt<Dense_weight_t, 25>(w6, "w6.txt");
        nnet::load_weights_from_txt<Dense_bias_t, 5>(b6, "b6.txt");
        nnet::load_weights_from_txt<Dense_weight_t, 25>(w8, "w8.txt");
        nnet::load_weights_from_txt<Dense_bias_t, 5>(b8, "b8.txt");
        nnet::load_weights_from_txt<Dense_weight_t, 5>(w10, "w10.txt");
        nnet::load_weights_from_txt<Dense_bias_t, 1>(b10, "b10.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

	unsigned int i;
	for(int i = 0;  i <= (max_size -row_length); i+=row_length)
	{
		#pragma HLS pipeline

		if(inStream.empty()){
			break;
		}else{
			input_t input1[row_length];
			result_t layer11_out[N_LAYER_10];

			// Axi stream data type in myproject.h for reading/writing to input/output streams respectively
			featuresSdCh valIn, valOut;

			unsigned int j;
			//Read inputs from input stream and puts in hls4ml generated input array
			for(j = 0 ; j <  row_length; j++){
				#pragma HLS unroll

				inStream.read(valIn);

				input1[j] = valIn.data;
			}

			layer2_t layer2_out[N_LAYER_2];
			#pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
			nnet::dense_latency<input_t, layer2_t, config2>(input1, layer2_out, w2, b2);

			layer3_t layer3_out[N_LAYER_2];
			#pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
			nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out);

			layer4_t layer4_out[N_LAYER_4];
			#pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
			nnet::dense_latency<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4);

			layer5_t layer5_out[N_LAYER_4];
			#pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
			nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out);

			layer6_t layer6_out[N_LAYER_6];
			#pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
			nnet::dense_latency<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

			layer7_t layer7_out[N_LAYER_6];
			#pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
			nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out);

			layer8_t layer8_out[N_LAYER_8];
			#pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
			nnet::dense_latency<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8);

			layer9_t layer9_out[N_LAYER_8];
			#pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
			nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out);

			layer10_t layer10_out[N_LAYER_10];
			#pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
			nnet::dense_latency<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10);

			nnet::linear<layer10_t, result_t, linear_config11>(layer10_out, layer11_out);

			// Write to output Stream after NN functions
			// Output layer returns an array of size 1 in this example
			valOut.data = layer11_out[0];

			valOut.keep = valIn.keep;
			valOut.strb = valIn.strb;
			valOut.user = valIn.user;
			valOut.id = valIn.id;
			valOut.dest = valIn.dest;

			valOut.last = valIn.last;

			outStream.write(valOut);

		}
	}
}
