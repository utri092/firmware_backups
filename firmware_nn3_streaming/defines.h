#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 2
#define N_LAYER_2 2
#define N_LAYER_4 5
#define N_LAYER_6 5
#define N_LAYER_8 5
#define N_LAYER_10 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_uint<32> model_default_t;
typedef ap_uint<32> input_t;
typedef ap_fixed<64,32> layer2_t;
typedef ap_fixed<64,32> Dense_weight_t;
typedef ap_fixed<64,32> Dense_bias_t;
typedef ap_uint<32> layer3_t;
typedef ap_fixed<64,32> layer4_t;
typedef ap_uint<32> layer5_t;
typedef ap_fixed<64,32> layer6_t;
typedef ap_uint<32> layer7_t;
typedef ap_fixed<64,32> layer8_t;
typedef ap_uint<32> layer9_t;
typedef ap_fixed<64,32> layer10_t;
typedef ap_uint<32> result_t;

#endif
