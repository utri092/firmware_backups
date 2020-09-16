#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_large.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w10.h"
#include "weights/b10.h"

//hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4;
    static const bool store_weights_in_bram = false;
    typedef ap_uint<32> accum_t;
    typedef Dense_bias_t bias_t;
    typedef Dense_weight_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    typedef ap_fixed<18,8> table_t;
};

struct config4 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 10;
    static const bool store_weights_in_bram = false;
    typedef ap_uint<32> accum_t;
    typedef Dense_bias_t bias_t;
    typedef Dense_weight_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    typedef ap_fixed<18,8> table_t;
};

struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 25;
    static const bool store_weights_in_bram = false;
    typedef ap_uint<32> accum_t;
    typedef Dense_bias_t bias_t;
    typedef Dense_weight_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    typedef ap_fixed<18,8> table_t;
};

struct config8 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 25;
    static const bool store_weights_in_bram = false;
    typedef ap_uint<32> accum_t;
    typedef Dense_bias_t bias_t;
    typedef Dense_weight_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    typedef ap_fixed<18,8> table_t;
};

struct config10 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_out = N_LAYER_10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 5;
    static const bool store_weights_in_bram = false;
    typedef ap_uint<32> accum_t;
    typedef Dense_bias_t bias_t;
    typedef Dense_weight_t weight_t;
    typedef ap_uint<1> index_t;
};

struct linear_config11 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 3;
    typedef ap_fixed<18,8> table_t;
};


#endif
