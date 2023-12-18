/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Dec 18 22:33:03 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "15e4bdee5d389c4dccad1f1b517ae09a"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Mon Dec 18 22:33:03 2023"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  batch_normalization_1_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  batch_normalization_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 768, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  number_output_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 160, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  number_output_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  activation_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  activation_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  batch_normalization_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  activation_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10560, AI_STATIC)
/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21120, AI_STATIC)
/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  activation_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21120, AI_STATIC)
/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21120, AI_STATIC)
/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 42240, AI_STATIC)
/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  activation_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 42240, AI_STATIC)
/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 42240, AI_STATIC)
/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  global_average_pooling2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  number_output_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  number_output_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 5, AI_STATIC)
/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  depthwise_conv2d_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_conv2d_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &depthwise_conv2d_conv2d_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_weights, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 16, 4, 4, 16), AI_STRIDE_INIT(4, 4, 64, 1024, 4096),
  1, &conv2d_1_conv2d_weights_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_1_conv2d_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_1_scale, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &batch_normalization_1_scale_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_1_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &batch_normalization_1_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 16, 6, 6, 16), AI_STRIDE_INIT(4, 4, 64, 1024, 6144),
  1, &conv2d_2_conv2d_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_2_conv2d_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_weights, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 16), AI_STRIDE_INIT(4, 1,16, 16, 16),
  1, &depthwise_conv2d_1_conv2d_weights_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &depthwise_conv2d_1_conv2d_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_weights, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &conv2d_3_conv2d_weights_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_3_conv2d_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 32), AI_STRIDE_INIT(4, 1,32, 32, 32),
  1, &depthwise_conv2d_2_conv2d_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &depthwise_conv2d_2_conv2d_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 8192),
  1, &conv2d_4_conv2d_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_4_conv2d_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 64), AI_STRIDE_INIT(4, 1,64, 64, 64),
  1, &depthwise_conv2d_3_conv2d_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &depthwise_conv2d_3_conv2d_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 64, 128, 1, 1), AI_STRIDE_INIT(4, 4, 256, 32768, 32768),
  1, &dense_dense_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &dense_dense_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 128, 64, 1, 1), AI_STRIDE_INIT(4, 4, 512, 32768, 32768),
  1, &dense_1_dense_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_1_dense_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 64, 32, 1, 1), AI_STRIDE_INIT(4, 4, 256, 8192, 8192),
  1, &dense_2_dense_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_2_dense_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 32, 24), AI_STRIDE_INIT(4, 4, 4, 4, 128),
  1, &input_1_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  number_output_dense_weights, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 32, 5, 1, 1), AI_STRIDE_INIT(4, 4, 128, 640, 640),
  1, &number_output_dense_weights_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 32, 24), AI_STRIDE_INIT(4, 4, 4, 64, 2048),
  1, &conv2d_conv2d_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  number_output_dense_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &number_output_dense_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  activation_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 32, 24), AI_STRIDE_INIT(4, 4, 4, 64, 2048),
  1, &activation_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_conv2d_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &depthwise_conv2d_conv2d_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_conv2d_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &conv2d_1_conv2d_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  activation_1_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &activation_1_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_1_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &batch_normalization_1_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_conv2d_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &conv2d_2_conv2d_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  activation_2_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &activation_2_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 30, 22), AI_STRIDE_INIT(4, 4, 4, 64, 1920),
  1, &depthwise_conv2d_1_conv2d_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_conv2d_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 30, 22), AI_STRIDE_INIT(4, 4, 4, 128, 3840),
  1, &conv2d_3_conv2d_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  activation_3_output, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 30, 22), AI_STRIDE_INIT(4, 4, 4, 128, 3840),
  1, &activation_3_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 30, 22), AI_STRIDE_INIT(4, 4, 4, 128, 3840),
  1, &depthwise_conv2d_2_conv2d_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_conv2d_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 30, 22), AI_STRIDE_INIT(4, 4, 4, 256, 7680),
  1, &conv2d_4_conv2d_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  activation_4_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 30, 22), AI_STRIDE_INIT(4, 4, 4, 256, 7680),
  1, &activation_4_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 30, 22), AI_STRIDE_INIT(4, 4, 4, 256, 7680),
  1, &depthwise_conv2d_3_conv2d_output_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  global_average_pooling2d_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &global_average_pooling2d_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &dense_dense_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  dense_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &dense_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_1_dense_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_1_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_2_dense_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &dense_2_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  number_output_dense_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &number_output_dense_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  number_output_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &number_output_output_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_weights, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 2, 16), AI_STRIDE_INIT(4, 4, 4, 64, 128),
  1, &conv2d_conv2d_weights_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_conv2d_bias, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_conv2d_bias_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_conv2d_weights, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1,16, 16, 16),
  1, &depthwise_conv2d_conv2d_weights_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  number_output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &number_output_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &number_output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  number_output_layer, 27,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &number_output_chain,
  NULL, &number_output_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  number_output_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &number_output_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &number_output_dense_weights, &number_output_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  number_output_dense_layer, 27,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &number_output_dense_chain,
  NULL, &number_output_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_2_chain,
  NULL, &number_output_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_2_dense_weights, &dense_2_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_dense_layer, 25,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_2_dense_chain,
  NULL, &dense_2_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_1_chain,
  NULL, &dense_2_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_1_dense_weights, &dense_1_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_dense_layer, 23,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_1_dense_chain,
  NULL, &dense_1_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_chain,
  NULL, &dense_1_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_dense_weights, &dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_dense_layer, 21,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_dense_chain,
  NULL, &dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  global_average_pooling2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling2d_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  global_average_pooling2d_layer, 20,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &global_average_pooling2d_chain,
  NULL, &dense_dense_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(30, 22), 
  .pool_stride = AI_SHAPE_2D_INIT(30, 22), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &depthwise_conv2d_3_conv2d_weights, &depthwise_conv2d_3_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  depthwise_conv2d_3_conv2d_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &depthwise_conv2d_3_conv2d_chain,
  NULL, &global_average_pooling2d_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_4_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_4_chain,
  NULL, &depthwise_conv2d_3_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_conv2d_weights, &conv2d_4_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_conv2d_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_4_conv2d_chain,
  NULL, &activation_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &depthwise_conv2d_2_conv2d_weights, &depthwise_conv2d_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  depthwise_conv2d_2_conv2d_layer, 15,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &depthwise_conv2d_2_conv2d_chain,
  NULL, &conv2d_4_conv2d_layer, AI_STATIC, 
  .groups = 32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_3_layer, 13,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_3_chain,
  NULL, &depthwise_conv2d_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_conv2d_weights, &conv2d_3_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_conv2d_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_3_conv2d_chain,
  NULL, &activation_3_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &depthwise_conv2d_1_conv2d_weights, &depthwise_conv2d_1_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  depthwise_conv2d_1_conv2d_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &depthwise_conv2d_1_conv2d_chain,
  NULL, &conv2d_3_conv2d_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_2_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_2_chain,
  NULL, &depthwise_conv2d_1_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &batch_normalization_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_conv2d_weights, &conv2d_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_conv2d_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_2_conv2d_chain,
  NULL, &activation_2_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 2, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  batch_normalization_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &batch_normalization_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &batch_normalization_1_scale, &batch_normalization_1_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  batch_normalization_1_layer, 7,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &batch_normalization_1_chain,
  NULL, &conv2d_2_conv2d_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_1_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_1_chain,
  NULL, &batch_normalization_1_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_conv2d_weights, &conv2d_1_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_conv2d_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_1_conv2d_chain,
  NULL, &activation_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  depthwise_conv2d_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &depthwise_conv2d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &depthwise_conv2d_conv2d_weights, &depthwise_conv2d_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  depthwise_conv2d_conv2d_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &depthwise_conv2d_conv2d_chain,
  NULL, &conv2d_1_conv2d_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_chain,
  NULL, &depthwise_conv2d_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_conv2d_weights, &conv2d_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_conv2d_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_conv2d_chain,
  NULL, &activation_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 157652, 1, 1),
    157652, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 337920, 1, 1),
    337920, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &number_output_output),
  &conv2d_conv2d_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 157652, 1, 1),
      157652, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 337920, 1, 1),
      337920, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &number_output_output),
  &conv2d_conv2d_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_1_output_array.data = AI_PTR(g_network_activations_map[0] + 165888);
    input_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 165888);
    
    conv2d_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    conv2d_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    activation_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    activation_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    depthwise_conv2d_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 126720);
    depthwise_conv2d_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 126720);
    
    conv2d_1_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    conv2d_1_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    activation_1_output_array.data = AI_PTR(g_network_activations_map[0] + 126720);
    activation_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 126720);
    
    batch_normalization_1_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    batch_normalization_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    conv2d_2_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 126720);
    conv2d_2_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 126720);
    
    activation_2_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    activation_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    depthwise_conv2d_1_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 126720);
    depthwise_conv2d_1_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 126720);
    
    conv2d_3_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    conv2d_3_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    activation_3_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    activation_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    depthwise_conv2d_2_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 84480);
    depthwise_conv2d_2_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84480);
    
    conv2d_4_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    conv2d_4_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    activation_4_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    activation_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    depthwise_conv2d_3_conv2d_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    depthwise_conv2d_3_conv2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    global_average_pooling2d_output_array.data = AI_PTR(g_network_activations_map[0] + 168960);
    global_average_pooling2d_output_array.data_start = AI_PTR(g_network_activations_map[0] + 168960);
    
    dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_output_array.data = AI_PTR(g_network_activations_map[0] + 512);
    dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    
    dense_1_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_1_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_1_output_array.data = AI_PTR(g_network_activations_map[0] + 256);
    dense_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 256);
    
    dense_2_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_2_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_2_output_array.data = AI_PTR(g_network_activations_map[0] + 128);
    dense_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 128);
    
    number_output_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    number_output_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    number_output_output_array.data = AI_PTR(g_network_activations_map[0] + 20);
    number_output_output_array.data_start = AI_PTR(g_network_activations_map[0] + 20);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    depthwise_conv2d_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 0);
    depthwise_conv2d_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    conv2d_1_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 64);
    conv2d_1_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 64);
    
    conv2d_1_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 16448);
    conv2d_1_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16448);
    
    batch_normalization_1_scale_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_1_scale_array.data = AI_PTR(g_network_weights_map[0] + 16512);
    batch_normalization_1_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 16512);
    
    batch_normalization_1_bias_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 16576);
    batch_normalization_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16576);
    
    conv2d_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 16640);
    conv2d_2_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 16640);
    
    conv2d_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 53504);
    conv2d_2_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 53504);
    
    depthwise_conv2d_1_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_1_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 53568);
    depthwise_conv2d_1_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 53568);
    
    depthwise_conv2d_1_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_1_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 53632);
    depthwise_conv2d_1_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 53632);
    
    conv2d_3_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 53696);
    conv2d_3_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 53696);
    
    conv2d_3_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 72128);
    conv2d_3_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 72128);
    
    depthwise_conv2d_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_2_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 72256);
    depthwise_conv2d_2_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 72256);
    
    depthwise_conv2d_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_2_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 72384);
    depthwise_conv2d_2_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 72384);
    
    conv2d_4_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 72512);
    conv2d_4_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 72512);
    
    conv2d_4_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 80704);
    conv2d_4_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 80704);
    
    depthwise_conv2d_3_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_3_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 80960);
    depthwise_conv2d_3_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 80960);
    
    depthwise_conv2d_3_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_3_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 81216);
    depthwise_conv2d_3_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 81216);
    
    dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 81472);
    dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 81472);
    
    dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 114240);
    dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 114240);
    
    dense_1_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 114752);
    dense_1_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 114752);
    
    dense_1_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 147520);
    dense_1_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 147520);
    
    dense_2_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_2_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 147776);
    dense_2_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 147776);
    
    dense_2_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_2_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 155968);
    dense_2_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 155968);
    
    number_output_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    number_output_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 156096);
    number_output_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 156096);
    
    number_output_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    number_output_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 156736);
    number_output_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 156736);
    
    conv2d_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 156756);
    conv2d_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 156756);
    
    conv2d_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_conv2d_bias_array.data = AI_PTR(g_network_weights_map[0] + 157012);
    conv2d_conv2d_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 157012);
    
    depthwise_conv2d_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_conv2d_weights_array.data = AI_PTR(g_network_weights_map[0] + 157076);
    depthwise_conv2d_conv2d_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 157076);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 13576512,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 13576512,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

