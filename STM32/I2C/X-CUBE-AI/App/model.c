/**
  ******************************************************************************
  * @file    model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sun Feb  4 19:23:28 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "model.h"
#include "model_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_model
 
#undef AI_MODEL_MODEL_SIGNATURE
#define AI_MODEL_MODEL_SIGNATURE     "942d55d93af718ec6f2c019a46bf37f6"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Sun Feb  4 19:23:28 2024"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_MODEL_N_BATCHES
#define AI_MODEL_N_BATCHES         (1)

static ai_ptr g_model_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_model_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  activation_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  block_2_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  block_2_output_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  block_3_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  activation_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  block_3_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  block_4_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  activation_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  block_4_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  block_4_output_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  block_5_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  activation_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  block_5_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  block_6_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  activation_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  block_6_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  global_average_pooling2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  number_output_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  number_output_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 5, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  block_1_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  block_1_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  block_1_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  block_1_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  block_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  block_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  block_2_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  block_2_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  block_3_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  block_3_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  block_3_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  block_3_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  block_4_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  block_4_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  block_4_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  block_4_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  block_5_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  block_5_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  block_5_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  block_5_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  block_6_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  block_6_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  block_6_2_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  block_6_2_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  number_output_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 160, AI_STATIC)
/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  number_output_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 768, AI_STATIC)
/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  block_1_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)
/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  activation_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)
/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  block_1_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)
/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  block_2_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  activation_1_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &activation_1_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  block_2_2_conv2d_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 12), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &block_2_2_conv2d_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  block_2_output_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 12), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &block_2_output_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  block_3_conv2d_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_3_conv2d_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  activation_2_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &activation_2_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  block_3_2_conv2d_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_3_2_conv2d_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  block_4_conv2d_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &block_4_conv2d_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  activation_3_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &activation_3_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  block_4_2_conv2d_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_4_2_conv2d_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  block_4_output_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_4_output_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  block_5_conv2d_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &block_5_conv2d_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  activation_4_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &activation_4_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  block_5_2_conv2d_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &block_5_2_conv2d_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  block_6_conv2d_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &block_6_conv2d_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  activation_5_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &activation_5_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  block_6_2_conv2d_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 2, 1), AI_STRIDE_INIT(4, 4, 4, 128, 256),
  1, &block_6_2_conv2d_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  global_average_pooling2d_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &global_average_pooling2d_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  number_output_dense_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &number_output_dense_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  number_output_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &number_output_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  block_1_conv2d_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 4, 4, 64, 192),
  1, &block_1_conv2d_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  block_1_conv2d_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_1_conv2d_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  block_1_2_conv2d_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &block_1_2_conv2d_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  block_1_2_conv2d_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_1_2_conv2d_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  block_2_conv2d_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &block_2_conv2d_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  block_2_conv2d_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_2_conv2d_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  block_2_2_conv2d_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &block_2_2_conv2d_weights_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  block_2_2_conv2d_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_2_2_conv2d_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  block_3_conv2d_weights, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &block_3_conv2d_weights_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  block_3_conv2d_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_3_conv2d_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  block_3_2_conv2d_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &block_3_2_conv2d_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  block_3_2_conv2d_bias, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_3_2_conv2d_bias_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  block_4_conv2d_weights, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &block_4_conv2d_weights_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  block_4_conv2d_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_4_conv2d_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  block_4_2_conv2d_weights, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &block_4_2_conv2d_weights_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  block_4_2_conv2d_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_4_2_conv2d_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  block_5_conv2d_weights, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &block_5_conv2d_weights_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  block_5_conv2d_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_5_conv2d_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  block_5_2_conv2d_weights, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &block_5_2_conv2d_weights_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  block_5_2_conv2d_bias, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_5_2_conv2d_bias_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  block_6_conv2d_weights, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &block_6_conv2d_weights_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  block_6_conv2d_bias, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_6_conv2d_bias_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  block_6_2_conv2d_weights, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &block_6_2_conv2d_weights_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  block_6_2_conv2d_bias, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_6_2_conv2d_bias_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  number_output_dense_weights, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 32, 5, 1, 1), AI_STRIDE_INIT(4, 4, 128, 640, 640),
  1, &number_output_dense_weights_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  number_output_dense_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &number_output_dense_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 32, 24), AI_STRIDE_INIT(4, 4, 4, 4, 128),
  1, &input_1_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  block_1_conv2d_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 32, 24), AI_STRIDE_INIT(4, 4, 4, 64, 2048),
  1, &block_1_conv2d_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  activation_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 32, 24), AI_STRIDE_INIT(4, 4, 4, 64, 2048),
  1, &activation_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  block_1_2_conv2d_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 12), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &block_1_2_conv2d_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  block_2_conv2d_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_2_conv2d_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  number_output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &number_output_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &number_output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  number_output_layer, 44,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &number_output_chain,
  NULL, &number_output_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  number_output_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &number_output_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &number_output_dense_weights, &number_output_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  number_output_dense_layer, 44,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &number_output_dense_chain,
  NULL, &number_output_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  global_average_pooling2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling2d_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  global_average_pooling2d_layer, 43,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &global_average_pooling2d_chain,
  NULL, &number_output_dense_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_6_2_conv2d_weights, &block_6_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_6_2_conv2d_layer, 40,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_6_2_conv2d_chain,
  NULL, &global_average_pooling2d_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_5_layer, 38,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_5_chain,
  NULL, &block_6_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_6_conv2d_weights, &block_6_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_6_conv2d_layer, 37,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_6_conv2d_chain,
  NULL, &activation_5_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_5_2_conv2d_weights, &block_5_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_5_2_conv2d_layer, 33,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_5_2_conv2d_chain,
  NULL, &block_6_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_4_layer, 31,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_4_chain,
  NULL, &block_5_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_output_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_5_conv2d_weights, &block_5_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_5_conv2d_layer, 30,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_5_conv2d_chain,
  NULL, &activation_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_3_2_conv2d_output, &block_4_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_output_layer, 28,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_4_output_chain,
  NULL, &block_5_conv2d_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_4_2_conv2d_weights, &block_4_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_2_conv2d_layer, 27,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_4_2_conv2d_chain,
  NULL, &block_4_output_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_3_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_3_chain,
  NULL, &block_4_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_4_conv2d_weights, &block_4_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_conv2d_layer, 23,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_4_conv2d_chain,
  NULL, &activation_3_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_3_2_conv2d_weights, &block_3_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_3_2_conv2d_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_3_2_conv2d_chain,
  NULL, &block_4_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_2_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_2_chain,
  NULL, &block_3_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_output_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_3_conv2d_weights, &block_3_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_3_conv2d_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_3_conv2d_chain,
  NULL, &activation_2_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_1_2_conv2d_output, &block_2_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_output_layer, 14,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_2_output_chain,
  NULL, &block_3_conv2d_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_2_2_conv2d_weights, &block_2_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_2_conv2d_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_2_2_conv2d_chain,
  NULL, &block_2_output_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_1_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_1_chain,
  NULL, &block_2_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_2_conv2d_weights, &block_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_conv2d_layer, 9,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_2_conv2d_chain,
  NULL, &activation_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_2_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_2_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_1_2_conv2d_weights, &block_1_2_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_1_2_conv2d_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_1_2_conv2d_chain,
  NULL, &block_2_conv2d_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &activation_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &activation_chain,
  NULL, &block_1_2_conv2d_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_1_conv2d_weights, &block_1_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_1_conv2d_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_1_conv2d_chain,
  NULL, &activation_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 205140, 1, 1),
    205140, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 57536, 1, 1),
    57536, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_OUT_NUM, &number_output_output),
  &block_1_conv2d_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 205140, 1, 1),
      205140, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 57536, 1, 1),
      57536, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_OUT_NUM, &number_output_output),
  &block_1_conv2d_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_1_output_array.data = AI_PTR(g_model_activations_map[0] + 54464);
    input_1_output_array.data_start = AI_PTR(g_model_activations_map[0] + 54464);
    
    block_1_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 2176);
    block_1_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 2176);
    
    activation_output_array.data = AI_PTR(g_model_activations_map[0] + 2176);
    activation_output_array.data_start = AI_PTR(g_model_activations_map[0] + 2176);
    
    block_1_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    block_1_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    block_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    block_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    activation_1_output_array.data = AI_PTR(g_model_activations_map[0] + 45248);
    activation_1_output_array.data_start = AI_PTR(g_model_activations_map[0] + 45248);
    
    block_2_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 27456);
    block_2_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 27456);
    
    block_2_output_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    block_2_output_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    block_3_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    block_3_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    activation_2_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    activation_2_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    block_3_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    block_3_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    block_4_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    block_4_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    activation_3_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    activation_3_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    block_4_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    block_4_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    block_4_output_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    block_4_output_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    block_5_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    block_5_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    activation_4_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    activation_4_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    block_5_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    block_5_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    block_6_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    block_6_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    activation_5_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    activation_5_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    block_6_2_conv2d_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    block_6_2_conv2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    global_average_pooling2d_output_array.data = AI_PTR(g_model_activations_map[0] + 256);
    global_average_pooling2d_output_array.data_start = AI_PTR(g_model_activations_map[0] + 256);
    
    number_output_dense_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    number_output_dense_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    number_output_output_array.data = AI_PTR(g_model_activations_map[0] + 20);
    number_output_output_array.data_start = AI_PTR(g_model_activations_map[0] + 20);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool model_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_model_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    block_1_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_1_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 0);
    block_1_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 0);
    
    block_1_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_1_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 576);
    block_1_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 576);
    
    block_1_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_1_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 640);
    block_1_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 640);
    
    block_1_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_1_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 19072);
    block_1_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 19072);
    
    block_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 19200);
    block_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 19200);
    
    block_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 37632);
    block_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 37632);
    
    block_2_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_2_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 37696);
    block_2_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 37696);
    
    block_2_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_2_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 56128);
    block_2_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 56128);
    
    block_3_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_3_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 56256);
    block_3_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 56256);
    
    block_3_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_3_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 74688);
    block_3_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 74688);
    
    block_3_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_3_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 74752);
    block_3_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 74752);
    
    block_3_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_3_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 93184);
    block_3_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 93184);
    
    block_4_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_4_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 93312);
    block_4_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 93312);
    
    block_4_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_4_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 111744);
    block_4_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 111744);
    
    block_4_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_4_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 111808);
    block_4_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 111808);
    
    block_4_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_4_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 130240);
    block_4_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 130240);
    
    block_5_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_5_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 130368);
    block_5_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 130368);
    
    block_5_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_5_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 148800);
    block_5_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 148800);
    
    block_5_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_5_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 148864);
    block_5_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 148864);
    
    block_5_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_5_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 167296);
    block_5_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 167296);
    
    block_6_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_6_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 167424);
    block_6_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 167424);
    
    block_6_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_6_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 185856);
    block_6_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 185856);
    
    block_6_2_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    block_6_2_conv2d_weights_array.data = AI_PTR(g_model_weights_map[0] + 185920);
    block_6_2_conv2d_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 185920);
    
    block_6_2_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    block_6_2_conv2d_bias_array.data = AI_PTR(g_model_weights_map[0] + 204352);
    block_6_2_conv2d_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 204352);
    
    number_output_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    number_output_dense_weights_array.data = AI_PTR(g_model_weights_map[0] + 204480);
    number_output_dense_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 204480);
    
    number_output_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    number_output_dense_bias_array.data = AI_PTR(g_model_weights_map[0] + 205120);
    number_output_dense_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 205120);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_model_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MODEL_MODEL_NAME,
      .model_signature   = AI_MODEL_MODEL_SIGNATURE,
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
      
      .n_macc            = 4682512,
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
ai_bool ai_model_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MODEL_MODEL_NAME,
      .model_signature   = AI_MODEL_MODEL_SIGNATURE,
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
      
      .n_macc            = 4682512,
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
ai_error ai_model_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_model_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_model_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_model_create(network, AI_MODEL_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_model_data_params_get(&params) != true) {
        err = ai_model_get_error(*network);
        return err;
    }
#if defined(AI_MODEL_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_MODEL_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_model_init(*network, &params) != true) {
        err = ai_model_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_model_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_model_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_model_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_model_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= model_configure_weights(net_ctx, params);
  ok &= model_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_model_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_model_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_MODEL_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

