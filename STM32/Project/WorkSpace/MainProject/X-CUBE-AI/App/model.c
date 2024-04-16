/**
  ******************************************************************************
  * @file    model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Mar 26 00:07:46 2024
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
#define AI_MODEL_MODEL_SIGNATURE     "20192bb928213843838f0019f7f62a16"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue Mar 26 00:07:46 2024"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_MODEL_N_BATCHES
#define AI_MODEL_N_BATCHES         (1)

static ai_ptr g_model_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_model_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1568, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_52_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)
/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_52_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)
/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_64_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)
/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_64_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_70_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3136, AI_STATIC)
/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_76_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98, AI_STATIC)
/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_79_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)
/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_79_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_98_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_98_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)
/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_84_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_84_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_87_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_87_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_90_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)
/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_90_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_93_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)
/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_93_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_102_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)
/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_102_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_108_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3136, AI_STATIC)
/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_114_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 98, AI_STATIC)
/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_117_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)
/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_117_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  gemm_121_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 320, AI_STATIC)
/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  gemm_121_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_10_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 768, AI_STATIC)
/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  nl_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  nl_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  split_6_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  split_6_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  nl_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  nl_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_13_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  nl_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  nl_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_20_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  concat_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)
/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  nl_27_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_28_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  pool_30_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  pool_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  concat_31_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  nl_33_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_34_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  reduce_36_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  reduce_36_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  reduce_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  concat_37_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  nl_39_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_40_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  nl_42_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_43_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  split_44_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  split_44_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_59_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  nl_61_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_62_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  nl_47_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_49_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  nl_50_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_51_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_52_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  nl_53_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  nl_56_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_57_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_58_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  concat_63_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)
/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_64_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  nl_65_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_66_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  pool_68_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  pool_67_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  concat_69_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_70_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  nl_71_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_72_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  reduce_74_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)
/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  reduce_74_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)
/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  reduce_73_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)
/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  concat_75_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)
/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_76_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)
/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  nl_77_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)
/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_78_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)
/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_79_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  nl_80_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_81_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  split_82_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  split_82_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_97_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_98_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  nl_99_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_100_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_83_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_84_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  nl_85_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_86_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_87_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  nl_88_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_89_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_90_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  nl_91_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_92_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)
/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_93_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  nl_94_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_95_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_96_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  concat_101_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_102_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  nl_103_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_104_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  pool_106_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  pool_105_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)
/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  concat_107_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_108_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  nl_109_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_110_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  reduce_112_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)
/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  reduce_112_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)
/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  reduce_111_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)
/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  concat_113_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)
/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_114_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)
/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  nl_115_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)
/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_116_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)
/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_117_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  nl_118_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_119_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)
/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  pool_120_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  gemm_121_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  nl_122_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 5, AI_STATIC)
/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  reduce_112_Placeholder_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  reduce_74_Placeholder_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  reduce_36_Placeholder_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 6, 8), AI_STRIDE_INIT(4, 4, 4, 32, 192),
  1, &conv2d_0_weights_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_0_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 8, 3, 3, 16), AI_STRIDE_INIT(4, 4, 32, 512, 1536),
  1, &conv2d_3_weights_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_weights, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1,8, 8, 8),
  1, &conv2d_21_weights_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_21_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_weights, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &conv2d_22_weights_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_bias, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_22_bias_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1,8, 8, 8),
  1, &conv2d_7_weights_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &conv2d_8_weights_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 4), AI_STRIDE_INIT(4, 4, 64, 256, 256),
  1, &conv2d_11_weights_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_11_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_weights, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 4, 3, 3, 4), AI_STRIDE_INIT(4, 4, 16, 64, 192),
  1, &conv2d_14_weights_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_14_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_weights, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 4, 1, 1, 16), AI_STRIDE_INIT(4, 4, 16, 256, 256),
  1, &conv2d_17_weights_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_bias, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_17_bias_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_weights, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &conv2d_26_weights_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_26_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_weights, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 32, 7, 7, 1), AI_STRIDE_INIT(4, 4, 128, 128, 896),
  1, &conv2d_32_weights_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_bias, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_32_bias_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_weights, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 2, 7, 7, 1), AI_STRIDE_INIT(4, 4, 8, 8, 56),
  1, &conv2d_38_weights_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &conv2d_41_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_41_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1,16, 16, 16),
  1, &conv2d_59_weights_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_59_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_weights, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &conv2d_60_weights_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_60_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1,16, 16, 16),
  1, &conv2d_45_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_bias, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_45_bias_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_weights, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &conv2d_46_weights_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_46_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_weights, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 8), AI_STRIDE_INIT(4, 4, 128, 1024, 1024),
  1, &conv2d_49_weights_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_49_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_52_weights, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 8, 3, 3, 8), AI_STRIDE_INIT(4, 4, 32, 256, 768),
  1, &conv2d_52_weights_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_52_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_52_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_weights, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 32), AI_STRIDE_INIT(4, 4, 32, 1024, 1024),
  1, &conv2d_55_weights_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_bias, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_55_bias_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_64_weights, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 32), AI_STRIDE_INIT(4, 4, 256, 8192, 24576),
  1, &conv2d_64_weights_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_64_bias, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_64_bias_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_70_weights, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 64, 7, 7, 1), AI_STRIDE_INIT(4, 4, 256, 256, 1792),
  1, &conv2d_70_weights_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_76_weights, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 2, 7, 7, 1), AI_STRIDE_INIT(4, 4, 8, 8, 56),
  1, &conv2d_76_weights_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_79_weights, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 12288),
  1, &conv2d_79_weights_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_79_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_79_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_weights, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1,16, 16, 16),
  1, &conv2d_97_weights_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_97_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_98_weights, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &conv2d_98_weights_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_98_bias, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_98_bias_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_weights, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1,16, 16, 16),
  1, &conv2d_83_weights_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_bias, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_83_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_84_weights, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &conv2d_84_weights_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_84_bias, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_84_bias_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_87_weights, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 2048),
  1, &conv2d_87_weights_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_87_bias, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_87_bias_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_90_weights, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 16), AI_STRIDE_INIT(4, 4, 64, 1024, 3072),
  1, &conv2d_90_weights_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_90_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_90_bias_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_93_weights, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &conv2d_93_weights_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_93_bias, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_93_bias_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_102_weights, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 32), AI_STRIDE_INIT(4, 4, 256, 8192, 24576),
  1, &conv2d_102_weights_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_102_bias, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_102_bias_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_108_weights, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 64, 7, 7, 1), AI_STRIDE_INIT(4, 4, 256, 256, 1792),
  1, &conv2d_108_weights_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_114_weights, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 2, 7, 7, 1), AI_STRIDE_INIT(4, 4, 8, 8, 56),
  1, &conv2d_114_weights_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_weights, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 8192),
  1, &conv2d_117_weights_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_117_bias_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  gemm_121_weights, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 64, 5, 1, 1), AI_STRIDE_INIT(4, 4, 256, 1280, 1280),
  1, &gemm_121_weights_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  gemm_121_bias, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &gemm_121_bias_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_10_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 32, 24), AI_STRIDE_INIT(4, 4, 4, 4, 128),
  1, &serving_default_input_10_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &conv2d_0_output_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  nl_1_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &nl_1_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_2_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &eltwise_2_output_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &conv2d_3_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  nl_4_output, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &nl_4_output_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_5_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_5_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  split_6_output0, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &split_6_output0_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  split_6_output1, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &split_6_output1_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_output, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &conv2d_21_output_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &conv2d_22_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  nl_23_output, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &nl_23_output_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_24_output, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_24_output_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 16, 12), AI_STRIDE_INIT(4, 4, 4, 32, 512),
  1, &conv2d_7_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &conv2d_8_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  nl_9_output, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &nl_9_output_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_10_output_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 16, 12), AI_STRIDE_INIT(4, 4, 4, 16, 256),
  1, &conv2d_11_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_output, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 16, 12), AI_STRIDE_INIT(4, 4, 4, 16, 256),
  1, &nl_12_output_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_13_output, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 16, 12), AI_STRIDE_INIT(4, 4, 4, 16, 256),
  1, &eltwise_13_output_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 16, 12), AI_STRIDE_INIT(4, 4, 4, 16, 256),
  1, &conv2d_14_output_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  nl_15_output, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 16, 12), AI_STRIDE_INIT(4, 4, 4, 16, 256),
  1, &nl_15_output_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_16_output, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 16, 12), AI_STRIDE_INIT(4, 4, 4, 16, 256),
  1, &eltwise_16_output_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_output, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &conv2d_17_output_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  nl_18_output, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &nl_18_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_19_output, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_19_output_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_20_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_20_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  concat_25_output, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 12), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &concat_25_output_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &conv2d_26_output_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  nl_27_output, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &nl_27_output_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_28_output, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_28_output_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  pool_30_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &pool_30_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  pool_29_output, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &pool_29_output_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  concat_31_output, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &concat_31_output_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_output, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_32_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  nl_33_output, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_33_output_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_34_output, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_34_output_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  reduce_36_output, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 16, 12), AI_STRIDE_INIT(4, 4, 4, 4, 64),
  1, &reduce_36_output_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  reduce_36_Mul_output, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 16, 12), AI_STRIDE_INIT(4, 4, 4, 4, 64),
  1, &reduce_36_Mul_output_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  reduce_35_output, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 16, 12), AI_STRIDE_INIT(4, 4, 4, 4, 64),
  1, &reduce_35_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  concat_37_output, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 16, 12), AI_STRIDE_INIT(4, 4, 4, 8, 128),
  1, &concat_37_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_output, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 16, 12), AI_STRIDE_INIT(4, 4, 4, 4, 64),
  1, &conv2d_38_output_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  nl_39_output, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 16, 12), AI_STRIDE_INIT(4, 4, 4, 4, 64),
  1, &nl_39_output_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_40_output, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 12), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &eltwise_40_output_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_output, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &conv2d_41_output_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  nl_42_output, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &nl_42_output_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_43_output, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_43_output_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  split_44_output0, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &split_44_output0_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  split_44_output1, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &split_44_output1_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_59_output, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &conv2d_59_output_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_output, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &conv2d_60_output_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  nl_61_output, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &nl_61_output_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_62_output, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_62_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_output, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 6), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &conv2d_45_output_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_output, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &conv2d_46_output_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  nl_47_output, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &nl_47_output_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_48_output, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_48_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_49_output, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 6), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_49_output_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  nl_50_output, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 6), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &nl_50_output_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_51_output, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 6), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &eltwise_51_output_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_52_output, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 6), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_52_output_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  nl_53_output, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 6), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &nl_53_output_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 6), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &eltwise_54_output_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_output, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &conv2d_55_output_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  nl_56_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &nl_56_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_57_output, AI_STATIC,
  132, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_57_output_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_58_output, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_58_output_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  concat_63_output, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 8, 6), AI_STRIDE_INIT(4, 4, 4, 256, 2048),
  1, &concat_63_output_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_64_output, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &conv2d_64_output_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  nl_65_output, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &nl_65_output_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_66_output, AI_STATIC,
  137, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_66_output_array, NULL)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  pool_68_output, AI_STATIC,
  138, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &pool_68_output_array, NULL)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  pool_67_output, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &pool_67_output_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  concat_69_output, AI_STATIC,
  140, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &concat_69_output_array, NULL)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_70_output, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_70_output_array, NULL)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  nl_71_output, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_71_output_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_72_output, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_72_output_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  reduce_74_output, AI_STATIC,
  144, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 8, 6), AI_STRIDE_INIT(4, 4, 4, 4, 32),
  1, &reduce_74_output_array, NULL)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  reduce_74_Mul_output, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 8, 6), AI_STRIDE_INIT(4, 4, 4, 4, 32),
  1, &reduce_74_Mul_output_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  reduce_73_output, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 8, 6), AI_STRIDE_INIT(4, 4, 4, 4, 32),
  1, &reduce_73_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  concat_75_output, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 8, 6), AI_STRIDE_INIT(4, 4, 4, 8, 64),
  1, &concat_75_output_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_76_output, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 8, 6), AI_STRIDE_INIT(4, 4, 4, 4, 32),
  1, &conv2d_76_output_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  nl_77_output, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 8, 6), AI_STRIDE_INIT(4, 4, 4, 4, 32),
  1, &nl_77_output_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_78_output, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 6), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &eltwise_78_output_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_79_output, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &conv2d_79_output_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  nl_80_output, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &nl_80_output_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_81_output, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_81_output_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  split_82_output0, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &split_82_output0_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  split_82_output1, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &split_82_output1_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_97_output, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &conv2d_97_output_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_98_output, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &conv2d_98_output_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  nl_99_output, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &nl_99_output_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_100_output, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_100_output_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_83_output, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &conv2d_83_output_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_84_output, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &conv2d_84_output_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  nl_85_output, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &nl_85_output_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_86_output, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_86_output_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_87_output, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &conv2d_87_output_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  nl_88_output, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &nl_88_output_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_89_output, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &eltwise_89_output_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_90_output, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &conv2d_90_output_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  nl_91_output, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &nl_91_output_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_92_output, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 4, 3), AI_STRIDE_INIT(4, 4, 4, 64, 256),
  1, &eltwise_92_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_93_output, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &conv2d_93_output_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  nl_94_output, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &nl_94_output_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_95_output, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_95_output_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_96_output, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_96_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  concat_101_output, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 3), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &concat_101_output_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_102_output, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &conv2d_102_output_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  nl_103_output, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &nl_103_output_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_104_output, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_104_output_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  pool_106_output, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &pool_106_output_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  pool_105_output, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &pool_105_output_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  concat_107_output, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &concat_107_output_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_108_output, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_108_output_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  nl_109_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &nl_109_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_110_output, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_110_output_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  reduce_112_output, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 3), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &reduce_112_output_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  reduce_112_Mul_output, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 3), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &reduce_112_Mul_output_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  reduce_111_output, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 3), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &reduce_111_output_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  concat_113_output, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 4, 3), AI_STRIDE_INIT(4, 4, 4, 8, 32),
  1, &concat_113_output_array, NULL)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_114_output, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 3), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &conv2d_114_output_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  nl_115_output, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 3), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &nl_115_output_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_116_output, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 3), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &eltwise_116_output_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_117_output, AI_STATIC,
  191, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 3), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &conv2d_117_output_array, NULL)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  nl_118_output, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 3), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &nl_118_output_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_119_output, AI_STATIC,
  193, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 3), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &eltwise_119_output_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  pool_120_output, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &pool_120_output_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  gemm_121_output, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &gemm_121_output_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  nl_122_output, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &nl_122_output_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  reduce_112_Placeholder, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_112_Placeholder_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  reduce_74_Placeholder, AI_STATIC,
  198, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_74_Placeholder_array, NULL)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  reduce_36_Placeholder, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &reduce_36_Placeholder_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_122_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_121_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_122_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_122_layer, 122,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &nl_122_chain,
  NULL, &nl_122_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_121_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_120_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_121_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_121_weights, &gemm_121_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_121_layer, 121,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_121_chain,
  NULL, &nl_122_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_120_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_119_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_120_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_120_layer, 120,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_120_chain,
  NULL, &gemm_121_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(4, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_119_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_117_output, &nl_118_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_119_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_119_layer, 119,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_119_chain,
  NULL, &pool_120_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_118_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_118_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_118_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_118_chain,
  NULL, &eltwise_119_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_117_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_116_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_117_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_117_weights, &conv2d_117_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_117_layer, 117,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_117_chain,
  NULL, &nl_118_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_116_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_115_output, &eltwise_110_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_116_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_116_layer, 116,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_116_chain,
  NULL, &conv2d_117_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_115_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_115_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_115_layer, 115,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_115_chain,
  NULL, &eltwise_116_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_113_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_114_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_114_layer, 114,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_114_chain,
  NULL, &nl_115_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 3, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_113_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_112_Mul_output, &reduce_111_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_113_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_113_layer, 113,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_113_chain,
  NULL, &conv2d_114_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float reduce_111_neutral_value_data[] = { -AI_FLT_MAX };
AI_ARRAY_OBJ_DECLARE(
    reduce_111_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_111_neutral_value_data, reduce_111_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_111_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_110_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_111_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_111_layer, 111,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_111_chain,
  NULL, &concat_113_layer, AI_STATIC, 
  .operation = ai_max, 
  .neutral_value = &reduce_111_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_112_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_112_output, &reduce_112_Placeholder),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_112_Mul_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_112_Mul_layer, 112,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &reduce_112_Mul_chain,
  NULL, &reduce_111_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)


AI_STATIC_CONST ai_float reduce_112_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_112_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_112_neutral_value_data, reduce_112_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_110_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_112_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_112_layer, 112,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_112_chain,
  NULL, &reduce_112_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_112_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_110_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_104_output, &nl_109_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_110_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_110_layer, 110,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_110_chain,
  NULL, &reduce_112_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_109_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_108_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_109_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_109_layer, 109,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_109_chain,
  NULL, &eltwise_110_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_108_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_107_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_108_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_108_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_108_layer, 108,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_108_chain,
  NULL, &nl_109_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 3, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_107_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &pool_106_output, &pool_105_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_107_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_107_layer, 107,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_107_chain,
  NULL, &conv2d_108_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_105_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_104_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_105_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_105_layer, 105,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_105_chain,
  NULL, &concat_107_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(4, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_106_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_104_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_106_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_106_layer, 106,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_106_chain,
  NULL, &pool_105_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(4, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 3), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_104_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_102_output, &nl_103_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_104_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_104_layer, 104,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_104_chain,
  NULL, &pool_106_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_103_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_102_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_103_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_103_layer, 103,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_103_chain,
  NULL, &eltwise_104_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_102_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_101_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_102_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_102_weights, &conv2d_102_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_102_layer, 102,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_102_chain,
  NULL, &nl_103_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_101_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_96_output, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_101_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_101_layer, 101,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_101_chain,
  NULL, &conv2d_102_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_96_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_95_output, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_96_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_96_layer, 96,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_96_chain,
  NULL, &concat_101_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_95_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_93_output, &nl_94_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_95_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_95_layer, 95,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_95_chain,
  NULL, &eltwise_96_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_94_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_94_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_94_layer, 94,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_94_chain,
  NULL, &eltwise_95_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_93_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_92_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_93_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_93_weights, &conv2d_93_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_93_layer, 93,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_93_chain,
  NULL, &nl_94_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_92_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_90_output, &nl_91_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_92_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_92_layer, 92,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_92_chain,
  NULL, &conv2d_93_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_91_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_90_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_91_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_91_layer, 91,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_91_chain,
  NULL, &eltwise_92_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_90_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_90_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_90_weights, &conv2d_90_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_90_layer, 90,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_90_chain,
  NULL, &nl_91_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_89_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_87_output, &nl_88_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_89_layer, 89,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_89_chain,
  NULL, &conv2d_90_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_88_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_88_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_88_layer, 88,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_88_chain,
  NULL, &eltwise_89_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_87_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_87_weights, &conv2d_87_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_87_layer, 87,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_87_chain,
  NULL, &nl_88_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_86_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_84_output, &nl_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_86_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_86_layer, 86,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_86_chain,
  NULL, &conv2d_87_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_85_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_85_layer, 85,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_85_chain,
  NULL, &eltwise_86_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_84_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_84_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_84_weights, &conv2d_84_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_84_layer, 84,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_84_chain,
  NULL, &nl_85_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_82_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_83_weights, &conv2d_83_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_83_layer, 83,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_83_chain,
  NULL, &conv2d_84_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_100_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_98_output, &nl_99_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_100_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_100_layer, 100,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_100_chain,
  NULL, &conv2d_83_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_99_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_98_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_99_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_99_layer, 99,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_99_chain,
  NULL, &eltwise_100_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_98_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_97_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_98_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_98_weights, &conv2d_98_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_98_layer, 98,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_98_chain,
  NULL, &nl_99_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_97_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_82_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_97_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_97_weights, &conv2d_97_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_97_layer, 97,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_97_chain,
  NULL, &conv2d_98_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  split_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_82_output0, &split_82_output1),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  split_82_layer, 82,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &split_82_chain,
  NULL, &conv2d_97_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_79_output, &nl_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_81_layer, 81,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_81_chain,
  NULL, &split_82_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_80_layer, 80,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_80_chain,
  NULL, &eltwise_81_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_79_weights, &conv2d_79_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_79_layer, 79,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_79_chain,
  NULL, &nl_80_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_77_output, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_78_layer, 78,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_78_chain,
  NULL, &conv2d_79_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_77_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_77_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_77_layer, 77,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_77_chain,
  NULL, &eltwise_78_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_76_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_76_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_76_layer, 76,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_76_chain,
  NULL, &nl_77_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 3, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_74_Mul_output, &reduce_73_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_75_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_75_layer, 75,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_75_chain,
  NULL, &conv2d_76_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float reduce_73_neutral_value_data[] = { -AI_FLT_MAX };
AI_ARRAY_OBJ_DECLARE(
    reduce_73_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_73_neutral_value_data, reduce_73_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_73_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_73_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_73_layer, 73,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_73_chain,
  NULL, &concat_75_layer, AI_STATIC, 
  .operation = ai_max, 
  .neutral_value = &reduce_73_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_74_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_74_output, &reduce_74_Placeholder),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_74_Mul_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_74_Mul_layer, 74,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &reduce_74_Mul_chain,
  NULL, &reduce_73_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)


AI_STATIC_CONST ai_float reduce_74_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_74_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_74_neutral_value_data, reduce_74_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_74_layer, 74,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_74_chain,
  NULL, &reduce_74_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_74_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_72_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_66_output, &nl_71_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_72_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_72_layer, 72,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_72_chain,
  NULL, &reduce_74_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_71_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_71_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_71_layer, 71,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_71_chain,
  NULL, &eltwise_72_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_70_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_70_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_70_layer, 70,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_70_chain,
  NULL, &nl_71_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 3, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &pool_68_output, &pool_67_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_69_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_69_layer, 69,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_69_chain,
  NULL, &conv2d_70_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_67_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_66_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_67_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_67_layer, 67,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_67_chain,
  NULL, &concat_69_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(8, 6), 
  .pool_stride = AI_SHAPE_2D_INIT(8, 6), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_68_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_66_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_68_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_68_layer, 68,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_68_chain,
  NULL, &pool_67_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(8, 6), 
  .pool_stride = AI_SHAPE_2D_INIT(8, 6), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_64_output, &nl_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_66_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_66_layer, 66,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_66_chain,
  NULL, &pool_68_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_64_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_65_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_65_layer, 65,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_65_chain,
  NULL, &eltwise_66_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_64_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_64_weights, &conv2d_64_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_64_layer, 64,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_64_chain,
  NULL, &nl_65_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_58_output, &eltwise_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_63_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_63_layer, 63,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_63_chain,
  NULL, &conv2d_64_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_57_output, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_58_layer, 58,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_58_chain,
  NULL, &concat_63_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_55_output, &nl_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_57_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_57_layer, 57,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_57_chain,
  NULL, &eltwise_58_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_56_layer, 56,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_56_chain,
  NULL, &eltwise_57_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_55_weights, &conv2d_55_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_55_layer, 55,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_55_chain,
  NULL, &nl_56_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_52_output, &nl_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_54_chain,
  NULL, &conv2d_55_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_53_layer, 53,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_53_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_52_weights, &conv2d_52_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_52_layer, 52,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_52_chain,
  NULL, &nl_53_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_49_output, &nl_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_51_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_51_layer, 51,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_51_chain,
  NULL, &conv2d_52_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_50_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_50_layer, 50,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_50_chain,
  NULL, &eltwise_51_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_49_weights, &conv2d_49_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_49_layer, 49,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_49_chain,
  NULL, &nl_50_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_46_output, &nl_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_48_layer, 48,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_48_chain,
  NULL, &conv2d_49_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_47_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_47_chain,
  NULL, &eltwise_48_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_46_weights, &conv2d_46_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_46_layer, 46,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_46_chain,
  NULL, &nl_47_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_44_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_45_weights, &conv2d_45_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_45_layer, 45,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_45_chain,
  NULL, &conv2d_46_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_60_output, &nl_61_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_62_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_62_layer, 62,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_62_chain,
  NULL, &conv2d_45_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_61_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_61_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_61_layer, 61,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_61_chain,
  NULL, &eltwise_62_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_59_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_60_weights, &conv2d_60_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_60_layer, 60,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_60_chain,
  NULL, &nl_61_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_44_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_59_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_59_weights, &conv2d_59_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_59_layer, 59,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_59_chain,
  NULL, &conv2d_60_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  split_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_44_output0, &split_44_output1),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  split_44_layer, 44,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &split_44_chain,
  NULL, &conv2d_59_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_41_output, &nl_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_43_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_43_layer, 43,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_43_chain,
  NULL, &split_44_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_42_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_42_layer, 42,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_42_chain,
  NULL, &eltwise_43_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_41_weights, &conv2d_41_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_41_layer, 41,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_41_chain,
  NULL, &nl_42_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_39_output, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_40_layer, 40,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_40_chain,
  NULL, &conv2d_41_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_39_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_39_layer, 39,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_39_chain,
  NULL, &eltwise_40_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_38_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_38_layer, 38,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_38_chain,
  NULL, &nl_39_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 3, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_36_Mul_output, &reduce_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_37_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_37_layer, 37,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_37_chain,
  NULL, &conv2d_38_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_float reduce_35_neutral_value_data[] = { -AI_FLT_MAX };
AI_ARRAY_OBJ_DECLARE(
    reduce_35_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_35_neutral_value_data, reduce_35_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_35_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_35_layer, 35,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_35_chain,
  NULL, &concat_37_layer, AI_STATIC, 
  .operation = ai_max, 
  .neutral_value = &reduce_35_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_36_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &reduce_36_output, &reduce_36_Placeholder),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_36_Mul_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_36_Mul_layer, 36,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &reduce_36_Mul_chain,
  NULL, &reduce_35_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)


AI_STATIC_CONST ai_float reduce_36_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    reduce_36_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    reduce_36_neutral_value_data, reduce_36_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  reduce_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &reduce_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reduce_36_layer, 36,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &reduce_36_chain,
  NULL, &reduce_36_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &reduce_36_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_28_output, &nl_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_34_layer, 34,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_34_chain,
  NULL, &reduce_36_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_33_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_33_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_33_chain,
  NULL, &eltwise_34_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_32_weights, &conv2d_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_32_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_32_chain,
  NULL, &nl_33_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 3, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &pool_30_output, &pool_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_31_layer, 31,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_31_chain,
  NULL, &conv2d_32_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_29_layer, 29,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &pool_29_chain,
  NULL, &concat_31_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(16, 12), 
  .pool_stride = AI_SHAPE_2D_INIT(16, 12), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_30_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_30_layer, 30,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &pool_30_chain,
  NULL, &pool_29_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(16, 12), 
  .pool_stride = AI_SHAPE_2D_INIT(16, 12), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_26_output, &nl_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_28_layer, 28,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_28_chain,
  NULL, &pool_30_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_27_layer, 27,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_27_chain,
  NULL, &eltwise_28_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_26_weights, &conv2d_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_26_layer, 26,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_26_chain,
  NULL, &nl_27_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_20_output, &eltwise_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_25_layer, 25,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_25_chain,
  NULL, &conv2d_26_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_19_output, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_20_layer, 20,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_20_chain,
  NULL, &concat_25_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_17_output, &nl_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_19_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_19_layer, 19,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_19_chain,
  NULL, &eltwise_20_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_18_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_18_chain,
  NULL, &eltwise_19_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_17_weights, &conv2d_17_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_17_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_17_chain,
  NULL, &nl_18_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_14_output, &nl_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_16_layer, 16,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_16_chain,
  NULL, &conv2d_17_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_15_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_15_chain,
  NULL, &eltwise_16_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_14_weights, &conv2d_14_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_14_chain,
  NULL, &nl_15_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_11_output, &nl_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_13_layer, 13,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_13_chain,
  NULL, &conv2d_14_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_12_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_12_chain,
  NULL, &eltwise_13_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_11_chain,
  NULL, &nl_12_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_8_output, &nl_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_layer, 10,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_10_chain,
  NULL, &conv2d_11_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_9_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_9_chain,
  NULL, &eltwise_10_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_8_chain,
  NULL, &nl_9_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_6_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_7_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .groups = 8, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_22_output, &nl_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_24_layer, 24,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_24_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_23_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_23_chain,
  NULL, &eltwise_24_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_22_weights, &conv2d_22_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_22_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_22_chain,
  NULL, &nl_23_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_6_output1),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_21_weights, &conv2d_21_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_21_layer, 21,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_21_chain,
  NULL, &conv2d_22_layer, AI_STATIC, 
  .groups = 8, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  split_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_6_output0, &split_6_output1),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  split_6_layer, 6,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &split_6_chain,
  NULL, &conv2d_21_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_3_output, &nl_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_5_layer, 5,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_5_chain,
  NULL, &split_6_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_4_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_4_chain,
  NULL, &eltwise_5_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_3_chain,
  NULL, &nl_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_0_output, &nl_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_2_layer, 2,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_2_chain,
  NULL, &conv2d_3_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_1_layer, 1,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_1_chain,
  NULL, &eltwise_2_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 0,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_0_chain,
  NULL, &nl_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 2, 3, 3), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 302268, 1, 1),
    302268, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 49152, 1, 1),
    49152, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_OUT_NUM, &nl_122_output),
  &conv2d_0_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 302268, 1, 1),
      302268, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 49152, 1, 1),
      49152, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_OUT_NUM, &nl_122_output),
  &conv2d_0_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input_10_output_array.data = AI_PTR(g_model_activations_map[0] + 27328);
    serving_default_input_10_output_array.data_start = AI_PTR(g_model_activations_map[0] + 27328);
    
    conv2d_0_output_array.data = AI_PTR(g_model_activations_map[0] + 21184);
    conv2d_0_output_array.data_start = AI_PTR(g_model_activations_map[0] + 21184);
    
    nl_1_output_array.data = AI_PTR(g_model_activations_map[0] + 15040);
    nl_1_output_array.data_start = AI_PTR(g_model_activations_map[0] + 15040);
    
    eltwise_2_output_array.data = AI_PTR(g_model_activations_map[0] + 21184);
    eltwise_2_output_array.data_start = AI_PTR(g_model_activations_map[0] + 21184);
    
    conv2d_3_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    conv2d_3_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    nl_4_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    nl_4_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    eltwise_5_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    eltwise_5_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    split_6_output0_array.data = AI_PTR(g_model_activations_map[0] + 18432);
    split_6_output0_array.data_start = AI_PTR(g_model_activations_map[0] + 18432);
    
    split_6_output1_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    split_6_output1_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    conv2d_21_output_array.data = AI_PTR(g_model_activations_map[0] + 30720);
    conv2d_21_output_array.data_start = AI_PTR(g_model_activations_map[0] + 30720);
    
    conv2d_22_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    conv2d_22_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    nl_23_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    nl_23_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    eltwise_24_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    eltwise_24_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    conv2d_7_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    conv2d_7_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    conv2d_8_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_8_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_9_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    nl_9_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    eltwise_10_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    eltwise_10_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    conv2d_11_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    conv2d_11_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    nl_12_output_array.data = AI_PTR(g_model_activations_map[0] + 15360);
    nl_12_output_array.data_start = AI_PTR(g_model_activations_map[0] + 15360);
    
    eltwise_13_output_array.data = AI_PTR(g_model_activations_map[0] + 15360);
    eltwise_13_output_array.data_start = AI_PTR(g_model_activations_map[0] + 15360);
    
    conv2d_14_output_array.data = AI_PTR(g_model_activations_map[0] + 14800);
    conv2d_14_output_array.data_start = AI_PTR(g_model_activations_map[0] + 14800);
    
    nl_15_output_array.data = AI_PTR(g_model_activations_map[0] + 36864);
    nl_15_output_array.data_start = AI_PTR(g_model_activations_map[0] + 36864);
    
    eltwise_16_output_array.data = AI_PTR(g_model_activations_map[0] + 39936);
    eltwise_16_output_array.data_start = AI_PTR(g_model_activations_map[0] + 39936);
    
    conv2d_17_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    conv2d_17_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    nl_18_output_array.data = AI_PTR(g_model_activations_map[0] + 36864);
    nl_18_output_array.data_start = AI_PTR(g_model_activations_map[0] + 36864);
    
    eltwise_19_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    eltwise_19_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    eltwise_20_output_array.data = AI_PTR(g_model_activations_map[0] + 36864);
    eltwise_20_output_array.data_start = AI_PTR(g_model_activations_map[0] + 36864);
    
    concat_25_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    concat_25_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    conv2d_26_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    conv2d_26_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    nl_27_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    nl_27_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    eltwise_28_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    eltwise_28_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    pool_30_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    pool_30_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    pool_29_output_array.data = AI_PTR(g_model_activations_map[0] + 64);
    pool_29_output_array.data_start = AI_PTR(g_model_activations_map[0] + 64);
    
    concat_31_output_array.data = AI_PTR(g_model_activations_map[0] + 128);
    concat_31_output_array.data_start = AI_PTR(g_model_activations_map[0] + 128);
    
    conv2d_32_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_32_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_33_output_array.data = AI_PTR(g_model_activations_map[0] + 4);
    nl_33_output_array.data_start = AI_PTR(g_model_activations_map[0] + 4);
    
    eltwise_34_output_array.data = AI_PTR(g_model_activations_map[0] + 24576);
    eltwise_34_output_array.data_start = AI_PTR(g_model_activations_map[0] + 24576);
    
    reduce_36_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    reduce_36_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    reduce_36_Mul_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    reduce_36_Mul_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    reduce_35_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    reduce_35_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    concat_37_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    concat_37_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    conv2d_38_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_38_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_39_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    nl_39_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    eltwise_40_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    eltwise_40_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    conv2d_41_output_array.data = AI_PTR(g_model_activations_map[0] + 13824);
    conv2d_41_output_array.data_start = AI_PTR(g_model_activations_map[0] + 13824);
    
    nl_42_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    nl_42_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    eltwise_43_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    eltwise_43_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    split_44_output0_array.data = AI_PTR(g_model_activations_map[0] + 0);
    split_44_output0_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    split_44_output1_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    split_44_output1_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    conv2d_59_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    conv2d_59_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    conv2d_60_output_array.data = AI_PTR(g_model_activations_map[0] + 9216);
    conv2d_60_output_array.data_start = AI_PTR(g_model_activations_map[0] + 9216);
    
    nl_61_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    nl_61_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    eltwise_62_output_array.data = AI_PTR(g_model_activations_map[0] + 15360);
    eltwise_62_output_array.data_start = AI_PTR(g_model_activations_map[0] + 15360);
    
    conv2d_45_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    conv2d_45_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    conv2d_46_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    conv2d_46_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    nl_47_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    nl_47_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    eltwise_48_output_array.data = AI_PTR(g_model_activations_map[0] + 21504);
    eltwise_48_output_array.data_start = AI_PTR(g_model_activations_map[0] + 21504);
    
    conv2d_49_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_49_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_50_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    nl_50_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    eltwise_51_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    eltwise_51_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    conv2d_52_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_52_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_53_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    nl_53_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    eltwise_54_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    eltwise_54_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    conv2d_55_output_array.data = AI_PTR(g_model_activations_map[0] + 4608);
    conv2d_55_output_array.data_start = AI_PTR(g_model_activations_map[0] + 4608);
    
    nl_56_output_array.data = AI_PTR(g_model_activations_map[0] + 27648);
    nl_56_output_array.data_start = AI_PTR(g_model_activations_map[0] + 27648);
    
    eltwise_57_output_array.data = AI_PTR(g_model_activations_map[0] + 33792);
    eltwise_57_output_array.data_start = AI_PTR(g_model_activations_map[0] + 33792);
    
    eltwise_58_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    eltwise_58_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    concat_63_output_array.data = AI_PTR(g_model_activations_map[0] + 21504);
    concat_63_output_array.data_start = AI_PTR(g_model_activations_map[0] + 21504);
    
    conv2d_64_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_64_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_65_output_array.data = AI_PTR(g_model_activations_map[0] + 6144);
    nl_65_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6144);
    
    eltwise_66_output_array.data = AI_PTR(g_model_activations_map[0] + 12288);
    eltwise_66_output_array.data_start = AI_PTR(g_model_activations_map[0] + 12288);
    
    pool_68_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    pool_68_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    pool_67_output_array.data = AI_PTR(g_model_activations_map[0] + 128);
    pool_67_output_array.data_start = AI_PTR(g_model_activations_map[0] + 128);
    
    concat_69_output_array.data = AI_PTR(g_model_activations_map[0] + 256);
    concat_69_output_array.data_start = AI_PTR(g_model_activations_map[0] + 256);
    
    conv2d_70_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_70_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_71_output_array.data = AI_PTR(g_model_activations_map[0] + 4);
    nl_71_output_array.data_start = AI_PTR(g_model_activations_map[0] + 4);
    
    eltwise_72_output_array.data = AI_PTR(g_model_activations_map[0] + 8);
    eltwise_72_output_array.data_start = AI_PTR(g_model_activations_map[0] + 8);
    
    reduce_74_output_array.data = AI_PTR(g_model_activations_map[0] + 6152);
    reduce_74_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6152);
    
    reduce_74_Mul_output_array.data = AI_PTR(g_model_activations_map[0] + 6344);
    reduce_74_Mul_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6344);
    
    reduce_73_output_array.data = AI_PTR(g_model_activations_map[0] + 6152);
    reduce_73_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6152);
    
    concat_75_output_array.data = AI_PTR(g_model_activations_map[0] + 6536);
    concat_75_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6536);
    
    conv2d_76_output_array.data = AI_PTR(g_model_activations_map[0] + 6152);
    conv2d_76_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6152);
    
    nl_77_output_array.data = AI_PTR(g_model_activations_map[0] + 6344);
    nl_77_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6344);
    
    eltwise_78_output_array.data = AI_PTR(g_model_activations_map[0] + 6536);
    eltwise_78_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6536);
    
    conv2d_79_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_79_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_80_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    nl_80_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    eltwise_81_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    eltwise_81_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    split_82_output0_array.data = AI_PTR(g_model_activations_map[0] + 0);
    split_82_output0_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    split_82_output1_array.data = AI_PTR(g_model_activations_map[0] + 768);
    split_82_output1_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    conv2d_97_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    conv2d_97_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    conv2d_98_output_array.data = AI_PTR(g_model_activations_map[0] + 2304);
    conv2d_98_output_array.data_start = AI_PTR(g_model_activations_map[0] + 2304);
    
    nl_99_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    nl_99_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    eltwise_100_output_array.data = AI_PTR(g_model_activations_map[0] + 3840);
    eltwise_100_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3840);
    
    conv2d_83_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    conv2d_83_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    conv2d_84_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    conv2d_84_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    nl_85_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    nl_85_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    eltwise_86_output_array.data = AI_PTR(g_model_activations_map[0] + 5376);
    eltwise_86_output_array.data_start = AI_PTR(g_model_activations_map[0] + 5376);
    
    conv2d_87_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_87_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_88_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    nl_88_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    eltwise_89_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    eltwise_89_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    conv2d_90_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_90_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_91_output_array.data = AI_PTR(g_model_activations_map[0] + 768);
    nl_91_output_array.data_start = AI_PTR(g_model_activations_map[0] + 768);
    
    eltwise_92_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    eltwise_92_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    conv2d_93_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_93_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_94_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    nl_94_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    eltwise_95_output_array.data = AI_PTR(g_model_activations_map[0] + 6912);
    eltwise_95_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6912);
    
    eltwise_96_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    eltwise_96_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    concat_101_output_array.data = AI_PTR(g_model_activations_map[0] + 5376);
    concat_101_output_array.data_start = AI_PTR(g_model_activations_map[0] + 5376);
    
    conv2d_102_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_102_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_103_output_array.data = AI_PTR(g_model_activations_map[0] + 1536);
    nl_103_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1536);
    
    eltwise_104_output_array.data = AI_PTR(g_model_activations_map[0] + 3072);
    eltwise_104_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3072);
    
    pool_106_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    pool_106_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    pool_105_output_array.data = AI_PTR(g_model_activations_map[0] + 128);
    pool_105_output_array.data_start = AI_PTR(g_model_activations_map[0] + 128);
    
    concat_107_output_array.data = AI_PTR(g_model_activations_map[0] + 256);
    concat_107_output_array.data_start = AI_PTR(g_model_activations_map[0] + 256);
    
    conv2d_108_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    conv2d_108_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    nl_109_output_array.data = AI_PTR(g_model_activations_map[0] + 4);
    nl_109_output_array.data_start = AI_PTR(g_model_activations_map[0] + 4);
    
    eltwise_110_output_array.data = AI_PTR(g_model_activations_map[0] + 8);
    eltwise_110_output_array.data_start = AI_PTR(g_model_activations_map[0] + 8);
    
    reduce_112_output_array.data = AI_PTR(g_model_activations_map[0] + 1544);
    reduce_112_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1544);
    
    reduce_112_Mul_output_array.data = AI_PTR(g_model_activations_map[0] + 1592);
    reduce_112_Mul_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1592);
    
    reduce_111_output_array.data = AI_PTR(g_model_activations_map[0] + 1544);
    reduce_111_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1544);
    
    concat_113_output_array.data = AI_PTR(g_model_activations_map[0] + 1640);
    concat_113_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1640);
    
    conv2d_114_output_array.data = AI_PTR(g_model_activations_map[0] + 1544);
    conv2d_114_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1544);
    
    nl_115_output_array.data = AI_PTR(g_model_activations_map[0] + 1592);
    nl_115_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1592);
    
    eltwise_116_output_array.data = AI_PTR(g_model_activations_map[0] + 1640);
    eltwise_116_output_array.data_start = AI_PTR(g_model_activations_map[0] + 1640);
    
    conv2d_117_output_array.data = AI_PTR(g_model_activations_map[0] + 3176);
    conv2d_117_output_array.data_start = AI_PTR(g_model_activations_map[0] + 3176);
    
    nl_118_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    nl_118_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    eltwise_119_output_array.data = AI_PTR(g_model_activations_map[0] + 6248);
    eltwise_119_output_array.data_start = AI_PTR(g_model_activations_map[0] + 6248);
    
    pool_120_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    pool_120_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
    gemm_121_output_array.data = AI_PTR(g_model_activations_map[0] + 256);
    gemm_121_output_array.data_start = AI_PTR(g_model_activations_map[0] + 256);
    
    nl_122_output_array.data = AI_PTR(g_model_activations_map[0] + 0);
    nl_122_output_array.data_start = AI_PTR(g_model_activations_map[0] + 0);
    
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
    
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(g_model_weights_map[0] + 0);
    conv2d_0_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 0);
    
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(g_model_weights_map[0] + 1152);
    conv2d_0_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 1152);
    
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_model_weights_map[0] + 1184);
    conv2d_3_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 1184);
    
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_model_weights_map[0] + 5792);
    conv2d_3_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 5792);
    
    conv2d_21_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_weights_array.data = AI_PTR(g_model_weights_map[0] + 5856);
    conv2d_21_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 5856);
    
    conv2d_21_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_bias_array.data = AI_PTR(g_model_weights_map[0] + 6144);
    conv2d_21_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 6144);
    
    conv2d_22_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_weights_array.data = AI_PTR(g_model_weights_map[0] + 6176);
    conv2d_22_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 6176);
    
    conv2d_22_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_bias_array.data = AI_PTR(g_model_weights_map[0] + 6688);
    conv2d_22_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 6688);
    
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_model_weights_map[0] + 6752);
    conv2d_7_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 6752);
    
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_model_weights_map[0] + 7040);
    conv2d_7_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 7040);
    
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_model_weights_map[0] + 7072);
    conv2d_8_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 7072);
    
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_model_weights_map[0] + 7584);
    conv2d_8_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 7584);
    
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(g_model_weights_map[0] + 7648);
    conv2d_11_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 7648);
    
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(g_model_weights_map[0] + 7904);
    conv2d_11_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 7904);
    
    conv2d_14_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_weights_array.data = AI_PTR(g_model_weights_map[0] + 7920);
    conv2d_14_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 7920);
    
    conv2d_14_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_bias_array.data = AI_PTR(g_model_weights_map[0] + 8496);
    conv2d_14_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 8496);
    
    conv2d_17_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_weights_array.data = AI_PTR(g_model_weights_map[0] + 8512);
    conv2d_17_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 8512);
    
    conv2d_17_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_bias_array.data = AI_PTR(g_model_weights_map[0] + 8768);
    conv2d_17_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 8768);
    
    conv2d_26_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_weights_array.data = AI_PTR(g_model_weights_map[0] + 8832);
    conv2d_26_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 8832);
    
    conv2d_26_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_bias_array.data = AI_PTR(g_model_weights_map[0] + 27264);
    conv2d_26_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 27264);
    
    conv2d_32_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_32_weights_array.data = AI_PTR(g_model_weights_map[0] + 27328);
    conv2d_32_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 27328);
    
    conv2d_32_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_32_bias_array.data = AI_PTR(g_model_weights_map[0] + 33600);
    conv2d_32_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 33600);
    
    conv2d_38_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_38_weights_array.data = AI_PTR(g_model_weights_map[0] + 33604);
    conv2d_38_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 33604);
    
    conv2d_41_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_weights_array.data = AI_PTR(g_model_weights_map[0] + 33996);
    conv2d_41_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 33996);
    
    conv2d_41_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_bias_array.data = AI_PTR(g_model_weights_map[0] + 52428);
    conv2d_41_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 52428);
    
    conv2d_59_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_59_weights_array.data = AI_PTR(g_model_weights_map[0] + 52556);
    conv2d_59_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 52556);
    
    conv2d_59_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_59_bias_array.data = AI_PTR(g_model_weights_map[0] + 53132);
    conv2d_59_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 53132);
    
    conv2d_60_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_60_weights_array.data = AI_PTR(g_model_weights_map[0] + 53196);
    conv2d_60_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 53196);
    
    conv2d_60_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_60_bias_array.data = AI_PTR(g_model_weights_map[0] + 55244);
    conv2d_60_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 55244);
    
    conv2d_45_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_45_weights_array.data = AI_PTR(g_model_weights_map[0] + 55372);
    conv2d_45_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 55372);
    
    conv2d_45_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_45_bias_array.data = AI_PTR(g_model_weights_map[0] + 55948);
    conv2d_45_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 55948);
    
    conv2d_46_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_46_weights_array.data = AI_PTR(g_model_weights_map[0] + 56012);
    conv2d_46_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 56012);
    
    conv2d_46_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_46_bias_array.data = AI_PTR(g_model_weights_map[0] + 58060);
    conv2d_46_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 58060);
    
    conv2d_49_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_49_weights_array.data = AI_PTR(g_model_weights_map[0] + 58188);
    conv2d_49_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 58188);
    
    conv2d_49_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_49_bias_array.data = AI_PTR(g_model_weights_map[0] + 59212);
    conv2d_49_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 59212);
    
    conv2d_52_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_52_weights_array.data = AI_PTR(g_model_weights_map[0] + 59244);
    conv2d_52_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 59244);
    
    conv2d_52_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_52_bias_array.data = AI_PTR(g_model_weights_map[0] + 61548);
    conv2d_52_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 61548);
    
    conv2d_55_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_55_weights_array.data = AI_PTR(g_model_weights_map[0] + 61580);
    conv2d_55_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 61580);
    
    conv2d_55_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_55_bias_array.data = AI_PTR(g_model_weights_map[0] + 62604);
    conv2d_55_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 62604);
    
    conv2d_64_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_64_weights_array.data = AI_PTR(g_model_weights_map[0] + 62732);
    conv2d_64_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 62732);
    
    conv2d_64_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_64_bias_array.data = AI_PTR(g_model_weights_map[0] + 136460);
    conv2d_64_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 136460);
    
    conv2d_70_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_70_weights_array.data = AI_PTR(g_model_weights_map[0] + 136588);
    conv2d_70_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 136588);
    
    conv2d_76_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_76_weights_array.data = AI_PTR(g_model_weights_map[0] + 149132);
    conv2d_76_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 149132);
    
    conv2d_79_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_79_weights_array.data = AI_PTR(g_model_weights_map[0] + 149524);
    conv2d_79_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 149524);
    
    conv2d_79_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_79_bias_array.data = AI_PTR(g_model_weights_map[0] + 186388);
    conv2d_79_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 186388);
    
    conv2d_97_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_97_weights_array.data = AI_PTR(g_model_weights_map[0] + 186516);
    conv2d_97_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 186516);
    
    conv2d_97_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_97_bias_array.data = AI_PTR(g_model_weights_map[0] + 187092);
    conv2d_97_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 187092);
    
    conv2d_98_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_98_weights_array.data = AI_PTR(g_model_weights_map[0] + 187156);
    conv2d_98_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 187156);
    
    conv2d_98_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_98_bias_array.data = AI_PTR(g_model_weights_map[0] + 189204);
    conv2d_98_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 189204);
    
    conv2d_83_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_83_weights_array.data = AI_PTR(g_model_weights_map[0] + 189332);
    conv2d_83_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 189332);
    
    conv2d_83_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_83_bias_array.data = AI_PTR(g_model_weights_map[0] + 189908);
    conv2d_83_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 189908);
    
    conv2d_84_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_84_weights_array.data = AI_PTR(g_model_weights_map[0] + 189972);
    conv2d_84_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 189972);
    
    conv2d_84_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_84_bias_array.data = AI_PTR(g_model_weights_map[0] + 192020);
    conv2d_84_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 192020);
    
    conv2d_87_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_87_weights_array.data = AI_PTR(g_model_weights_map[0] + 192148);
    conv2d_87_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 192148);
    
    conv2d_87_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_87_bias_array.data = AI_PTR(g_model_weights_map[0] + 194196);
    conv2d_87_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 194196);
    
    conv2d_90_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_90_weights_array.data = AI_PTR(g_model_weights_map[0] + 194260);
    conv2d_90_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 194260);
    
    conv2d_90_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_90_bias_array.data = AI_PTR(g_model_weights_map[0] + 203476);
    conv2d_90_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 203476);
    
    conv2d_93_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_93_weights_array.data = AI_PTR(g_model_weights_map[0] + 203540);
    conv2d_93_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 203540);
    
    conv2d_93_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_93_bias_array.data = AI_PTR(g_model_weights_map[0] + 205588);
    conv2d_93_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 205588);
    
    conv2d_102_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_102_weights_array.data = AI_PTR(g_model_weights_map[0] + 205716);
    conv2d_102_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 205716);
    
    conv2d_102_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_102_bias_array.data = AI_PTR(g_model_weights_map[0] + 279444);
    conv2d_102_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 279444);
    
    conv2d_108_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_108_weights_array.data = AI_PTR(g_model_weights_map[0] + 279572);
    conv2d_108_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 279572);
    
    conv2d_114_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_114_weights_array.data = AI_PTR(g_model_weights_map[0] + 292116);
    conv2d_114_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 292116);
    
    conv2d_117_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_117_weights_array.data = AI_PTR(g_model_weights_map[0] + 292508);
    conv2d_117_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 292508);
    
    conv2d_117_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_117_bias_array.data = AI_PTR(g_model_weights_map[0] + 300700);
    conv2d_117_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 300700);
    
    gemm_121_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_121_weights_array.data = AI_PTR(g_model_weights_map[0] + 300956);
    gemm_121_weights_array.data_start = AI_PTR(g_model_weights_map[0] + 300956);
    
    gemm_121_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_121_bias_array.data = AI_PTR(g_model_weights_map[0] + 302236);
    gemm_121_bias_array.data_start = AI_PTR(g_model_weights_map[0] + 302236);
    
    reduce_112_Placeholder_array.format |= AI_FMT_FLAG_CONST;
    reduce_112_Placeholder_array.data = AI_PTR(g_model_weights_map[0] + 302256);
    reduce_112_Placeholder_array.data_start = AI_PTR(g_model_weights_map[0] + 302256);
    
    reduce_74_Placeholder_array.format |= AI_FMT_FLAG_CONST;
    reduce_74_Placeholder_array.data = AI_PTR(g_model_weights_map[0] + 302260);
    reduce_74_Placeholder_array.data_start = AI_PTR(g_model_weights_map[0] + 302260);
    
    reduce_36_Placeholder_array.format |= AI_FMT_FLAG_CONST;
    reduce_36_Placeholder_array.data = AI_PTR(g_model_weights_map[0] + 302264);
    reduce_36_Placeholder_array.data_start = AI_PTR(g_model_weights_map[0] + 302264);
    
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
      
      .n_macc            = 3320188,
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
      
      .n_macc            = 3320188,
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

