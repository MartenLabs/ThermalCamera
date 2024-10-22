//#ifdef __cplusplus
// extern "C" {
//#endif
//
///* Includes ------------------------------------------------------------------*/
//
//#if defined ( __ICCARM__ )
//#elif defined ( __CC_ARM ) || ( __GNUC__ )
//#endif
//
///* System headers */
//#include <stdint.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include <inttypes.h>
//#include <string.h>
//
//#include "ai.h"
//#include "main.h"
//#include "ai_datatypes_defines.h"
//#include "sin_model.h"
//#include "sin_model_data.h"
//
///* USER CODE BEGIN includes */
///* USER CODE END includes */
//
///* IO buffers ----------------------------------------------------------------*/
//
//#if !defined(AI_SIN_MODEL_INPUTS_IN_ACTIVATIONS)
//AI_ALIGNED(4) ai_i8 data_in_1[AI_SIN_MODEL_IN_1_SIZE_BYTES];
//ai_i8* data_ins[AI_SIN_MODEL_IN_NUM] = {
//data_in_1
//};
//#else
//ai_i8* data_ins[AI_SIN_MODEL_IN_NUM] = {
//NULL
//};
//#endif
//
//#if !defined(AI_SIN_MODEL_OUTPUTS_IN_ACTIVATIONS)
//AI_ALIGNED(4) ai_i8 data_out_1[AI_SIN_MODEL_OUT_1_SIZE_BYTES];
//ai_i8* data_outs[AI_SIN_MODEL_OUT_NUM] = {
//data_out_1
//};
//#else
//ai_i8* data_outs[AI_SIN_MODEL_OUT_NUM] = {
//NULL
//};
//#endif
//
///* Activations buffers -------------------------------------------------------*/
//
//AI_ALIGNED(32)
//static uint8_t pool0[AI_SIN_MODEL_DATA_ACTIVATION_1_SIZE];
//
//ai_handle data_activations0[] = {pool0};
//
///* AI objects ----------------------------------------------------------------*/
//
//static ai_handle sin_model = AI_HANDLE_NULL;
//
//static ai_buffer* ai_input;
//static ai_buffer* ai_output;
//
//static void ai_log_err(const ai_error err, const char *fct)
//{
//  /* USER CODE BEGIN log */
//  if (fct)
//    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
//        err.type, err.code);
//  else
//    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);
//
//  do {} while (1);
//  /* USER CODE END log */
//}
//
//static int ai_boostrap(ai_handle *act_addr)
//{
//  ai_error err;
//
//  /* Create and initialize an instance of the model */
//  err = ai_sin_model_create_and_init(&sin_model, act_addr, NULL);
//  if (err.type != AI_ERROR_NONE) {
//    ai_log_err(err, "ai_sin_model_create_and_init");
//    return -1;
//  }
//
//  ai_input = ai_sin_model_inputs_get(sin_model, NULL);
//  ai_output = ai_sin_model_outputs_get(sin_model, NULL);
//
//#if defined(AI_SIN_MODEL_INPUTS_IN_ACTIVATIONS)
//  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
//   *  used from the activations buffer. This is not mandatory.
//   */
//  for (int idx=0; idx < AI_SIN_MODEL_IN_NUM; idx++) {
//	data_ins[idx] = ai_input[idx].data;
//  }
//#else
//  for (int idx=0; idx < AI_SIN_MODEL_IN_NUM; idx++) {
//	  ai_input[idx].data = data_ins[idx];
//  }
//#endif
//
//#if defined(AI_SIN_MODEL_OUTPUTS_IN_ACTIVATIONS)
//  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
//   *  used from the activations buffer. This is no mandatory.
//   */
//  for (int idx=0; idx < AI_SIN_MODEL_OUT_NUM; idx++) {
//	data_outs[idx] = ai_output[idx].data;
//  }
//#else
//  for (int idx=0; idx < AI_SIN_MODEL_OUT_NUM; idx++) {
//	ai_output[idx].data = data_outs[idx];
//  }
//#endif
//
//  return 0;
//}
//
//static int ai_run(void)
//{
//  ai_i32 batch;
//
//  batch = ai_sin_model_run(sin_model, ai_input, ai_output);
//  if (batch != 1) {
//    ai_log_err(ai_sin_model_get_error(sin_model),
//        "ai_sin_model_run");
//    return -1;
//  }
//
//  return 0;
//}
//
///* USER CODE BEGIN 2 */
//int acquire_and_process_data(ai_i8* data[])
//{
//  /* fill the inputs of the c-model
//  for (int idx=0; idx < AI_SIN_MODEL_IN_NUM; idx++ )
//  {
//      data[idx] = ....
//  }
//
//  */
//  return 0;
//}
//
//int post_process(ai_i8* data[])
//{
//  /* process the predictions
//  for (int idx=0; idx < AI_SIN_MODEL_OUT_NUM; idx++ )
//  {
//      data[idx] = ....
//  }
//
//  */
//  return 0;
//}
///* USER CODE END 2 */
//
///* Entry points --------------------------------------------------------------*/
//
//void MX_X_CUBE_AI_Init(void)
//{
//    /* USER CODE BEGIN 5 */
//  printf("\r\nTEMPLATE - initialization\r\n");
//
//  ai_boostrap(data_activations0);
//    /* USER CODE END 5 */
//}
//
//void MX_X_CUBE_AI_Process(void)
//{
//    /* USER CODE BEGIN 6 */
//  int res = -1;
//
//  printf("TEMPLATE - run - main loop\r\n");
//
//  if (sin_model) {
//
//    do {
//      /* 1 - acquire and pre-process input data */
//      res = acquire_and_process_data(data_ins);
//      /* 2 - process the data - call inference engine */
//      if (res == 0)
//        res = ai_run();
//      /* 3- post-process the predictions */
//      if (res == 0)
//        res = post_process(data_outs);
//    } while (res==0);
//  }
//
//  if (res) {
//    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
//    ai_log_err(err, "Process has FAILED");
//  }
//    /* USER CODE END 6 */
//}
//#ifdef __cplusplus
//}
//#endif
