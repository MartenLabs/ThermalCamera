/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "i2c.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include "ai_datatypes_defines.h"
#include "model.h"
#include "model_data.h"

#include "MLX90640_API.h"
#include <float.h>
#include <math.h>

#include "dataset.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */



#define HEIGHT 24
#define WIDTH 32

#define FRAME_SIZE (HEIGHT * WIDTH)

#define  FPS2HZ   0x02
#define  FPS4HZ   0x03
#define  FPS8HZ   0x04
#define  FPS16HZ  0x05
#define  FPS32HZ  0x06

#define  MLX90640_ADDR 0x33
#define	 RefreshRate FPS8HZ
#define  TA_SHIFT 8 //Default shift for MLX90640 in open air

#define REDUCED_ROWS 12
#define REDUCED_COLS 16
#define OPENAIR_TA_SHIFT 8

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

#if !defined(AI_MODEL_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_MODEL_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_MODEL_IN_NUM] = {data_in_1};
#else
ai_i8* data_ins[AI_MODEL_IN_NUM] = {NULL};
#endif

#if !defined(AI_MODEL_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_MODEL_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_MODEL_OUT_NUM] = {data_out_1};
#else
ai_i8* data_outs[AI_MODEL_OUT_NUM] = {NULL};
#endif

AI_ALIGNED(32) static uint8_t pool0[AI_MODEL_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

static ai_handle model = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

int max_index = 0;

float ai_input_data[HEIGHT][WIDTH][1];
float input_data[HEIGHT][WIDTH][1];
float output_data[HEIGHT][WIDTH][1];

static uint16_t eeMLX90640[832];
static float mlx90640To[FRAME_SIZE];
const int output_size = 5;
float* output_value;

float frameBuffer[HEIGHT][WIDTH];
float preprocessedFrame[HEIGHT][WIDTH];
uint16_t frame[834];

float emissivity=0.95;
int status;

paramsMLX90640 mlx90640;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void ai_log_err(const ai_error err, const char *fct);
static int ai_boostrap(ai_handle *act_addr);
void AI_Init(void);
int argmax(float* array, int size);
void transform_to_4d(const float input[][WIDTH], float transformed_input[1][HEIGHT][WIDTH][1]);

void getFrame(float framebuf[HEIGHT][WIDTH]);
void captureAndPreprocessFrame(float (*frameBuffer)[WIDTH], float (*outputBuffer)[WIDTH]);
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */
/* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_CRC_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */
  AI_Init();

  MLX90640_SetRefreshRate(MLX90640_ADDR, RefreshRate);
  MLX90640_SetChessMode(MLX90640_ADDR);

  status = MLX90640_DumpEE(MLX90640_ADDR, eeMLX90640);
  if (status != 0) printf("\r\nload system parameters error with code:%d\r\n",status);
  status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
  if (status != 0) printf("\r\nParameter extraction failed with error code:%d\r\n",status);


  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  // I.0      (1,24,32,1)/float32     0.000   1.000    0.236    0.194 input_1
	  // O.0      (1,1,1,5)/float32       0.000   0.999    0.200    0.365 output_1
	  for(uint8_t i = 0; i < 2; i++){
		  getFrame(frameBuffer);
		  captureAndPreprocessFrame(frameBuffer, preprocessedFrame);
	  }

	  float transformed_input[1][HEIGHT][WIDTH][1];
	  transform_to_4d(preprocessedFrame, transformed_input);


	  memcpy(data_ins[0], transformed_input, sizeof(float) * 1 * 24 * 32 * 1);


	  if (ai_model_run(model, ai_input, ai_output) != 1) {
	        ai_log_err(ai_model_get_error(model), "ai_model_run");
	        return -1;
	      }


	  output_value = (float*)data_outs[0];

	  int max_index = argmax(output_value, output_size);
	  printf("Max index: %d\n", max_index);

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}


static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_model_create_and_init(&model, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_model_create_and_init");
    return -1;
  }

  ai_input = ai_model_inputs_get(model, NULL);
  ai_output = ai_model_outputs_get(model, NULL);

#if defined(AI_MODEL_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_MODEL_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_MODEL_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_MODEL_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_MODEL_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_MODEL_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

void AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}



int argmax(float* array, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] > array[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}


void transform_to_4d(const float input[][WIDTH], float transformed_input[1][HEIGHT][WIDTH][1]) {
    for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH; w++) {
            // �??? ?��?? 값을 255.0?���??? ?��?��?�� 0?��?�� 1 ?��?��?�� 값으�??? ?��규화
            transformed_input[0][h][w][0] = input[h][w] / 255.0f;
        }
    }
}

void captureAndPreprocessFrame(float (*frameBuffer)[WIDTH], float (*outputBuffer)[WIDTH]) {
    float minVal = FLT_MAX, maxVal = -FLT_MAX;

    // Find min and max values in the frame buffer
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float val = frameBuffer[i][j];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }

    // Scale frame buffer values to 0-255
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            outputBuffer[i][j] = (frameBuffer[i][j] - minVal) / (maxVal - minVal) * 255.0f;
        }
    }
}

void getFrame(float framebuf[HEIGHT][WIDTH]) {
    static uint16_t mlx90640Frame[834];
    float emissivity = 0.95;
    float tr = 23.15;
    float mlx90640To[768]; // 32x24 pixels

    for (int i = 0; i < 2; i++) {
        int status = MLX90640_GetFrameData(MLX90640_ADDR, mlx90640Frame);
        if (status < 0) {
            printf("Frame data fetch error: %d\n", status);
            return;
        }
        tr = MLX90640_GetTa(mlx90640Frame, &mlx90640) - TA_SHIFT;
        MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, mlx90640To);
        MLX90640_BadPixelsCorrection(mlx90640.brokenPixels, mlx90640To, 1, &mlx90640);
        MLX90640_BadPixelsCorrection(mlx90640.outlierPixels, mlx90640To, 1, &mlx90640);
    }

    float minTemp = FLT_MAX, maxTemp = -FLT_MAX;
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        if (mlx90640To[i] < minTemp) minTemp = mlx90640To[i];
        if (mlx90640To[i] > maxTemp) maxTemp = mlx90640To[i];
    }

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float temp = mlx90640To[i * WIDTH + j];
            // Scale temperatures to 0-255 range without normalizing to 0-1
            framebuf[i][j] = (temp - minTemp) / (maxTemp - minTemp) * 255.0;
        }
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
