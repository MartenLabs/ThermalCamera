/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
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
#include "MLX90640_API.h"
#include <float.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
#define  FPS2HZ   0x02
#define  FPS4HZ   0x03
#define  FPS8HZ   0x04
#define  FPS16HZ  0x05
#define  FPS32HZ  0x06

#define  MLX90640_ADDR 0x33
#define	 RefreshRate FPS8HZ
#define  TA_SHIFT 8 //Default shift for MLX90640 in open air

#define THERMAL_CAMERA_ROWS 24
#define THERMAL_CAMERA_COLS 32

#define REDUCED_ROWS 12
#define REDUCED_COLS 16
#define OPENAIR_TA_SHIFT 8
#define FRAME_SIZE 768
#define MLX_SHAPE_ROW 24
#define MLX_SHAPE_COL 32


float ai_input_data[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS][1];
float input_data[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS][1];
float output_data[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS][1];
static uint16_t eeMLX90640[832];
static float mlx90640To[768];
uint16_t frame[834];
float emissivity=0.95;
int status;
paramsMLX90640 mlx90640;

//uint8_t ai_input_data[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS][1];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void downscale_and_upscale(float input[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS],
                           float output[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS]);
void bilinear_interpolation(float input[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS],
                            float output[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS]);
void getFrame(float *framebuf);
void captureAndPreprocessFrame(float *frameBuffer, float *outputBuffer);


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
  MX_I2C1_Init();
  MX_USART3_UART_Init();
  MX_CRC_Init();
  /* USER CODE BEGIN 2 */


  MLX90640_SetRefreshRate(MLX90640_ADDR, RefreshRate);
  MLX90640_SetChessMode(MLX90640_ADDR);

  status = MLX90640_DumpEE(MLX90640_ADDR, eeMLX90640);
  if (status != 0) printf("\r\nload system parameters error with code:%d\r\n",status);
  status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
  if (status != 0) printf("\r\nParameter extraction failed with error code:%d\r\n",status);



  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_7, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);

  ai_error ai_err;
  float y_val;
  float frameBuffer[FRAME_SIZE];
  float preprocessedFrame[FRAME_SIZE];


  AI_ALIGNED(4) ai_u8 activations[AI_MODEL_DATA_ACTIVATIONS_SIZE];
  AI_ALIGNED(4) ai_u8 in_data[AI_MODEL_IN_1_SIZE_BYTES];
  AI_ALIGNED(4) ai_u8 out_data[AI_MODEL_OUT_1_SIZE_BYTES];

  ai_handle model = AI_HANDLE_NULL;

  ai_buffer ai_input[AI_MODEL_IN_NUM];
  ai_buffer ai_output[AI_MODEL_OUT_NUM];

  ai_network_params ai_params = {
		  AI_MODEL_DATA_WEIGHTS(ai_model_data_weights_get()),
		  AI_MODEL_DATA_ACTIVATIONS(activations)
  };

  //ai_input[0].data = AI_HANDLE_PTR(in_data);
  //ai_input[0].n_batches = 1;

  //ai_output[0].data = AI_HANDLE_PTR(out_data);
  //ai_output[0].n_batches = 1;

  ai_err = ai_model_create(&model, AI_MODEL_DATA_CONFIG);
  if(ai_err.type != AI_ERROR_NONE) while(1);
  if(!ai_model_init(model, &ai_params)) while(1);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  getFrame(frameBuffer);
	  captureAndPreprocessFrame(frameBuffer, preprocessedFrame);

	  memcpy(ai_input[0].data, preprocessedFrame, AI_MODEL_IN_1_SIZE_BYTES);

	  ai_i32 nbatch = ai_model_run(model, &ai_input[0], &ai_output[0]);
	  if (nbatch != 1) return -1;

	  y_val = ((float *)out_data)[0];
	  //ai_input[0].n_batches = 1;
	  //ai_input[0].data = AI_HANDLE_PTR(in_data);
	  //ai_output[0].n_batches = 1;
	  //ai_output[0].data = AI_HANDLE_PTR(out_data);
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

void captureAndPreprocessFrame(float *frameBuffer, float *outputBuffer) {
    // �????��: frameBuffer?�� ?���??? 캡처?�� ?��?��?�� ?��?��?���??? ?��?��?���??? ?��?��
    float minVal = FLT_MAX, maxVal = -FLT_MAX;

    // 최소/최�? �??? 찾기
    for (int i = 0; i < FRAME_SIZE; i++) {
        if (frameBuffer[i] < minVal) minVal = frameBuffer[i];
        if (frameBuffer[i] > maxVal) maxVal = frameBuffer[i];
    }

    // 0?��?�� 255 ?��?���??? ?��규화
    for (int i = 0; i < FRAME_SIZE; i++) {
        outputBuffer[i] = (frameBuffer[i] - minVal) / (maxVal - minVal) * 255.0f;
    }

    // ?��?�� 0?��?�� 1 ?��?���??? ?��규화
    for (int i = 0; i < FRAME_SIZE; i++) {
        outputBuffer[i] = outputBuffer[i] / 255.0f;
    }
}

void getFrame(float *framebuf) {
    static uint16_t mlx90640Frame[834];
    float emissivity = 0.95;
    float tr = 23.15;
    float mlx90640To[768]; // 32x24 pixels

    for (int i = 0; i < 2; i++) {
        int status = MLX90640_GetFrameData(mlx90640Frame, &mlx90640);
        if (status < 0) {
        	printf("status error");
            return;
        }
        // 공기 중에?��?�� ?��?�� 조정
        tr = MLX90640_GetTa(mlx90640Frame, &mlx90640) - OPENAIR_TA_SHIFT;
        MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, mlx90640To);
        MLX90640_BadPixelsCorrection(mlx90640.brokenPixels, mlx90640To, 1, &mlx90640);
        MLX90640_BadPixelsCorrection(mlx90640.outlierPixels, mlx90640To, 1, &mlx90640);
    }

    float minTemp = FLT_MAX, maxTemp = -FLT_MAX;
    for (int i = 0; i < 768; i++) {
        if (mlx90640To[i] < minTemp) minTemp = mlx90640To[i];
        if (mlx90640To[i] > maxTemp) maxTemp = mlx90640To[i];
    }

    // ?��?�� ?��?��?�� ?���????���??? �??? framebuf?�� ???��
    for (int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
        for (int j = 0; j < THERMAL_CAMERA_COLS; j++) {
            float temp = mlx90640To[i * THERMAL_CAMERA_COLS + j];
            float scaledTemp = (temp - minTemp) / (maxTemp - minTemp);
            framebuf[i * THERMAL_CAMERA_COLS + j] = scaledTemp;
        }
    }
}

void bilinear_interpolation(float input[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS],
                            float output[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS]) {
    for (int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
        for (int j = 0; j < THERMAL_CAMERA_COLS; j++) {
            float top = (i == 0) ? input[i][j] : input[i - 1][j];
            float bottom = (i == THERMAL_CAMERA_ROWS - 1) ? input[i][j] : input[i + 1][j];
            float left = (j == 0) ? input[i][j] : input[i][j - 1];
            float right = (j == THERMAL_CAMERA_COLS - 1) ? input[i][j] : input[i][j + 1];

            output[i][j] = (top + bottom + left + right) / 4.0;
        }
    }
}

void downscale_and_upscale(float input[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS],
                           float output[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS]) {
    float temp[REDUCED_ROWS][REDUCED_COLS];


    for (int i = 0; i < REDUCED_ROWS; i++) {
        for (int j = 0; j < REDUCED_COLS; j++) {
            float sum = input[i*2][j*2] + input[i*2][j*2+1] +
                        input[i*2+1][j*2] + input[i*2+1][j*2+1];
            temp[i][j] = sum / 4.0;
        }
    }


    for (int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
        for (int j = 0; j < THERMAL_CAMERA_COLS; j++) {
            output[i][j] = temp[i/2][j/2];
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
