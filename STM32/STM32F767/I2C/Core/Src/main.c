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
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "MLX90640_API.h"
#include <float.h>
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
#define	 RefreshRate FPS16HZ
#define  TA_SHIFT 8 //Default shift for MLX90640 in open air

#define THERMAL_CAMERA_ROWS 24
#define THERMAL_CAMERA_COLS 32
float ai_input_data[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS][1];

static uint16_t eeMLX90640[832];
static float mlx90640To[768];
uint16_t frame[834];
float emissivity=0.95;
int status;


//uint8_t ai_input_data[THERMAL_CAMERA_ROWS][THERMAL_CAMERA_COLS][1];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

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
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */


  MLX90640_SetRefreshRate(MLX90640_ADDR, RefreshRate);
  MLX90640_SetChessMode(MLX90640_ADDR);
  paramsMLX90640 mlx90640;
  status = MLX90640_DumpEE(MLX90640_ADDR, eeMLX90640);
  if (status != 0) printf("\r\nload system parameters error with code:%d\r\n",status);
  status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
  if (status != 0) printf("\r\nParameter extraction failed with error code:%d\r\n",status);



  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_7, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  int status = MLX90640_GetFrameData(MLX90640_ADDR, frame);
	  if (status < 0)
	  {
	  	printf("GetFrame Error: %d\r\n",status);
	  }
//	  		float vdd = MLX90640_GetVdd(frame, &mlx90640);
//	  		float Ta = MLX90640_GetTa(frame, &mlx90640);
//
//	  		float tr = Ta - TA_SHIFT;
//	  		MLX90640_CalculateTo(frame, &mlx90640, emissivity , tr, mlx90640To);
//
//	  			float minTemp = FLT_MAX;
//	  		    float maxTemp = -FLT_MAX;
//
//	  		    for(int i = 0; i < 768; i++) {
//	  		        if(mlx90640To[i] < minTemp) minTemp = mlx90640To[i];
//	  		        if(mlx90640To[i] > maxTemp) maxTemp = mlx90640To[i];
//	  		    }
//
//
//	  		    for(int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
//	  		        for(int j = 0; j < THERMAL_CAMERA_COLS; j++) {
//	  		            float temp = mlx90640To[i * THERMAL_CAMERA_COLS + j];
//	  		            uint8_t scaledTemp = (uint8_t)((temp - minTemp) / (maxTemp - minTemp) * 255.0);
//
//	  		            ai_input_data[i][j][0] = scaledTemp;
//	  		        }
//	  		    }

	  float vdd = MLX90640_GetVdd(frame, &mlx90640);
	  float Ta = MLX90640_GetTa(frame, &mlx90640);

	  float tr = Ta - TA_SHIFT;
	  MLX90640_CalculateTo(frame, &mlx90640, emissivity , tr, mlx90640To);

	  float minTemp = FLT_MAX;
	  float maxTemp = -FLT_MAX;

	  for(int i = 0; i < 768; i++) {
	      if(mlx90640To[i] < minTemp) minTemp = mlx90640To[i];
	      if(mlx90640To[i] > maxTemp) maxTemp = mlx90640To[i];
	  }

	  for(int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
	      for(int j = 0; j < THERMAL_CAMERA_COLS; j++) {
	          float temp = mlx90640To[i * THERMAL_CAMERA_COLS + j];
	          float scaledTemp = (temp - minTemp) / (maxTemp - minTemp); // ï¿??ê²½ëœ ï¿??ï¿??

	          ai_input_data[i][j][0] = scaledTemp;
	      }
	  }

	  char buffer[100];

	  for(int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
	      int idx = 0;
	      for(int j = 0; j < THERMAL_CAMERA_COLS; j++) {
	          idx += snprintf(&buffer[idx], sizeof(buffer) - idx, "%.2f ", ai_input_data[i][j][0]); // ï¿??ê²½ëœ ï¿??ï¿??
	          if (idx >= sizeof(buffer) - 5) {
	              HAL_UART_Transmit(&huart3, (uint8_t*)buffer, idx, HAL_MAX_DELAY);
	              idx = 0;
	          }
	      }
	      buffer[idx++] = '\r';
	      buffer[idx++] = '\n';
	      HAL_UART_Transmit(&huart3, (uint8_t*)buffer, idx, HAL_MAX_DELAY);
	  }

//	  int status = MLX90640_GetFrameData(MLX90640_ADDR, frame);
//	  if (status < 0) {
//	      printf("GetFrame Error: %d\r\n", status);
//	  } else {
//	      float vdd = MLX90640_GetVdd(frame, &mlx90640);
//	      float Ta = MLX90640_GetTa(frame, &mlx90640);
//
//	      float tr = Ta - TA_SHIFT; // TA_SHIFT is a constant that needs to be defined somewhere
//	      MLX90640_CalculateTo(frame, &mlx90640, emissivity, tr, mlx90640To);
//
//	      float minTemp = FLT_MAX;
//	      float maxTemp = -FLT_MAX;
//
//	      // Reset min and max with each new frame
//	      for (int i = 0; i < 768; i++) {
//	          minTemp = fmin(minTemp, mlx90640To[i]);
//	          maxTemp = fmax(maxTemp, mlx90640To[i]);
//	      }
//
//	      // Normalize and scale the temperature data for ai input
//	      for (int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
//	          for (int j = 0; j < THERMAL_CAMERA_COLS; j++) {
//	              float temp = mlx90640To[i * THERMAL_CAMERA_COLS + j];
//	              uint8_t scaledTemp = (uint8_t)((temp - minTemp) / (maxTemp - minTemp) * 255.0);
//
//	              ai_input_data[i][j][0] = scaledTemp;
//	          }
//	      }
//	  }


//	  		  	  		  char buffer[100];
//
//	  		  	  		     for(int i = 0; i < THERMAL_CAMERA_ROWS; i++) {
//	  		  	  		         int idx = 0;
//	  		  	  		         for(int j = 0; j < THERMAL_CAMERA_COLS; j++) {
//	  		  	  		             idx += snprintf(&buffer[idx], sizeof(buffer) - idx, "%u ", ai_input_data[i][j][0]);
//	  		  	  		             if (idx >= sizeof(buffer) - 5) {
//	  		  	  		            	 HAL_UART_Transmit(&huart3, (uint8_t*)buffer, idx, HAL_MAX_DELAY);
//	  		  	  		                 idx = 0;
//	  		  	  		             }
//	  		  	  		         }
//	  		  	  		         buffer[idx++] = '\r';
//	  		  	  		         buffer[idx++] = '\n';
//	  		  	  		         HAL_UART_Transmit(&huart3, (uint8_t*)buffer, idx, HAL_MAX_DELAY);
//	  		  	  		     }



//	  		  acquire_and_process_data(ai_input_data);

    /* USER CODE END WHILE */

  MX_X_CUBE_AI_Process();
    /* USER CODE BEGIN 3 */
  HAL_Delay(1000);
//  HAL_Delay(1000);
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
