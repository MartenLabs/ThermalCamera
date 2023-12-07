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
#include "i2c.h"
#include "usart.h"
#include "gpio.h"

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

static uint16_t eeMLX90640[832];
static float mlx90640To[768];
uint16_t frame[834];
float emissivity=0.95;
int status;
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
  /* USER CODE BEGIN 2 */


  MLX90640_SetRefreshRate(MLX90640_ADDR, RefreshRate);
  MLX90640_SetChessMode(MLX90640_ADDR);
  paramsMLX90640 mlx90640;
  status = MLX90640_DumpEE(MLX90640_ADDR, eeMLX90640);
  if (status != 0) printf("\r\nload system parameters error with code:%d\r\n",status);
  status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
  if (status != 0) printf("\r\nParameter extraction failed with error code:%d\r\n",status);
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
	  		float vdd = MLX90640_GetVdd(frame, &mlx90640);
	  		float Ta = MLX90640_GetTa(frame, &mlx90640);

	  		float tr = Ta - TA_SHIFT; //Reflected temperature based on the sensor ambient temperature
	  		printf("vdd:  %f Tr: %f\r\n",vdd,tr);
	  		MLX90640_CalculateTo(frame, &mlx90640, emissivity , tr, mlx90640To);

//	  		char uartBuffer[100]; // UART 전송을 위한 버퍼
//	  		int idx; // 현재 버퍼의 인덱스
//
//	  		for(int i = 0; i < 768; i++) {
//	  		    if(i % 32 == 0 && i != 0) {
//	  		        // 줄바꿈 문자 추가
//	  		        idx += snprintf(&uartBuffer[idx], sizeof(uartBuffer) - idx, "\r\n");
//	  		        // UART 전송
//	  		        HAL_UART_Transmit(&huart3, (uint8_t*)uartBuffer, strlen(uartBuffer), HAL_MAX_DELAY);
//	  		        idx = 0; // 버퍼 인덱스 초기화
//	  		    }
//	  		    // 온도 값을 문자열로 변환하고 버퍼에 추가
//	  		    idx += snprintf(&uartBuffer[idx], sizeof(uartBuffer) - idx, "%2.2f ", mlx90640To[i]);
//	  		}
//
//	  		// 마지막 데이터 전송
//	  		HAL_UART_Transmit(&huart3, (uint8_t*)uartBuffer, strlen(uartBuffer), HAL_MAX_DELAY);
//
//	  		HAL_Delay(500);


	  		// 데이터셋의 최소값과 최대값 찾기
	  		float minTemp = FLT_MAX;
	  		float maxTemp = -FLT_MAX;

	  		for(int i = 0; i < 768; i++) {
	  		    if(mlx90640To[i] < minTemp) minTemp = mlx90640To[i];
	  		    if(mlx90640To[i] > maxTemp) maxTemp = mlx90640To[i];
	  		}

	  		char uartBuffer[32 * 6 + 2]; // 충분히 큰 버퍼를 할당
	  		int idx = 0; // 버퍼 인덱스

	  		// 데이터 정규화 및 UART 전송
	  		for(int i = 0; i < 768; i++) {
	  		    if(i % 32 == 0) {
	  		        if(i != 0) {
	  		            // 줄바꿈 문자 추가 및 UART 전송
	  		            uartBuffer[idx++] = '\r';
	  		            uartBuffer[idx++] = '\n';
	  		            HAL_UART_Transmit(&huart3, (uint8_t*)uartBuffer, idx, HAL_MAX_DELAY);
	  		            idx = 0; // 버퍼 인덱스 초기화
	  		        } else {
	  		            // 시작 신호 추가
	  		            const char* startSignal = "start ";
	  		            idx += snprintf(&uartBuffer[idx], sizeof(uartBuffer) - idx, "%s", startSignal);
	  		        }
	  		    }

	  		    // 온도 값을 0-255 범위로 정규화
	  		    uint8_t scaledTemp = (uint8_t)((mlx90640To[i] - minTemp) / (maxTemp - minTemp) * 255.0);

	  		    // 정규화된 온도 값을 문자열로 변환하고 버퍼에 추가
	  		    idx += snprintf(&uartBuffer[idx], sizeof(uartBuffer) - idx, "%u ", scaledTemp);
	  		}

	  		if(idx > 0) {
	  		    // 종료 신호 추가
	  		    const char* endSignal = " end";
	  		    idx += snprintf(&uartBuffer[idx], sizeof(uartBuffer) - idx, "%s", endSignal);

	  		    // 마지막 데이터 전송
	  		    HAL_UART_Transmit(&huart3, (uint8_t*)uartBuffer, idx, HAL_MAX_DELAY);
	  		}

	  		HAL_Delay(5000);


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
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
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
