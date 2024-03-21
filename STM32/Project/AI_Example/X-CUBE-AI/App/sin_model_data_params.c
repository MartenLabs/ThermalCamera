/**
  ******************************************************************************
  * @file    sin_model_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sun Mar 17 17:51:35 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "sin_model_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_sin_model_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_sin_model_weights_array_u64[161] = {
  0x3e920ba0bda2e648U, 0xbee3e84e3eba074dU, 0x3e9b1bbb3e51e567U, 0xbe672ce0bdca1a20U,
  0xbf1508dabf0c9c0aU, 0xbf095f2abe85179cU, 0xbda36ea03eee13c8U, 0x3d686346be4d8976U,
  0x3d97d74500000000U, 0xbf354276U, 0x3eced4f6bf108b3fU, 0x0U,
  0x0U, 0x0U, 0xbf0734adU, 0x3f09c39a00000000U,
  0x3eeafc02be306962U, 0x3bfbb200bf33a39dU, 0x3eaea5b3bee3f5edU, 0xbe911f343e2c46e2U,
  0x3dc01e1cbed979baU, 0xbe22a7e23e6c728aU, 0x3dd5e10cbee2a3cfU, 0x3d76e71dbea26f09U,
  0xbe5a910cbeda121fU, 0x3ed88d7b3ecc3237U, 0xbd797d6d3cc9b0fbU, 0x3ea02c793e42698aU,
  0xbe33f270bd0732b8U, 0xbe34e7d83ecc3d63U, 0xbed1db58be282871U, 0x3c3e413e3ed20a9dU,
  0xbecb7b71bec073ffU, 0x3e5c5bc2bda9f1c4U, 0xbdf3cba4be5be114U, 0x3ed52b5fbe73f3e1U,
  0xbec7fd533c0d34c0U, 0xbd83fc94be65b721U, 0x3df2ec84bed46dffU, 0x3b9add00be7ed86dU,
  0x3ee6c338bdf21058U, 0x3ecaacb7be656125U, 0x3a9225c3bf324e6fU, 0x3d8fc954bed88135U,
  0x3eb4ca353ec7484dU, 0x3ecd46293eb350b1U, 0xbe47d140beaef5d6U, 0x3ef11d0e3c2b9bc0U,
  0xbde22d923db2f61cU, 0x3e228fc6bf1ab9d5U, 0x3e4fa0a1beb84a0eU, 0x3ebdde45bcf71f60U,
  0xbece13c3be7d6031U, 0x3dbd63f43e6297baU, 0xbe9508383e86392aU, 0x3eb12d09beab2146U,
  0x3dc1f166be94834cU, 0x3ebe80573e513bfbU, 0xbedd7cf9bee8fe02U, 0x3e8268ed3e6b40f2U,
  0x3ece7c75be57f183U, 0x3e579cd6be255cc8U, 0xbec86b913e38ac41U, 0xbe9bfbc1be89c7c8U,
  0x3becdf6bbd775ca8U, 0xbe95e61abe557472U, 0x3ba3d685bf0fc974U, 0x3eb99037be4d6052U,
  0x3de2ae4cbec44d92U, 0xbe7179d0bec43055U, 0x3ecf195f3e09df22U, 0x3df74b66be95ed5eU,
  0x3ef26f483e05c312U, 0xbe362366bf1f7ef5U, 0x3e7570dbbeccb396U, 0x3e9e1cfdbe425713U,
  0xbdf010bc3e37ee82U, 0xbea86c663d94a0acU, 0x3dfa1104bede731bU, 0x3ed457a7bcd14b10U,
  0x3ee79a72bed4ab4fU, 0xbe6c5fedbf0447e4U, 0xbcb4cfefbf0aa259U, 0x3e358d963e4619b2U,
  0x3e06cd5a3e9e84b5U, 0xbec37e673e0b34caU, 0x3d213ea8bdf5c33cU, 0x3f63a8023ec48bbdU,
  0xbeaf6a423dfd3c1cU, 0x3e949851bd5ea4a7U, 0x3ed40e6ebf632190U, 0x3ecfac61bebfb82dU,
  0xbed439ed3e3fb46aU, 0x3bab7c80becb8b5eU, 0xba06ce003cc5c916U, 0x3d908105beb2d0a0U,
  0x3e2f99cb3dd622dcU, 0xbe9739043ca3c05fU, 0xbea2d337bec16f40U, 0xbeb9ba3abd5fadb8U,
  0x3e3122f63e92de93U, 0xbea78a123e479432U, 0x3e4a7e023dde4917U, 0x3e8b0eb9be3905beU,
  0x3ea5d477be9310eaU, 0x3e70717ebe342068U, 0x3b4cb421bef5447cU, 0x3ec12009beca15deU,
  0x3eb41adb3ec65d9bU, 0x3ebd016b3ec4182dU, 0xbeb0cd73bdd3fa9dU, 0x3c86e3ce3dc87c2cU,
  0xbe84d3dc3ed0efa9U, 0xbdbebaf0be5d2a77U, 0xbecbcd10bd3b3342U, 0x3d5ca340be78fa8cU,
  0x3ac89100be221084U, 0x3e01dd263aa55800U, 0x3ecff5773d924a94U, 0x3e2e00923e807cbdU,
  0x3d9938bebe5f77efU, 0x3e900835beec517aU, 0x3e6b9808bfbf2b62U, 0x3e817e31bec08ff4U,
  0x3e25bc2e3e8f527dU, 0x3e80c8dbbdfabcfcU, 0x3e25d336be34dffaU, 0x3f516c7d3ed836bbU,
  0x3eae65103e143ef2U, 0x3e9bb439bf10dccaU, 0x3ec7d96cbeff2117U, 0xbde08844be872f5aU,
  0xbc0c9aa0be1c59beU, 0x3dd9d8943e69d29eU, 0x3d976facbf082c9eU, 0x3eb5b8e63ec57c51U,
  0x3e7d270d3dda591cU, 0x3e4aec0a3f089ec1U, 0xbd90e8c33f18242eU, 0x3ec90ec5bec1a6b0U,
  0x3b5009003dff5c84U, 0xbed0c00fbe039044U, 0x3e6a2c623eb83a48U, 0xbf10fba9be9375c0U,
  0xbcbb5cec3dbd0151U, 0x3ddabe9200000000U, 0xbce948c43e1e95a6U, 0x3e06cc003dc2930dU,
  0x3ea82a6d3f304240U, 0x3d8afeac3dadc44fU, 0x3f813972bc05a527U, 0xbf19e4353e024dd5U,
  0xbca725763fa6cbacU, 0x3f8d2807bebb7502U, 0x3e5fcb68be98a718U, 0x3f52020c3ef74cd8U,
  0xbf09ac58bec36deaU, 0x3f317acf3e010172U, 0xbecb70fd3eb858a3U, 0x3f12a7693f8f05fcU,
  0xbee55d42U,
};


ai_handle g_sin_model_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_sin_model_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

