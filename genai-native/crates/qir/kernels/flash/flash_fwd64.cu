// The same kernel at head dim 64 (Idol's TripoSplat heads): src/flash.rs `attention64`.
#define QIR_FLASH_D 64
#define qir_flash_fwd_bias qir_flash64_fwd_bias
#define qir_flash_fwd qir_flash64_fwd
#include "flash_fwd.cu"
