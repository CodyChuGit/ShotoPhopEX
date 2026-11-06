// Layered's exact 64x64 kernel reclaims Q staging memory after register loads.
// 34 KiB rather than 51 KiB per CTA lets sm_120 admit two CTAs per SM.
// Shared callers compile the original allocation and entry points unchanged.
#define QIR_FLASH_REUSE_QUERY_SMEM
#define qir_flash_fwd_bias qir_layered_flash_fwd_bias
#define qir_flash_fwd qir_layered_flash_fwd
#include "flash_fwd.cu"
