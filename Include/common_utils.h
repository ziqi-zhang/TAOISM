#include "common_with_enclaves.h"

void PrintEncDims(EncDimsT* EncDims);

int GetNumBatches(EncDimsT* EncDims);

int GetNumRowsThisShard(EncDimsT* EncDims, int i);

int GetNumElemInBatch(EncDimsT* EncDims, int i);

int GetSizeOfBatch(EncDimsT* EncDims, int i);

int GetNumCols(EncDimsT* EncDims, int i);

uint32_t CalcEncDataSize(const uint32_t add_mac_txt_size, const uint32_t txt_encrypt_size);