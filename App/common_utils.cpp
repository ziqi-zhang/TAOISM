// #include "common_with_enclaves.hpp"
#include "common_utils.h"

void PrintEncDims(EncDimsT* EncDims) {
    printf("TotalNumElem: %d\n",       EncDims->TotalNumElem);
    printf("NumBatches: %d\n",         EncDims->NumBatches);
    printf("NumRowsInLastShard: %d\n", EncDims->NumRowsInLastShard);
    printf("NumRowsInShard: %d\n",     EncDims->NumRowsInShard);
    printf("NumRows: %d\n",            EncDims->NumRows);
    printf("NumCols: %d\n",            EncDims->NumCols);
}

int GetNumBatches(EncDimsT* EncDims) {
    return EncDims->NumBatches;
}

int GetNumRowsThisShard(EncDimsT* EncDims, int i) {
    if (i < (GetNumBatches(EncDims) - 1)) {
        return EncDims->NumRowsInShard;
    } else {
        return EncDims->NumRowsInLastShard;
    }
}

int GetNumElemInBatch(EncDimsT* EncDims, int i) {
    return GetNumRowsThisShard(EncDims, i) * EncDims->NumCols;
}

int GetSizeOfBatch(EncDimsT* EncDims, int i) {
    return GetNumElemInBatch(EncDims, i) * sizeof(DtypeForCpuOp);
}

int GetNumCols(EncDimsT* EncDims, int i) {
    return EncDims->NumCols;
}

uint32_t CalcEncDataSize(const uint32_t add_mac_txt_size, const uint32_t txt_encrypt_size) {
    if(add_mac_txt_size > UINT32_MAX - txt_encrypt_size)
        return UINT32_MAX;
    uint32_t payload_size = add_mac_txt_size + txt_encrypt_size + sizeof(sgx_aes_gcm_data_t); //Calculate the payload size

    if(payload_size > UINT32_MAX - sizeof(sgx_sealed_data_t))
        return UINT32_MAX;
    return (uint32_t)(sizeof(sgx_sealed_data_t) + payload_size);
}
