#pragma once

typedef struct {
    size_t n;
    uint32_t *v_share_0_plus_half;  // Party 0's share: Array[n]
	uint32_t *v_share_0_minus_half;  // Party 0's share: Array[n]
    uint32_t *v_share_1;  // Party 1's share: Array[n]
    uint8_t *result_share_0;  // Array[n]
    uint8_t *result_share_1;  // Array[n]
} secureml_sigmoid_args;


void secureml_sigmoid(void *);