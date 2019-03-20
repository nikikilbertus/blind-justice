#pragma once

typedef struct {
    size_t n;
    uint32_t *theta_0_0;  // Party 0's share of theta_0: Array[n]
    uint32_t *theta_1_0;  // Party 0's share of theta_1: Array[n]
    uint32_t *theta_2_0;  // Party 0's share of theta_2: Array[n]
    uint32_t *theta_0_1;  // Party 1's share of theta_0: Array[n]
    uint32_t *theta_1_1;  // Party 1's share of theta_1: Array[n]
    uint32_t *theta_2_1;  // Party 1's share of theta_2: Array[n]
    uint8_t s_0_0;  // Party 0's share of selector for theta_0: Array[n]
    uint8_t s_1_0;  // Party 0's share of selector for theta_1: Array[n]
    uint8_t s_2_0;  // Party 0's share of selector for theta_2: Array[n]
    uint8_t s_0_1;  // ...
    uint8_t s_1_1;
    uint8_t s_2_1;
    uint32_t *result_share_0;  // Array[n]
    uint32_t *result_share_1;  // Array[n]
} select_theta_args;


void select_theta(void *);
