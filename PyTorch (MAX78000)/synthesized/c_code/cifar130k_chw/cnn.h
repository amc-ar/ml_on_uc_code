/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the cifar130k network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 122,803,712 ops (122,467,328 macc; 336,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0 (conv1_Conv_8): 917,504 ops (884,736 macc; 32,768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (conv2_Conv_6): 18,939,904 ops (18,874,368 macc; 65,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (conv3_Conv_6): 37,814,272 ops (37,748,736 macc; 65,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3 (conv4_Conv_6): 28,360,704 ops (28,311,552 macc; 49,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4 (conv5_Conv_6): 21,282,816 ops (21,233,664 macc; 49,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5 (conv6_Conv_6): 14,188,544 ops (14,155,776 macc; 32,768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (conv7_Conv_6): 1,216,512 ops (1,179,648 macc; 36,864 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7 (conv8_Conv_6): 78,336 ops (73,728 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (fc_MatMul_3): 5,120 ops (5,120 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 129,248 bytes out of 442,368 bytes total (29.2%)
  Bias memory:   322 bytes out of 2,048 bytes total (15.7%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 5

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
