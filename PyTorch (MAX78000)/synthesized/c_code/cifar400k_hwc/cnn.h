/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the cifar10_new network from a template.
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
  Hardware: 87,979,360 ops (87,590,208 macc; 301,088 comp; 88,064 add; 0 mul; 0 bitwise)
    Layer 0 (conv1): 917,504 ops (884,736 macc; 32,768 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (conv2): 18,939,904 ops (18,874,368 macc; 65,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (gap_conv2): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3 (conv3): 37,814,272 ops (37,748,736 macc; 65,536 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4 (resid1_add): 65,536 ops (0 macc; 0 comp; 65,536 add; 0 mul; 0 bitwise)
    Layer 5 (conv4): 9,519,104 ops (9,437,184 macc; 81,920 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (gap_conv4): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7 (conv5): 9,453,568 ops (9,437,184 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (resid2_add): 16,384 ops (0 macc; 0 comp; 16,384 add; 0 mul; 0 bitwise)
    Layer 9 (conv7): 3,561,472 ops (3,538,944 macc; 22,528 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10 (conv8): 5,314,560 ops (5,308,416 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11 (gap_conv8): 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12 (conv9): 595,968 ops (589,824 macc; 6,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13 (resid3_add): 6,144 ops (0 macc; 0 comp; 6,144 add; 0 mul; 0 bitwise)
    Layer 14 (conv10): 1,771,520 ops (1,769,472 macc; 2,048 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15 (conv11): 3,104 ops (1,024 macc; 2,080 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16 (classifier): 320 ops (320 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 306,336 bytes out of 442,368 bytes total (69.2%)
  Bias memory:   650 bytes out of 2,048 bytes total (31.7%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 10

/* Use this timer to time the inference */
#define CNN_INFERENCE_TIMER MXC_TMR0

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
