/**********************************************************************
* FILENAME :        gru_kernel.cu             
* 
* DESCRIPTION :
*       Kernel side implementation of GRU network
*
* NOTES :
*       This file includes implementation of GRU cell 
*       Device function of math function such as hard_sigmoid
*
* AUTHOR :    Aajna Karki 
*             https://www.linkedin.com/in/aajna/
*********************************************************************/
#include<stdio.h>
#include<math.h>

float sigmoid(float x)
{
    return 0.5*tanh(0.5*x) + 0.5;
}
__device__ float hard_sigmoid(float x)
{
    if (x<-2.5)
        return 0;
    else
        if(x>2.5)
            return 1;
        else
            return(0.2*x + 0.5);
}

__device__ float gpu_sigmoid(float x)
{
    return 0.5*tanh(0.5*x) + 0.5;
}

/*
* x_t : input to the cell
* GPU_states : previous hidden state (h(t-1)) which is updated to new hidden state (h(t)) at the end of the code
* GPU_kernel_h_w : weight for input (Wh) size of these weights is input_size * hidden_size (100 * 100 in this code)
* GPU_kernel_r_w : weight for input (Wr)
* GPU_kernel_z_w : weight for intput (Wz)
* GPU_bias_h : bias (bh)
* GPU_bias_r : bias (br)
* GPU_bias_z : bias (bz)
* GPU_rec_h_w : weight for hidden states (Uh) size of these weights is hidden_size * hidden_size (100*100 in this code)
* GPU_rec_r_w : weight for hidden states (Ur)
* GPU_rec_z_w : weight for hidden states (Uz)
*
* The following equations will be executed:
* z (Reset Gate) = sigmoid((x_t * Wz) + (h(t-1) * Uz) + bz)
* r (Update Gate) = sigmoid((x_t * Wr) + (h(t-1) * Ur) + br)
* h(t)~ = tanh((x_t * Wh) + dot_product(r, h(t-1)) * Uh + bh)
* h(t) (New Hidden State) = dot_product(1-z, h(t-1)) + dot_product(z, h_p)
*
* The code was implemented by referring to the following code:
* https://www.data-blogger.com/2017/08/27/gru-implementation-tensorflow/
*/

__global__ void GPU_forward_pass_gru(float *x_t, float *GPU_states ,float *GPU_kernel_h_w,float *GPU_kernel_r_w,float *GPU_kernel_z_w,float *GPU_bias_h,float *GPU_bias_r,float *GPU_bias_z,float *GPU_rec_h_w,float *GPU_rec_r_w,float *GPU_rec_z_w)
{
    int index = threadIdx.x * blockDim.x + threadIdx.y;
    float z,h;
    // Number of units = 100, hardcoded to 100
    int num_units = 100;
    __shared__  float update_gate[100];
        
	float rec_z = 0.0,rec_r = 0.0,rec_h = 0.0;

    // calculate h(t-1)*Uz and h(t-1)*Ur (matrix mul operation)
    for(int i=0;i < num_units; i++)
	{
         rec_z += GPU_states[i] * GPU_rec_z_w[index + num_units * i];
         rec_r += GPU_states[i] * GPU_rec_r_w[index + num_units * i];
              
	}

    // calculate Reset Gate
	z = hard_sigmoid((x_t[0] * GPU_kernel_z_w[index]) 
        + rec_z + GPU_bias_z[index]);

    // calculate Update Gate
	update_gate[index] = hard_sigmoid((x_t[0] * GPU_kernel_r_w[index]) 
                         + rec_r + GPU_bias_r[index]);

	__syncthreads();

    // calculate dot_product(r, h(t-1)) * Uh
    for(int i=0;i < num_units; i++)
	{
         rec_h += update_gate[i] * GPU_states[i] * GPU_rec_h_w[index + num_units * i];
	}

    // calculate h(t)~
	h = tanh((x_t[0] * GPU_kernel_h_w[index]) +  rec_h + GPU_bias_h[index]);

    // calculate h(t)
	GPU_states[index] = (1 - z) * GPU_states[index] + (z) * h;
}
















