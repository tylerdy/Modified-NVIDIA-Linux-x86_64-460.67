/**********************************************************************
* FILENAME :        rn_kernel.cu             
* 
* DESCRIPTION :
*       Kernel side implementation of ResNet network
*
* NOTES :
*       This file includes implementation of 2D/3D convolution
*       normalisation,pooling,scaling, ReLU,Eltwise
*       and fully connected layer kernels.
* 
* AUTHOR :    Aajna Karki 
*             https://www.linkedin.com/in/aajna/
*********************************************************************/

#ifndef _RN_KERNEL_H_
#define _RN_KERNEL_H_
#include <stdio.h>

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
/* Enable DEBUG printfs */

//#define LAYER2_DEBUG 
//#define POOL_DEBUG
#ifdef CPU 
void executeFirstLayer(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int stride_width,int pad, int col_width,int feature_r,int feature_c,int out)
{
        int kernel = 7;
        int x_pad,y_pad,loopc,loopr;
	float product = 0.0;
	int stride = 0,colstride = 0;
        //out = 1; feature_r = 1; feature_c = 112;
	{
		for(int output =0;output < out ;output++)
		{
                        colstride = 0;
			for(int row =0; row < feature_r ;row++)
			{  
				stride = 0;
				colstride = 3 *(row*stride_width - pad)*col_width;
				colstride = colstride < 0 ? 0 : colstride;
				for(int col =0; col < feature_c ;col++)
				{
					stride = 3 * (col*stride_width - pad);
					stride = stride < 0 ? 0 : stride;
					
					product = 0;
					x_pad = 0; y_pad = 0;
					/* set the loops value */
					loopc = kernel;loopr = kernel;
					/* take care of padding in left hand side of image*/ 
					if(row*stride_width < pad)
					{
						x_pad = pad - row*stride_width;
						loopr = kernel - x_pad;
					} 
					/* take care of padding in upper side of image*/ 
					if( (col*stride_width)  < pad ) 
					{
						y_pad = pad - col*stride_width;
						loopc = kernel - y_pad;
					}
                                                 
					/* take care of padding in right side of image*/ 
					if((col) > (feature_c - pad))
					{
						loopc = col_width - (stride/3); 
					}  
					/* take care of padding in bottom of image */ 
					if(row > feature_r - pad)
					{
						loopr =  col_width - colstride/(3*col_width);
					}
					/* RGB weights and input 7*7*3 , kernel is 7*7 */
					for(int i = 0; i < loopr; i++)
					{			
						for(int j = 0; j < loopc; j++)
						{
							product +=        ((Layer1_Neurons_GPU[i*col_width*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*7 + j + (output * 7*7*3) + kernel*x_pad + y_pad])  
									+ (Layer1_Neurons_GPU[i*col_width*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*7 + 7*7 + j+ (output * 7*7*3)+ kernel*x_pad + y_pad])
									+ (Layer1_Neurons_GPU[i*col_width*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*7 + 7*7*2 + j+ (output * 7*7*3)+kernel*x_pad + y_pad]));
						}
					}
					Layer2_Neurons_GPU[output*feature_r*feature_c + row*feature_c + col] = product;
#ifdef LAYER1_DEBUG			
					printf("%f\n",product);
#endif		
					product = 0.0;

				}
			}
		}
	}
}
/* IN : Layer2_Neurons_GPU // Neurons input
        Layer2_pool_GPU    // output after pooling
        out                // number of outputs 
        out_fr             // feature map size of output in terms of row 
        out_fc             // feature map size of output in terms of column
        kernel             // kernel size
        stride_width       // stride
        in_fr             // feature map size of input in terms of row
        in_fc             // feature map size of input in terms of column 
*/
void pooling(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad)
{
#ifdef POOL_DEBUG
        printf("pooling Activation layer \n");
#endif
	float max = 0.0;
	int downsample = 0;
        int loopr = kernel, loopc = kernel;
	int stride = 0,colstride = 0;
	{
		for(int output =0;output < out ;output++)
		{
			for(int row =0; row < out_fr ;row++)
			{  
                                colstride = (row*stride_width -pad)*in_fr;	
				colstride = colstride < 0 ? 0: colstride; 
				stride = 0;
				for(int col =0; col < out_fc ;col++)
				{
					loopr = kernel; loopc = kernel;
					stride = col*stride_width - pad;	
                                        stride = stride < 0 ? 0: stride; 
                                        
					if(col < pad)
					{
						loopc = kernel - pad;
                                              //  printf("col %d loopc %d\n",col,loopc); 
					}
                                        if(row < pad)
					{
						loopr = kernel - pad; 
                                               // printf("row %d loopr %d\n",row,loopr); 
					}
                                        if(col > out_fc - pad)
					{
						loopc = in_fc - stride; 
                                               // printf("col %d loopc %d\n",col,loopc); 

					}
                                        if(row > out_fr - pad)
					{
						loopr = in_fr - colstride/in_fr; 
                                               // printf("row %d loopr %d\n",row,loopr); 

					}
					for(int i = 0; i < loopr; i++)
					{			
						for(int j = 0; j < loopc; j++)
						{
							if(max < ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
								max =   ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;
					//		printf("%d %d %d %f\n",(output*in_fr*in_fc) + i*in_fc + j + stride + colstride,row,col,max);

						}
					}
					Layer2_pool_GPU[downsample] = max;
#ifdef POOL_DEBUG			
					printf("\n %f %d\n",max,downsample);
#endif	
					max = 0.0;
					downsample++;
				}
			}
		}
	}
}
void poolingAverage(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad)
{
#ifdef POOL_DEBUG
        printf("pooling Activation layer \n");
#endif
	float sum = 0.0;
	int downsample = 0;
        int loopr = kernel, loopc = kernel;
	int stride = 0,colstride = 0;
	{
		for(int output =0;output < out ;output++)
		{
                        int row = 0;
			{  
                                colstride = (row*stride_width -pad)*in_fr;	
				colstride = colstride < 0 ? 0: colstride; 
				stride = 0;
				int col = 0;
                                {
					loopr = kernel; loopc = kernel;
					stride = col*stride_width - pad;	
                                        stride = stride < 0 ? 0: stride; 
                                        
					for(int i = 0; i < loopr; i++)
					{			
						for(int j = 0; j < loopc; j++)
						{
							sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride] ;

						}
					}
					Layer2_pool_GPU[downsample] = (sum/(loopc*loopr));
#ifdef POOL_DEBUG			
					printf("\n %f %d\n",max,downsample);
#endif	
					sum = 0.0;
					downsample++;
				}
			}
		}
	}
}
void execute3Dconvolution(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int in_fr)
{
	float product = 0.0;
        int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
#ifdef CONV_DEBUG
	printf(" 3D convolution with group %d,output %d,feature %d x %d ,stride %d, kernel %d, pad %d, input %d\n",in_fr,out,fr,fc,stride_width,kernel,pad,in_output); 
#endif
	int stride = 0,colstride = 0;
	{
		for(int output =0;output < out  ;output++) /* out = 256 */
		{      
			colstride = 0;	
			for(int row =0; row < in_fr ; row++) /* out = 256 */
			{	
					stride = 0;
					colstride = (row*stride_width - pad)*in_fr;
					colstride = colstride < 0 ? 0 : colstride;
					for(int col =0; col < in_fr ;col++)
					{
						stride = (col*stride_width - pad);
						stride = stride < 0 ? 0 : stride;

						product = 0;
						x_pad = 0; y_pad = 0;
						/* set the loops value */
						loopc = kernel;loopr = kernel;
						/* take care of padding in left hand side of image*/ 
						if(row*stride_width < pad)
						{
							x_pad = pad - row*stride_width;
							loopr = kernel - x_pad;
						} 
						/* take care of padding in upper side of image*/ 
						if( (col*stride_width)  < pad ) 
						{
							y_pad = pad - col*stride_width;
							loopc = kernel - y_pad;
						}

						/* take care of padding in right side of image*/ 
						if((col*stride_width) >= (in_fr - pad))
						{
							loopc = in_fr - (stride); 
						}  
						/* take care of padding in bottom of image */ 
						if((row*stride_width) >= in_fr - pad)
						{
							loopr =  in_fr - colstride/(in_fr);
						}
						for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
						{
							for(int i =0; i < loopr ; i++) // kernel convolution
							{
								for(int j =0; j < loopc ; j++) // kernel convolution
								{
									product += ( Layer2_Neurons_GPU[feature*in_fr*in_fr + i*in_fr + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
								}
							}
						}
#ifdef CONV_DEBUG
						printf("%f\n",product);
#endif                 
						if(loopc > 0 && loopr > 0)
							Layer3_Neurons_GPU[output*fr*fc + row*fc + col] = product;
						product = 0.0;
					}
				}

			}
		}
	}

void executeBnNormLayer(float *Layer_Neurons,float *mean,float *var,int out,int f_size)
{
           /* Calculate mean */
#if 0          
	   for(int i=0;i < size;i++)
           {
                  sum += out[i];
	   }
           mean = sum/size;
           printf("Mean %f\n",mean);   
           /* Calculate standard Deviation */
           for(int i=0;i < size;i++)
	   {
                 variance += pow((out[i] - mean),2);
	   }
           variance = variance/(size-1);

           printf("Variance %f\n",variance);   
           /* Batch Normalisation */
   
           for(int i=0;i < size;i++)
	   {
                float result = (out[i] - mean)/(sqrt(variance) + 1e-5);
                printf("%f\n",result);
	   }
#else
	   for(int output = 0;output < out;output++)
	   {
		   for(int i=0;i < f_size;i++)
		   {
			   Layer_Neurons[output*f_size + i] = (Layer_Neurons[output*f_size + i] - mean[output])/(sqrt(var[output] + 1e-5));
#ifdef BN_LAYER			
			   printf("%f\n",Layer_Neurons[output*f_size + i]);
#endif	
		   }
	   }
#endif
}
void executeEltWiseLayer(float *Layer1_Neurons,float *Layer2_Neurons,float *Layer_Out_Neurons,int size)
{
        /* By default its sum of 2 layers */
	for(int i=0;i < size; i++)
	{
		Layer_Out_Neurons[i] = Layer1_Neurons[i] + Layer2_Neurons[i];
	}
}
void executeScaleLayer(float *Layer_Neurons,float *scale,float *bias,int out,int f_size)
{
#ifdef SCALE_LAYER			
	printf("Execute Scale Activation Layer\n");
#endif
	for(int output = 0;output < out;output++)
	{
		for(int i=0;i < f_size;i++)
		{
			/* Scale Layer = input * scale + bias */
			Layer_Neurons[output*f_size + i]  = (Layer_Neurons[output*f_size + i] * scale[output]) + bias[output];
#ifdef SCALE_LAYER			
			if(out == 256)
				printf("%f\n",Layer_Neurons[output*f_size + i]);
#endif
		}
	}

}
void executeReLULayer(float *Layer_Neurons,int size)
{
	for(int i=0;i < size;i++)
	{
		if(Layer_Neurons[i] < 0)
			Layer_Neurons[i] = 0;
	}

}
void executeFCLayer(float *bias,float *Layer_InNeurons_GPU,float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int output, int input,bool reLU,bool dropout)
{
#ifdef FC_DEBUG
        printf("Execute FC Layer of output : %d input %d\n",output,input);
#endif
        float product = 0.0,max = 0.0; int weight = 0,index = 0;
	for(int out=0; out < output ; out++)
	{
		for(int in = 0; in < input; in++)
		{
                     product += Layer_InNeurons_GPU[in] * Layer_Weights_GPU[weight++];
		}
		if(max < product)
		{
			index = out;
			max = product;
		}
		Layer_OutNeurons_GPU[out] = product;
#ifdef FC_DEBUG
		printf("%f\n",product);
#endif
		product = 0.0;
	}
        printf(" MAX from FC layer = %d\n",index);

}

void executeSoftMax(float *Layer_In_Neurons_GPU)
{
#ifdef SOFTMAX_DEBUG
        printf("executeSoftMax \n");
#endif
        float max = 0,sum = 0;
	float output[1000] = {0};
        for(int i = 0; i < 1000; i++)
	{
		if(Layer_In_Neurons_GPU[i] > max)
			max = Layer_In_Neurons_GPU[i];
	}
#ifdef SOFTMAX_DEBUG
	printf("Max = %10e\n",max);
#endif
        for(int i = 0; i < 1000; i++)
	{
		output[i] = exp(Layer_In_Neurons_GPU[i] - max);
                sum += output[i];
	}
#ifdef SOFTMAX_DEBUG
	printf("Sum =  %10e\n",sum);
#endif
        for(int i = 0; i < 1000; i++)
	{
		output[i] *= (1/sum);
#ifdef SOFTMAX_DEBUG
                printf("%10e\n",output[i]);
#endif
	}
        
}

#else
__global__ void executeFirstLayerCUDA(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int stride_width,int pad, int col_width,int feature_r,int feature_c,int out,int tfactor)
{
	int stride = 0,colstride = 0;
	int output = blockIdx.x;
	int row_even = threadIdx.x * tfactor; 
	int col_even = threadIdx.y * tfactor;
	int kernel = 7;
        int x_pad,y_pad,loopc,loopr;
	float product = 0.0;
        //out = 1; feature_r = 1; feature_c = 112;
	{
		//for(int output =0;output < out ;output++)
                if(row_even < feature_r && col_even < feature_c)
		{
                        colstride = 0;
			for(int row =row_even; row < row_even+tfactor ;row++)
			{  
				stride = 0;
				colstride = 3 *(row*stride_width - pad)*col_width;
				colstride = colstride < 0 ? 0 : colstride;
				for(int col =col_even; col < col_even+tfactor ;col++)
				{
					stride = 3 * (col*stride_width - pad);
					stride = stride < 0 ? 0 : stride;
					
					product = 0;
					x_pad = 0; y_pad = 0;
					/* set the loops value */
					loopc = kernel;loopr = kernel;
					/* take care of padding in left hand side of image*/ 
					if(row*stride_width < pad)
					{
						x_pad = pad - row*stride_width;
						loopr = kernel - x_pad;
					} 
					/* take care of padding in upper side of image*/ 
					if( (col*stride_width)  < pad ) 
					{
						y_pad = pad - col*stride_width;
						loopc = kernel - y_pad;
					}
                                                 
					/* take care of padding in right side of image*/ 
					if((col) > (feature_c - pad))
					{
						loopc = col_width - (stride/3); 
					}  
					/* take care of padding in bottom of image */ 
					if(row > feature_r - pad)
					{
						loopr =  col_width - colstride/(3*col_width);
					}
					/* RGB weights and input 7*7*3 , kernel is 7*7 */
					for(int i = 0; i < loopr; i++)
					{			
						for(int j = 0; j < loopc; j++)
						{
							product +=        ((Layer1_Neurons_GPU[i*col_width*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*7 + j + (output * 7*7*3) + kernel*x_pad + y_pad])  
									+ (Layer1_Neurons_GPU[i*col_width*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*7 + 7*7 + j+ (output * 7*7*3)+ kernel*x_pad + y_pad])
									+ (Layer1_Neurons_GPU[i*col_width*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*7 + 7*7*2 + j+ (output * 7*7*3)+kernel*x_pad + y_pad]));
						}
					}
					Layer2_Neurons_GPU[output*feature_r*feature_c + row*feature_c + col] = product;
#ifdef LAYER1_DEBUG			
					printf("%f\n",product);
#endif		
					product = 0.0;

				}
			}
		}
	}
}
__global__ void poolingAverageCUDA(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad)
{
	float sum = 0.0;
        int loopr = kernel, loopc = kernel;
	int stride = 0,colstride = 0;
	{
		int output = blockIdx.x;//for(int output =0;output < out ;output++)
		{
                        int row = 0;
			{  
                                colstride = (row*stride_width -pad)*in_fr;	
				colstride = colstride < 0 ? 0: colstride; 
				stride = 0;
				int col = 0;
                                {
					loopr = kernel; loopc = kernel;
					stride = col*stride_width - pad;	
                                        stride = stride < 0 ? 0: stride; 
                                        
					for(int i = 0; i < loopr; i++)
					{			
						for(int j = 0; j < loopc; j++)
						{
							sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride] ;

						}
					}
					Layer2_pool_GPU[output] = (sum/(loopc*loopr));
#ifdef POOL_DEBUG			
					printf("\n %f %d\n",max,downsample);
#endif	
					sum = 0.0;
				}
			}
		}
	}
}

__global__ void executepoolingCuda(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad,int tfactor)
{
	float max = 0.0;
	int stride = 0,colstride = 0;
        int output = blockIdx.x;
	int row_even = threadIdx.x * tfactor;
	int col_even = threadIdx.y * tfactor;
        int loopr = kernel, loopc = kernel;
	{
		//for(int output =0;output < out ;output++)
		if(row_even < out_fr && col_even < out_fc)
		{
			for(int row = row_even; row < row_even+tfactor;row++)
			{  
                                colstride = (row*stride_width -pad)*in_fr;	
				colstride = colstride < 0 ? 0: colstride; 
				stride = 0;
				for(int col = col_even; col < col_even +tfactor ;col++)
				{
					loopr = kernel; loopc = kernel;
					stride = col*stride_width - pad;	
                                        stride = stride < 0 ? 0: stride; 
                                        
					if(col < pad)
					{
						loopc = kernel - pad;
                                              //  printf("col %d loopc %d\n",col,loopc); 
					}
                                        if(row < pad)
					{
						loopr = kernel - pad; 
                                               // printf("row %d loopr %d\n",row,loopr); 
					}
                                        if(col > out_fc - pad)
					{
						loopc = in_fc - stride; 
                                               // printf("col %d loopc %d\n",col,loopc); 

					}
                                        if(row > out_fr - pad)
					{
						loopr = in_fr - colstride/in_fr; 
                                               // printf("row %d loopr %d\n",row,loopr); 

					}
					for(int i = 0; i < loopr; i++)
					{			
						for(int j = 0; j < loopc; j++)
						{
							if(max < ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
								max =   ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;
					//		printf("%d %d %d %f\n",(output*in_fr*in_fc) + i*in_fc + j + stride + colstride,row,col,max);

						}
					}
					Layer2_pool_GPU[output*out_fr*out_fc + row*out_fc + col] = max;
					max = 0.0;
				}
			}
		}
	}
}

__global__ void execute3DconvolutionCuda_split(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int in_fr,int tfactor)
{
	float product = 0.0;
        int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
	int stride = 0,colstride = 0;
	{
                int output =   blockIdx.x; // 128
		int row_even = threadIdx.x*tfactor; 
		int col_even = threadIdx.y*tfactor; 
		if(row_even < fr && col_even < fc)//for(int output =0;output < out  ;output++) /* out = 256 */
		{      
                        colstride = 0;	
			for(int row= row_even; row < row_even + tfactor ; row++) /* out = 256 */
			{	
				stride = 0;	
				colstride = (row*stride_width - pad)*in_fr;
				colstride = colstride < 0 ? 0 : colstride;
				
				for(int col =col_even; col < col_even+ tfactor ;col++) /* out = 256 */
				{
					stride = (col*stride_width - pad);
					stride = stride < 0 ? 0 : stride;

					x_pad = 0; y_pad = 0;
					/* set the loops value */
					loopc = kernel;loopr = kernel;
					/* take care of padding in left hand side of image*/ 
					if(row*stride_width < pad)
					{
						x_pad = pad - row*stride_width;
						loopr = kernel - x_pad;
					} 
					/* take care of padding in upper side of image*/ 
					if( (col*stride_width)  < pad ) 
					{
						y_pad = pad - col*stride_width;
						loopc = kernel - y_pad;
					}

					/* take care of padding in right side of image*/ 
					if((col*stride_width) >= (in_fr - pad))
					{
						loopc = in_fr - (stride); 
					}  
					/* take care of padding in bottom of image */ 
					if((row*stride_width) >= in_fr - pad)
					{
						loopr =  in_fr - colstride/(in_fr);
					}

					for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
					{
						for(int i =0; i < loopr ; i++) // kernel convolution
						{
							for(int j =0; j < loopc ; j++) // kernel convolution
							{
								product += ( Layer2_Neurons_GPU[feature*in_fr*in_fr + i*in_fr + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
							}
						}
					}
					if(loopc > 0 && loopr > 0)
						Layer3_Neurons_GPU[output*fr*fc + row*fc + col] = product;

					product = 0;
				}
			}

		}
	}	
}
__global__ void execute3DconvolutionCuda(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int in_fr)
{
	float product = 0.0;
        int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
	int stride = 0,colstride = 0;
	{
                int output = blockIdx.x; // 128
		//for(int output =0;output < out  ;output++) /* out = 256 */
		{      
                        colstride = 0;	
                        int row = threadIdx.x; 
			//for(int row =0; row < fr ; row++) /* out = 256 */
			{	
				stride = 0;	
				colstride = (row*stride_width - pad)*in_fr;
				colstride = colstride < 0 ? 0 : colstride;
				
				int col = threadIdx.y; 
				//for(int col =0; col < fc ;col++) /* out = 256 */
				{
					stride = (col*stride_width - pad);
					stride = stride < 0 ? 0 : stride;

					x_pad = 0; y_pad = 0;
					/* set the loops value */
					loopc = kernel;loopr = kernel;
					/* take care of padding in left hand side of image*/ 
					if(row*stride_width < pad)
					{
						x_pad = pad - row*stride_width;
						loopr = kernel - x_pad;
					} 
					/* take care of padding in upper side of image*/ 
					if( (col*stride_width)  < pad ) 
					{
						y_pad = pad - col*stride_width;
						loopc = kernel - y_pad;
					}

					/* take care of padding in right side of image*/ 
					if((col*stride_width) >= (in_fr - pad))
					{
						loopc = in_fr - (stride); 
					}  
					/* take care of padding in bottom of image */ 
					if((row*stride_width) >= in_fr - pad)
					{
						loopr =  in_fr - colstride/(in_fr);
					}

					for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
					{
						for(int i =0; i < loopr ; i++) // kernel convolution
						{
							for(int j =0; j < loopc ; j++) // kernel convolution
							{
								product += ( Layer2_Neurons_GPU[feature*in_fr*in_fr + i*in_fr + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
							}
						}
					}
					if(loopc > 0 && loopr > 0)
						Layer3_Neurons_GPU[output*fr*fc + row*fc + col] = product;
				}
			}

		}
	}	
}
__global__ void execute3Dconvolutiongroup2Cuda(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group)
{
	float product = 0.0;
	int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
	int stride = 0,colstride = 0;
		{
			/* Execute second set of inputs */
			int output = blockIdx.x + out;
			//for(int output = out ;output < (out << 1)   ;output++) /* out = 256 */
			{      
				colstride = 0;	
                                int row = threadIdx.x;
				//for(int row =0; row < fr; row++) /* out = 256 */
				{	
					stride = 0;	
					if(row > pad)
						colstride = (row - pad) * fr;
					int col = threadIdx.y;
					//for(int col =0; col < fc ;col++) /* out = 256 */
					{
						if(col >= pad)
							stride = col*stride_width;
						x_pad = 0; y_pad = 0;
						/* set the loops value */
						loopc = kernel;loopr = kernel;
						/* take care of padding in left hand side of image*/ 
						if( row < pad)
						{
							x_pad = pad - row;
							loopr = kernel - x_pad;
						} 
						/* take care of padding in upper side of image*/ 
						if( col < pad )
						{
							y_pad = pad - col;
							loopc = kernel - y_pad;
						} 
						/* take care of padding in right side of image*/ 
						if(col >= fc - pad)
							loopc =  fc + pad - col;  
						/* take care of padding in bottom of image */ 
						if(row >= fr - pad)
							loopr =  fr + pad - row;
						for(int feature = in_output ; feature < (in_output << 1) ; feature++) // calculate the feature maps
						{
							for(int i =0; i < loopr ; i++) // kernel convolution
							{
								for(int j =0; j < loopc ; j++) // kernel convolution
								{
									product += (( Layer2_Neurons_GPU[feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + (feature-in_output)*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]));
								}
							}
						}
						product += bias[output];
						if(product < 0) /* ReLU Layer */
							product = 0;
						Layer3_Neurons_GPU[output*fr*fc + row*fc + col] = product;
						product = 0.0;
					}
				}

			}
		}
}
__global__ void executeBnNormLayerCUDA(float *Layer_Neurons,float *mean,float *var,int out,int f_size)
{
           int output = blockIdx.x;
	   //for(int output = 0;output < out;output++)
	   {
                   int i = threadIdx.x * blockDim.x + threadIdx.y;
		   //for(int i=0;i < f_size;i++)
		   {
			   Layer_Neurons[output*f_size + i] = (Layer_Neurons[output*f_size + i] - mean[output])/(sqrt(var[output] + 1e-5));
		   }
	   }
}
__global__ void executeBnNormLayerCUDA_split(float *Layer_Neurons,float *mean,float *var,int out,int f_size,int tfactor)
{
           int output = blockIdx.x;
	   int row_even = threadIdx.x * tfactor;
	   int col_even = threadIdx.y * tfactor;
	   if(row_even < f_size && col_even < f_size)
	   {      
		   for(int row= row_even; row < row_even + tfactor ; row++) 
		   {	
			   for(int col =col_even; col < col_even+ tfactor ;col++) 
			   {
				   { 
					   Layer_Neurons[output*f_size*f_size + row*f_size + col] = (Layer_Neurons[output*f_size*f_size + row*f_size + col] - mean[output])/(sqrt(var[output] + 1e-5));

				   }
			   }
		   }
	   }
}


__global__ void executeScaleLayerCUDA(float *Layer_Neurons,float *scale,float *bias,int out,int f_size)
{
        
	int output = blockIdx.x;
	//for(int output = 0;output < out;output++)
	{
		int i = threadIdx.x * blockDim.x + threadIdx.y;
		//for(int i=0;i < f_size;i++)
		{
			/* Scale Layer = input * scale + bias */
			Layer_Neurons[output*f_size + i]  = (Layer_Neurons[output*f_size + i] * scale[output]) + bias[output];
		}
	}

}
__global__ void executeScaleLayerCUDA_split(float *Layer_Neurons,float *scale,float *bias,int out,int f_size,int tfactor)
{

	int output = blockIdx.x;
	int row_even = threadIdx.x * tfactor;
	int col_even = threadIdx.y * tfactor;
	if(row_even < f_size && col_even < f_size)
	{      
		for(int row= row_even; row < row_even + tfactor ; row++) 
		{	
			for(int col =col_even; col < col_even+ tfactor;col++) 
			{
				{ 
					/* Scale Layer = input * scale + bias */
					Layer_Neurons[output*f_size*f_size + row*f_size + col]  = (Layer_Neurons[output*f_size*f_size + row*f_size + col]  * scale[output]) + bias[output];
				}
			}
		}
	}

}

__global__ void executeEltWiseLayerCUDA(float *Layer1_Neurons,float *Layer2_Neurons,float *Layer_Out_Neurons,int size)
{
        /* By default its sum of 2 layers */
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y ; //for(int i=0;i < size;i++)
	//for(int i=0;i < size; i++)
	{
		Layer_Out_Neurons[i] = Layer1_Neurons[i] + Layer2_Neurons[i];
	}
}
__global__ void executeEltWiseLayerCUDA_split(float *Layer1_Neurons,float *Layer2_Neurons,float *Layer_Out_Neurons,int f_size,int tfactor)
{
	int output = blockIdx.x;
	int row_even = threadIdx.x * tfactor;
	int col_even = threadIdx.y * tfactor;
	if(row_even < f_size && col_even < f_size)
	{      
		for(int row= row_even; row < row_even + tfactor ; row++) 
		{	
			for(int col =col_even; col < col_even+ tfactor ;col++) 
			{
				{
					Layer_Out_Neurons[output*f_size*f_size + row*f_size + col] = Layer1_Neurons[output*f_size*f_size + row*f_size + col] + Layer2_Neurons[output*f_size*f_size + row*f_size + col];
				}
			}
		}
	}
}
__global__ void executeReLULayerCUDA(float *Layer_Neurons,int size)
{
        
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y ; //for(int i=0;i < size;i++)
	{
		if(Layer_Neurons[i] < 0)
			Layer_Neurons[i] = 0;
	}

}

__global__ void executeReLULayerCUDA_split(float *Layer_Neurons,int f_size,int tfactor)
{
       	int output = blockIdx.x;
	int row_even = threadIdx.x * tfactor;
	int col_even = threadIdx.y * tfactor;
	if(row_even < f_size && col_even < f_size)
	{      
		for(int row= row_even; row < row_even + tfactor ; row++) 
		{	
			for(int col =col_even; col < col_even+ tfactor ;col++) 
			{
				{
					if(Layer_Neurons[output*f_size*f_size + row*f_size + col] < 0)
						Layer_Neurons[output*f_size*f_size + row*f_size + col] = 0;
				}
			}
		}
	}
}


__global__ void executeFCLayerCUDA(float *Layer_InNeurons_GPU,float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int input)
{
        float product = 0.0;
        int out = blockIdx.x; //* 32 + blockIdx.y;
	int weight =  out * input;
	//for(int out=0; out < 1000 ; out++)
	{
		for(int in = 0; in < input; in++)
		{
                     product += Layer_InNeurons_GPU[in] * Layer_Weights_GPU[weight+in];
		}
		Layer_OutNeurons_GPU[out] = product;
		product = 0.0;
	}


}
#endif
#endif // #ifndef _RN_KERNEL_H_
