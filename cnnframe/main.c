#include "net.h"
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>


int main(int argn, char* argv[]){
	
	if(argn < 2){
		printf("a file name is required \n");
	
		return 1;
	}
	
	IplImage* image = cvLoadImage(argv[1],0);	


	Net net;
	FType jiu;
	int i,j,k;
	LayerType types[14] = {data,conv,bias,pool,sigmoid,conv,bias,pool,sigmoid,conv,bias,sigmoid,fcmf,fcbias};
	int param0[12] = {28,28,1,0,0,0,0,0,0,0,0,0};//data
	int param1[12] = {28,28,1,3,3,1,20,26,26,20,0,1};//conv1
	int param2[12] = {26,26,20,0,0,0,0,26,26,20,0,0};//bias1
	int param3[12] = {26,26,20,0,0,0,0,13,13,20,2,2};//pool1
	int param4[12] = {13,13,20,0,0,0,0,0,0,0,0,0};//relu
	int param5[12] = {13,13,20,2,2,20,20,12,12,20,0,1};//conv2
	int param6[12] = {12,12,20,0,0,0,0,12,12,20,0,0};//bias2
	int param7[12] = {12,12,20,0,0,0,0,6,6,20,2,2};//pool2
	int param8[12] = {6,6,20,0,0,0,0,0,0,0,0,0};//relu
	int param9[12] = {6,6,20,2,2,20,10,5,5,10,0,1};//conv3
	int param10[12] = {5,5,10,0,0,0,0,5,5,10,0,0};//bias3
	int param11[12] = {5,5,10,0,0,0,0,0,0,0,0,0};//relu
	int param12[12] = {5,5,10,250,10,0,0,10,0,0,0,0};//fc1
	int param13[12] = {10,0,0,0,0,0,0,10,0,0,0,0};//fcbias
	int* all_param12[14] = {param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11,param12,param13};
//	void* Space1[3];
//	void* Space2[3];
//	void* Space3[3];
//	void** Space;
//	FeaturemapLayer FML1,FML2;
//	FeaturemapLayerSize FLS1,FLS2;
//	KernelLayer KL,KL2;
//	KernelLayerSize KLS;
//	FCfeature FCF;
//	FCfeatureSize FCFS;
//	FCKernel FCK;
//	FCKernelSize FCKS;
	FILE *outfp,*infp; 
//	uchar buf[] = {1,2,3,4,5,6,7,8,9}; 
//	uchar buf2[] = {0,0,0,0,0,0,0,0,0}; 
	char* kername[14];
//	FLS1.height = 8;FLS1.width = 8;FLS1.channel = 1;
//	FLS2.height = 2;FLS2.width = 2;FLS2.channel = 1;
//	KLS.width = 3;KLS.height=3;KLS.channel=1;KLS.depth=1;
//	FCFS.lenth = 2;
//	FCKS.height = 4;FCKS.width = 2;
//
//	FML1 = newFeaturemapLayer(FLS1);
//	FML2 = newFeaturemapLayer(FLS2);
//	KL = newKernelLayer(KLS);
//	KL2 = newKernelLayer(KLS);
//	FCK = newFCKernel(FCKS);
//	FCF = newFCfeature(FCFS);
//
//	for(j=0;j<1;j++){
//		for(i=0;i<64;i++){
//		FML1[j][i] = (i%16)+1;
//		}
//	}
//
//	for(j=0;j<1;j++){
//		for(i=0;i<9;i++){
//		KL[j][i] = i+1;
//		}
//	}
//	for(j=0;j<4;j++){
//		for(i=0;i<2;i++){
//		FCK[j][i] = i+1;
//		}
//	}
//
//	Space1[0]=FML1;Space1[1]=FML1;Space1[2]=FML1;
//	Space2[0]=FML1;Space2[1]=KL;Space2[2]=FML2;
//	Space3[0]=FML2;Space3[1]=FCK;Space3[2]=FCF;
//
//	Space = (void**)malloc(sizeof(void*)*3);
//	Space[0] = Space1;Space[1] = Space2;Space[2] = Space3;


	InitialNetStruct(&net,14,types,all_param12);
	
	printf("rear1\n");

	//IndexNet(&net,Space);
	//ForwardNet(&net);
	kername[1]= "../../data/param_conv1.bin";
	kername[2]= "../../data/param_conv1_bias.bin";
	kername[5]= "../../data/param_conv2.bin";
	kername[6]= "../../data/param_conv2_bias.bin";
	kername[9]= "../../data/param_conv3.bin";
	kername[10]="../../data/param_conv3_bias.bin";
	kername[12]="../../data/param_fc1.bin";
	kername[13]="../../data/param_fc1_bias.bin";

	InitialNetData(&net);
	printf("rear21\n");
	SetDataNet(&net,kername);	
	
	for(i = 0;i<28*28;i++){
		*(net.layers[0].layerprama.dataLayer.datasource[0]+i) =(*( uchar *)( image->imageData + i))/255.0;
//		printf("%f ",*(net.layers[0].layerprama.dataLayer.datasource[0]+i));
//		if(!(i%28)){
//			printf("\n");
//		}
	}

	
	printf("rear3\n");
	
	ForwardNet(&net);
	jiu = -1000000;
	j = 0;
	for(i=0;i<10;i++){
		if (jiu < (net.layers[13].layerprama.fcbiasLayer.srcFC[i])){
			jiu = (net.layers[13].layerprama.fcbiasLayer.srcFC[i]);
			j = i;
		}
	}
	printf("Is Likely: %d \n",j);
	cvShowImage("1.jpg",image);
	cvWaitKey(0);
	return 0;
}
