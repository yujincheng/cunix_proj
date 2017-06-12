#include "cnn.h"

//新建FeaturemapLayer
FeaturemapLayer newFeaturemapLayer(FeaturemapLayerSize FLsize){
	int i = 0;
	FeaturemapLayer FL;
	FL = (FeaturemapLayer)malloc(sizeof(Featuremap)*FLsize.channel);
	for (i=0;i<FLsize.channel;i++){
		FL[i] = (Featuremap)malloc(sizeof(FType)*FLsize.height*FLsize.width);
		memset(FL[i],sizeof(FType)*FLsize.height*FLsize.width,0);
	}
	return FL;
}
//删除FeaturemapLayer
void deleteFeaturemapLayer(FeaturemapLayer FL,FeaturemapLayerSize FLsize){
	int i = 0;
	for (i=0;i<FLsize.channel;i++){
		free(FL[i]);
	}
	free(FL);
}
//新建KernalLayer
KernelLayer newKernelLayer(KernelLayerSize KLsize){
	int i = 0;
	KernelLayer KL;
	KL = (KernelLayer)malloc(sizeof(Kernel)*KLsize.depth);
	for (i=0;i<KLsize.depth;i++){
		KL[i] = (Kernel)malloc(sizeof(FType)*KLsize.channel*KLsize.width*KLsize.height);
		memset(KL[i],sizeof(FType)*KLsize.channel*KLsize.width*KLsize.height,0);
	}
	return KL;
}
//删除KernalLayer
void deleteKernelLayer(KernelLayer KL,KernelLayerSize KLsize){
	int i = 0;
	for (i=0;i<KLsize.depth;i++){
		free(KL[i]);
	}
	free(KL);
}
//新建BiasLayer
BiasLayer newBiasLayer(FeaturemapLayerSize FLsize){
	BiasLayer BL = (BiasLayer)malloc(sizeof(FType)*FLsize.channel);
	memset(BL,sizeof(FType)*FLsize.channel,0);
	return BL;
}
//删除BiasLayer
void deleteBiasLayer(BiasLayer BL){
	free(BL);
}

//新建FcBiasLayer
BiasLayer newFcBiasLayer(FCfeatureSize FCsize){
	BiasLayer BL = (BiasLayer)malloc(sizeof(FType)*FCsize.lenth);
	memset(BL,sizeof(FType)*FCsize.lenth,0);
	return BL;
}
//删除FcBiasLayer
void deleteFcBiasLayer(BiasLayer BL){
	free(BL);
}

//新建FCfeature
FCfeature newFCfeature(FCfeatureSize FCFS){
	FCfeature BL = (FCfeature)malloc(sizeof(FType)*FCFS.lenth);
	memset(BL,sizeof(FType)*FCFS.lenth,0);
	return BL;
}
//删除FCfeature
void deleteFCfeature(FCfeature FCF){
	free(FCF);
}

//新建FCKernel
FCKernel newFCKernel(FCKernelSize FCKsize){
	int i = 0;
	FCKernel BL = (FCKernel)malloc(sizeof(SingleKernel)*FCKsize.height);
	memset(BL,sizeof(FType)*FCKsize.height,0);
	for(i=0;i<FCKsize.height;i++){
		BL[i] = (SingleKernel)malloc(sizeof(FType)*FCKsize.width);
	}
	return BL;
}
//删除FCKernel
void deleteFCKernel(FCKernel KL,FCKernelSize KLS){
	int i = 0;
	for (i=0;i<KLS.height;i++){
		free(KL[i]);
	}
	free(KL);
}

//验证是否可以完成一层的卷积
uchar CheckSizeConv(KernelLayerSize sourceKLsize,FeaturemapLayerSize sourceFLsize,FeaturemapLayerSize destFLsize,ConvParam CP){
	if(((sourceFLsize.height - sourceKLsize.height+ 2*CP.padding )/CP.stride + 1) != (destFLsize.height)){return 1;}
	if(((sourceFLsize.width - sourceKLsize.width  + 2*CP.padding)/CP.stride+1) != destFLsize.width){return 2;}
	if(sourceFLsize.channel != sourceKLsize.channel){return 3;}
	if(sourceKLsize.depth != destFLsize.channel){return 4;}
	return 0;
}
//打印一个feaMap
void printMat(FType* mat,int width,int height){
	int i =0,j=0;
	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			printf("%3.4f ",mat[j+i*width]);
		}
		printf("\n");
	}
}
//通过一个 kernel 计算一个 Featuremap
uchar ConvKernelFeaturemap(Kernel sourceK,KernelLayerSize sourceKsize,FeaturemapLayer sourceFL,\
						  FeaturemapLayerSize sourceFLsize,Featuremap destF,FeaturemapLayerSize destFLsize,ConvParam CP){
	int i;
	int singlekernelsize = sourceKsize.height*sourceKsize.width;
	for(i=0;i<sourceKsize.channel;i++){
		Conv2(sourceK+singlekernelsize*i,sourceKsize,sourceFL[i],sourceFLsize,destF,destFLsize,CP);
	}
	return 0;
}
//计算Conv2
uchar Conv2(SingleKernel sourceK,KernelLayerSize sourceKsize,Featuremap sourceF,\
						  FeaturemapLayerSize sourceFLsize,Featuremap destF,FeaturemapLayerSize destFLsize,ConvParam CP){
	int tempw,temph,innerw1,innerh1,innerw2,innerh2;
	int right,left,up,down;//范围都是 left <= x < right
	int rightk,leftk,upk,downk;
	for(temph = 0;temph<destFLsize.height;temph++){
		for(tempw = 0;tempw<destFLsize.width;tempw++){
			left = (tempw*CP.stride-CP.padding);  right=left+sourceKsize.width;  if(left<0){leftk=-left;  left = 0;}else{leftk=0;}
			if(right > sourceFLsize.width){rightk=sourceKsize.width-(right-sourceFLsize.width);  right=sourceFLsize.width;}else{rightk = sourceKsize.width;}
			up = (temph*CP.stride-CP.padding);    down=up+sourceKsize.height;		if(up<0){upk=-up;up = 0;}else{upk=0;}
			if(down > sourceFLsize.height){downk=sourceKsize.height-(down-sourceFLsize.height);  down=sourceFLsize.height;}else{downk = sourceKsize.height;}
			//printf("l:%d r:%d u:%d d:%d",left,right,up,down);
			//printf("  lk:%d rk:%d uk:%d dk:%d \n",leftk,rightk,upk,downk);
			for(innerh1=up,innerh2=upk;innerh1<down;innerh1++,innerh2++){
				for(innerw1=left,innerw2=leftk;innerw1<right;innerw1++,innerw2++){
					destF[temph*destFLsize.width+tempw] += sourceF[innerh1*sourceFLsize.width+innerw1]*sourceK[innerh2*sourceKsize.width+innerw2];
				}
			}
		}
	}
	return 0;
}
//计算一层卷积和
uchar ConvLayer(KernelLayer sourceKL,KernelLayerSize sourceKLsize,FeaturemapLayer sourceFL,\
						  FeaturemapLayerSize sourceFLsize,FeaturemapLayer destFL,FeaturemapLayerSize destFLsize,ConvParam CP){
	int i,j;
	int kernelsize = sourceKLsize.height*sourceKLsize.width*sourceKLsize.channel;
	if(CheckSizeConv(sourceKLsize,sourceFLsize,destFLsize,CP) != 0){ printf("conv error\n");return 1;}
	for(i=0;i<sourceKLsize.depth;i++){
		for(j=0;j<destFLsize.width*destFLsize.height;j++){destFL[i][j]=0;}
		ConvKernelFeaturemap(sourceKL[i],sourceKLsize,sourceFL,sourceFLsize,destFL[i],destFLsize,CP);
	}
	return 0;
}

//为一层FCfeature添加偏置
uchar AddbiasFC(FCfeature FC,FCfeatureSize FCszie,BiasLayer BL){
	int i = 0;
	for(i=0;i<FCszie.lenth;i++){
		FC[i] += BL[i];
	}
}

//一个Featuremap添加偏置
uchar AddbiasFeaturemap(Featuremap FM,FeaturemapLayerSize FMS,Bias B){
	int i = 0;
	for(i=0;i<FMS.height*FMS.width;i++){
		FM[i] += B;
	}
}

//为一层Featuremap添加偏置
uchar AddbiasFeaturemapLayer(FeaturemapLayer FML,FeaturemapLayerSize FMS,BiasLayer BL){
	int i = 0;
	for(i=0;i<FMS.channel;i++){
		AddbiasFeaturemap(FML[i],FMS,BL[i]);
	}
}

//一个Featuremap 取 sigmoid
uchar SigmoidFeaturemap(Featuremap FM,FeaturemapLayerSize FMS){
	int i = 0;
	for(i=0;i<FMS.height*FMS.width;i++){
		FM[i] = (FM[i]>0)?FM[i]:0;
	}
}

//为一层Featuremap 取 sigmoid
uchar SigmoidFeaturemapLayer(FeaturemapLayer FML,FeaturemapLayerSize FMS){
	int i = 0;
	for(i=0;i<FMS.channel;i++){
		SigmoidFeaturemap(FML[i],FMS);
	}
}
//验证是否满足pooling的尺寸
uchar CheckSizePool(FeaturemapLayerSize sourceFLsize,FeaturemapLayerSize destFLsize,PoolParam PP){
	if(destFLsize.width !=  (int)ceil((double)(sourceFLsize.width)/PP.stride)){return 1;}
	if(destFLsize.height != (int)ceil((double)(sourceFLsize.height)/PP.stride)){return 2;}
	if(sourceFLsize.channel != destFLsize.channel){return 3;}
	return 0;
}

//pooling操作
uchar PoolFeaturemapLayer(FeaturemapLayer srcFM,FeaturemapLayerSize sourceFLsize,FeaturemapLayer destFM,FeaturemapLayerSize destFLsize,PoolParam PP){
	int i,j;
	if(CheckSizePool(sourceFLsize,destFLsize,PP) != 0){return 1;}
	for (i=0;i<destFLsize.channel;i++){
		PoolFeaturemap(srcFM[i],sourceFLsize,destFM[i],destFLsize,PP);
	}
	return 0;
}

//pooling 一张 featruemap
uchar PoolFeaturemap(Featuremap srcFM,FeaturemapLayerSize sourceFLsize,Featuremap destFM,FeaturemapLayerSize destFLsize,PoolParam PP){
	int i,j,k,i1,j1,k1,qcoutn;
	FType tempmax;
	if(CheckSizePool(sourceFLsize,destFLsize,PP) != 0){printf("pool error\n");return 1;}
	for (i=0;i<destFLsize.height;i++){
		for(j=0;j<destFLsize.width;j++){
#ifdef POOLMAX
			tempmax = srcFM[j*PP.stride+i*PP.stride*sourceFLsize.width];
			for(i1=0;i1<PP.size;i1++){
				for(j1=0;j1<PP.size;j1++){
					if((j*PP.stride+j1 < sourceFLsize.width) && (i*PP.stride+i1 < sourceFLsize.height)){
							if ((tempmax)<(srcFM[j*PP.stride+j1+(i*PP.stride+i1)*sourceFLsize.width]))
							{
								tempmax=(srcFM[j*PP.stride+j1+(i*PP.stride+i1)*sourceFLsize.width]);
							}
					}
				}
			}
			destFM[j+i*destFLsize.width] = tempmax;
#else
			tempmax = 0;
			qcoutn = 0;
			for(i1=0;i1<PP.size;i1++){
				for(j1=0;j1<PP.size;j1++){
					if((j*PP.stride+j1 < sourceFLsize.width) && (i*PP.stride+i1 < sourceFLsize.height)){
						tempmax += (srcFM[j*PP.stride+j1+(i*PP.stride+i1)*sourceFLsize.width]);
						qcoutn++;
					}
				}
			}
			destFM[j+i*destFLsize.width] = tempmax/qcoutn;
#endif
		}
	}
}


//尺寸验证featuremap 变成 FCfeature
uchar ChecksizeFeatureFC(FeaturemapLayerSize FLS,FCfeatureSize FCFS){
	if(FLS.channel*FLS.height*FLS.width != FCFS.lenth){return 1;}
	return 0;
}


//featuremap 变成 FCfeature
uchar FM2FC(FeaturemapLayer FL,FeaturemapLayerSize FLS,FCfeature FCF,FCfeatureSize FCFS){
	int i,j,k,m,length=0;
	if(ChecksizeFeatureFC(FLS,FCFS) != 0){printf("FM2FC error\n");return 1;}
	for(i=0;i<FLS.channel;i++){
		/*
		for(j=0;j<FLS.height;j++){
			for(k=0;k<FLS.width;k++,length++){
				FCF[length] = FL[i][k+FLS.width*j];
			}
		}
*/
		memcpy(FCF+i*FLS.width*FLS.height,FL[i],FLS.width*FLS.height*sizeof(FType));

	}
	return 0;
}


//尺寸验证Fullyconnect能否执行
uchar ChecksizeFC(FCfeatureSize FLS1,FCKernelSize FCKS,FCfeatureSize FLS2){
	if(FLS1.lenth != FCKS.width){return 1;}
	if(FLS2.lenth != FCKS.height){return 2;}
	return 0;
}

//计算Fully　Connect
uchar FCfeaturefeature(FCfeature FCf1,FCfeatureSize FLS1,FCKernel FCk,FCKernelSize FCKS,FCfeature FCf2,FCfeatureSize FLS2){
	int i,j;
	if(ChecksizeFC(FLS1,FCKS,FLS2) != 0){printf("FCff error\n");return 1;}
	for(i=0;i<FCKS.height;i++){
		FCf2[i] = 0;
		for(j=0;j<FCKS.width;j++){
			FCf2[i] += FCk[i][j]*FCf1[j];
		}
	}
	return 0;
}

//计算第一层的FC
uchar FCmapfeature(FeaturemapLayer FM1,FeaturemapLayerSize FMS1,FCKernel FCk,FCKernelSize FCKS,FCfeature FCf2,FCfeatureSize FLS2){
	int i,j;
	FCfeature FCf1;
	FCfeatureSize FLS1;
	FLS1.lenth = FMS1.channel*FMS1.width*FMS1.height;
	if(ChecksizeFC(FLS1,FCKS,FLS2) != 0){printf("FFmf error\n");return 1;}
	FCf1 = newFCfeature(FLS1);
	if(FM2FC(FM1,FMS1,FCf1,FLS1) != 0){return 2;}
	if(FCfeaturefeature(FCf1,FLS1,FCk,FCKS,FCf2,FLS2) != 0){return 3;};
	deleteFCfeature(FCf1);
	return 0;
}