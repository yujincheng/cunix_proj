#ifndef _CNN_H_
#define _CNN_H_

#include <malloc.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

typedef unsigned char uchar;
//����featuremap�����ݽṹ������featuremap���Ƕ�ά�ṹ������ͼƬ����ô���ʱ�����ǲ���һ��һά�����һ��featuremap��һ����ά�����һ���featuremap
//����һ��Ԫ�ص��������͡�
typedef float FType;

//#define POOLMAX

//һ��featuremap�������ߴ���� width height
typedef FType* Featuremap;


//һ�� n*n ��kenel
typedef FType* SingleKernel;
typedef SingleKernel* FCKernel;

//һ��kenel�� channel �� Sigel kenel ����Ȼ�����洢
typedef FType* Kernel;

//һ��featuremap��channel��Featuremap
typedef Featuremap* FeaturemapLayer;

//һ��kenel��channel��kenel
typedef Kernel* KernelLayer;

//����һ��kenel�ĳߴ�
typedef struct
{
	int height;//rowsize
	int width;//colsize
	int channel;//input channel
	int depth;//out channel
}KernelLayerSize;

//����һ��featuremap�ĳߴ�
typedef struct
{
	int height;//rowsize
	int width;//colsize
	int channel;//input channel
}FeaturemapLayerSize;
//����һ��ƫ��
typedef FType Bias;
//����һ��ƫ��
typedef FType* BiasLayer;
//FCfeature
typedef FType* FCfeature;
//FCfeaturesize
typedef struct
{
	int lenth;
}FCfeatureSize;
//����FCkernel�ߴ�
typedef struct{
	int width;
	int height;
}FCKernelSize;


//��������ѡ��
typedef struct{
	int padding;
	int stride;
}ConvParam;
//����pool��ѡ��
typedef struct
{
	int stride;
	int size;
}PoolParam;

//�½�FeaturemapLayer
FeaturemapLayer newFeaturemapLayer(FeaturemapLayerSize FLsize);
//ɾ��FeaturemapLayer
void deleteFeaturemapLayer(FeaturemapLayer FL,FeaturemapLayerSize FLsize);

//�½�KernalLayer
KernelLayer newKernelLayer(KernelLayerSize KLsize);
//ɾ��KernalLayer
void deleteKernelLayer(KernelLayer KL,KernelLayerSize KLsize);


//�½�BiasLayer
BiasLayer newBiasLayer(FeaturemapLayerSize FLsize);
//ɾ��BiasLayer
void deleteBiasLayer(BiasLayer BL);

//�½�FcBiasLayer
BiasLayer newFcBiasLayer(FCfeatureSize FLsize);
//ɾ��FcBiasLayer
void deleteFcBiasLayer(BiasLayer BL);


//�½�FCfeature
FCfeature newFCfeature(FCfeatureSize FCFS);
//ɾ��FCfeature
void deleteFCfeature(FCfeature FLsize);

//�½�FCKernel
FCKernel newFCKernel(FCKernelSize FCKsize);
//ɾ��FCKernel
void deleteFCKernel(FCKernel KL,FCKernelSize KLS);

//��ӡһ��feaMap
void printMat(FType* mat,int width,int height);

//��֤�Ƿ�������һ��ľ��
uchar CheckSizeConv(KernelLayerSize sourceKLsize,FeaturemapLayerSize sourceFLsize,FeaturemapLayerSize destFLsize,ConvParam CP);

//����һ������
uchar ConvLayer(KernelLayer sourceKL,KernelLayerSize sourceKLsize,FeaturemapLayer sourceFL,\
						  FeaturemapLayerSize sourceFLsize,FeaturemapLayer destFL,FeaturemapLayerSize destFLsize,ConvParam CP);

//ͨ��һ�� kernel ����һ�� Featuremap
uchar ConvKernelFeaturemap(Kernel sourceK,KernelLayerSize sourceKsize,FeaturemapLayer sourceFL,\
						  FeaturemapLayerSize sourceFLsize,Featuremap destF,FeaturemapLayerSize destFLsize,ConvParam CP);
//�����ά���
uchar Conv2(SingleKernel sourceK,KernelLayerSize sourceKsize,Featuremap sourceF,\
						  FeaturemapLayerSize sourceFLsize,Featuremap destF,FeaturemapLayerSize destFLsize,ConvParam CP);

//һ��Featuremap���ƫ��
uchar AddbiasFeaturemap(Featuremap FM,FeaturemapLayerSize FMS,Bias B);

//Ϊһ��FCfeature���ƫ��
uchar AddbiasFC(FCfeature FC,FCfeatureSize FCszie,BiasLayer BL);

//Ϊһ��Featuremap���ƫ��
uchar AddbiasFeaturemapLayer(FeaturemapLayer FML,FeaturemapLayerSize FMS,BiasLayer BL);
//һ��Featuremap ȡ sigmoid
uchar SigmoidFeaturemap(Featuremap FM,FeaturemapLayerSize FMS);
//Ϊһ��Featuremap ȡ sigmoid
uchar SigmoidFeaturemapLayer(FeaturemapLayer FML,FeaturemapLayerSize FMS);

//��֤�Ƿ�����pooling��
uchar CheckSizePool(FeaturemapLayerSize sourceFLsize,FeaturemapLayerSize destFLsize,PoolParam PP);

//pooling����
uchar PoolFeaturemapLayer(FeaturemapLayer srcFM,FeaturemapLayerSize sourceFLsize,FeaturemapLayer destFM,FeaturemapLayerSize destFLsize,PoolParam PP);

//pooling һ�� featruemap
uchar PoolFeaturemap(Featuremap srcFM,FeaturemapLayerSize sourceFLsize,Featuremap destFM,FeaturemapLayerSize destFLsize,PoolParam PP);


//�ߴ���֤featuremap ��� FCfeature
uchar ChecksizeFeatureFC(FeaturemapLayerSize FLS,FCfeatureSize FCFS);

//featuremap ��� FCfeature
uchar FM2FC(FeaturemapLayer FL,FeaturemapLayerSize FLS,FCfeature FCF,FCfeatureSize FCFS);

//�ߴ���֤Fullyconnect�ܷ�ִ��
uchar ChecksizeFC(FCfeatureSize FLS1,FCKernelSize FCKS,FCfeatureSize FCFS2);

//����Fully��Connect
uchar FCfeaturefeature(FCfeature FCf1,FCfeatureSize FLS1,FCKernel FCk,FCKernelSize FCKS,FCfeature FCf2,FCfeatureSize FLS2);

//�����һ��FC
uchar FCmapfeature(FeaturemapLayer FM1,FeaturemapLayerSize FMS1,FCKernel FCk,FCKernelSize FCKS,FCfeature FCf2,FCfeatureSize FLS2);


#endif