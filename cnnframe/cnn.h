#ifndef _CNN_H_
#define _CNN_H_

#include <malloc.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

typedef unsigned char uchar;
//定义featuremap的数据结构，由于featuremap都是二维结构，假设图片不那么大的时候我们采用一个一维数组存一张featuremap，一个二维数组存一层的featuremap
//定义一个元素的数据类型。
typedef float FType;

//#define POOLMAX

//一个featuremap有两个尺寸参数 width height
typedef FType* Featuremap;


//一张 n*n 的kenel
typedef FType* SingleKernel;
typedef SingleKernel* FCKernel;

//一个kenel是 channel 个 Sigel kenel 但仍然连续存储
typedef FType* Kernel;

//一层featuremap有channel个Featuremap
typedef Featuremap* FeaturemapLayer;

//一层kenel有channel个kenel
typedef Kernel* KernelLayer;

//定义一层kenel的尺寸
typedef struct
{
	int height;//rowsize
	int width;//colsize
	int channel;//input channel
	int depth;//out channel
}KernelLayerSize;

//定义一层featuremap的尺寸
typedef struct
{
	int height;//rowsize
	int width;//colsize
	int channel;//input channel
}FeaturemapLayerSize;
//定义一个偏置
typedef FType Bias;
//定义一层偏置
typedef FType* BiasLayer;
//FCfeature
typedef FType* FCfeature;
//FCfeaturesize
typedef struct
{
	int lenth;
}FCfeatureSize;
//定义FCkernel尺寸
typedef struct{
	int width;
	int height;
}FCKernelSize;


//定义卷积的选项
typedef struct{
	int padding;
	int stride;
}ConvParam;
//定义pool的选项
typedef struct
{
	int stride;
	int size;
}PoolParam;

//新建FeaturemapLayer
FeaturemapLayer newFeaturemapLayer(FeaturemapLayerSize FLsize);
//删除FeaturemapLayer
void deleteFeaturemapLayer(FeaturemapLayer FL,FeaturemapLayerSize FLsize);

//新建KernalLayer
KernelLayer newKernelLayer(KernelLayerSize KLsize);
//删除KernalLayer
void deleteKernelLayer(KernelLayer KL,KernelLayerSize KLsize);


//新建BiasLayer
BiasLayer newBiasLayer(FeaturemapLayerSize FLsize);
//删除BiasLayer
void deleteBiasLayer(BiasLayer BL);

//新建FcBiasLayer
BiasLayer newFcBiasLayer(FCfeatureSize FLsize);
//删除FcBiasLayer
void deleteFcBiasLayer(BiasLayer BL);


//新建FCfeature
FCfeature newFCfeature(FCfeatureSize FCFS);
//删除FCfeature
void deleteFCfeature(FCfeature FLsize);

//新建FCKernel
FCKernel newFCKernel(FCKernelSize FCKsize);
//删除FCKernel
void deleteFCKernel(FCKernel KL,FCKernelSize KLS);

//打印一个feaMap
void printMat(FType* mat,int width,int height);

//验证是否可以完成一层的卷积
uchar CheckSizeConv(KernelLayerSize sourceKLsize,FeaturemapLayerSize sourceFLsize,FeaturemapLayerSize destFLsize,ConvParam CP);

//计算一层卷积和
uchar ConvLayer(KernelLayer sourceKL,KernelLayerSize sourceKLsize,FeaturemapLayer sourceFL,\
						  FeaturemapLayerSize sourceFLsize,FeaturemapLayer destFL,FeaturemapLayerSize destFLsize,ConvParam CP);

//通过一个 kernel 计算一个 Featuremap
uchar ConvKernelFeaturemap(Kernel sourceK,KernelLayerSize sourceKsize,FeaturemapLayer sourceFL,\
						  FeaturemapLayerSize sourceFLsize,Featuremap destF,FeaturemapLayerSize destFLsize,ConvParam CP);
//计算二维卷积
uchar Conv2(SingleKernel sourceK,KernelLayerSize sourceKsize,Featuremap sourceF,\
						  FeaturemapLayerSize sourceFLsize,Featuremap destF,FeaturemapLayerSize destFLsize,ConvParam CP);

//一个Featuremap添加偏置
uchar AddbiasFeaturemap(Featuremap FM,FeaturemapLayerSize FMS,Bias B);

//为一层FCfeature添加偏置
uchar AddbiasFC(FCfeature FC,FCfeatureSize FCszie,BiasLayer BL);

//为一层Featuremap添加偏置
uchar AddbiasFeaturemapLayer(FeaturemapLayer FML,FeaturemapLayerSize FMS,BiasLayer BL);
//一个Featuremap 取 sigmoid
uchar SigmoidFeaturemap(Featuremap FM,FeaturemapLayerSize FMS);
//为一层Featuremap 取 sigmoid
uchar SigmoidFeaturemapLayer(FeaturemapLayer FML,FeaturemapLayerSize FMS);

//验证是否满足pooling的
uchar CheckSizePool(FeaturemapLayerSize sourceFLsize,FeaturemapLayerSize destFLsize,PoolParam PP);

//pooling操作
uchar PoolFeaturemapLayer(FeaturemapLayer srcFM,FeaturemapLayerSize sourceFLsize,FeaturemapLayer destFM,FeaturemapLayerSize destFLsize,PoolParam PP);

//pooling 一张 featruemap
uchar PoolFeaturemap(Featuremap srcFM,FeaturemapLayerSize sourceFLsize,Featuremap destFM,FeaturemapLayerSize destFLsize,PoolParam PP);


//尺寸验证featuremap 变成 FCfeature
uchar ChecksizeFeatureFC(FeaturemapLayerSize FLS,FCfeatureSize FCFS);

//featuremap 变成 FCfeature
uchar FM2FC(FeaturemapLayer FL,FeaturemapLayerSize FLS,FCfeature FCF,FCfeatureSize FCFS);

//尺寸验证Fullyconnect能否执行
uchar ChecksizeFC(FCfeatureSize FLS1,FCKernelSize FCKS,FCfeatureSize FCFS2);

//计算Fully　Connect
uchar FCfeaturefeature(FCfeature FCf1,FCfeatureSize FLS1,FCKernel FCk,FCKernelSize FCKS,FCfeature FCf2,FCfeatureSize FLS2);

//计算第一层FC
uchar FCmapfeature(FeaturemapLayer FM1,FeaturemapLayerSize FMS1,FCKernel FCk,FCKernelSize FCKS,FCfeature FCf2,FCfeatureSize FLS2);


#endif