#ifndef _NET_H_
#define _NET_H_
#include "cnn.h"

//定义网络类型
typedef enum {data=1,conv,bias,sigmoid,pool,fcmf,fcff,fcbias}LayerType;

typedef struct
{
	FeaturemapLayerSize srcFMLSize;
	FeaturemapLayer datasource;
	void* datasrc;
	void* datadest;
}DataLayer;

typedef struct
{
	FeaturemapLayerSize srcFMLSize;
	FeaturemapLayerSize destFMLSize;
	KernelLayerSize KLSize;
	KernelLayer KLsource;
	FeaturemapLayer srcFM;
	FeaturemapLayer destFM;
	ConvParam CP;
	void* datasrc;
	void* datadest;
}ComConvLayer;

typedef struct
{
	FeaturemapLayerSize srcFMLSize;
	FeaturemapLayer srcFM;
	BiasLayer biassource;
	void* datasrc;
	void* datadest;
}AddbiasLayer;

typedef struct
{
	FCfeatureSize srcFCSize;
	FCfeature srcFC;
	BiasLayer biassource;
	void* datasrc;
	void* datadest;
}AddfcbiasLayer;

typedef struct
{
	FeaturemapLayerSize srcFMLSize;
	FeaturemapLayer srcFM;
	void* datasrc;
	void* datadest;
}SigmoidLayer;

typedef struct
{
	FeaturemapLayerSize srcFMLSize;
	FeaturemapLayerSize destFMLSize;
	FeaturemapLayer srcFM;
	FeaturemapLayer destFM;
	PoolParam PP;
	void* datasrc;
	void* datadest;
}PoolLayer;

typedef struct
{
	FeaturemapLayerSize srcFMLSize;
	FCfeatureSize destFCFSize;
	FeaturemapLayer srcFM;
	FCfeature destFC;
	FCKernelSize KSize;
	FCKernel KNEel;
	void* datasrc;
	void* datadest;
}FcmfLayer;

typedef struct
{
	FCfeatureSize srcFCFSize;
	FCfeatureSize destFCFSize;
	FCfeature destFC;
	FCfeature srcFC;
	FCKernelSize KSize;
	FCKernel KNEel;
	void* datasrc;
	void* datadest;
}FcffLayer;

typedef union
{
	DataLayer dataLayer;
	ComConvLayer convLayer;
	AddbiasLayer biasLayer;
	AddfcbiasLayer fcbiasLayer;
	SigmoidLayer sigmoidLayer;
	PoolLayer poolLayer;
	FcmfLayer fcmfLayer;
	FcffLayer fcffLayer;
}LayerPrama;

typedef struct
{
	LayerType type;
	LayerPrama layerprama;
}Layer;

typedef struct
{
	int LayerCount;
	Layer* layers;
}Net;

//设置一层的尺寸
void SetLayerSize(Layer*,void* src,int* layerPrama12);
void SetLayerSizeData(Layer*,void* src,int* layerPrama12);
void SetLayerSizeConv(Layer*,void* src,int* layerPrama12);
void SetLayerSizeBias(Layer*,void* src,int* layerPrama12);
void SetLayerSizeFcBias(Layer*,void* src,int* layerPrama12);
void SetLayerSizeSigmoid(Layer*,void* src,int* layerPrama12);
void SetLayerSizePool(Layer*,void* src,int* layerPrama12);
void SetLayerSizeFcmf(Layer*,void* src,int* layerPrama12);
void SetLayerSizeFcff(Layer*,void* src,int* layerPrama12);


//为网络分配数据的空间
void** mallocNet(Net* net);

//为网络结构分配空间
void mallocNetStruct(Net* net);

//为一层结构分配空间
void** mallocLayer(Layer* layer);

//初始化网络结构
void InitialNetStruct(Net* net,int m,LayerType types[],int** tyu);

//为网络结构分配空间
void** InitialNetData(Net* net);


//为确定的一层网络指定空间
void IndexLayer(Layer* layer,void** layerSpace3);

//为整个网络指定空间
void IndexNet(Net* net,void** layerSpace3);

//计算一层网络的前向
void ForwardLyer(Layer* layer);

//前向整个络的
void ForwardNet(Net* net);

//从文件中搬运一层数据
void SetDataLayer(Layer* layer,char* kernelname);

//从文件中搬运整个网络
void SetDataNet(Net* net,char** kernelname);
#endif