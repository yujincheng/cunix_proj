#ifndef _NET_H_
#define _NET_H_
#include "cnn.h"

//������������
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

//����һ��ĳߴ�
void SetLayerSize(Layer*,void* src,int* layerPrama12);
void SetLayerSizeData(Layer*,void* src,int* layerPrama12);
void SetLayerSizeConv(Layer*,void* src,int* layerPrama12);
void SetLayerSizeBias(Layer*,void* src,int* layerPrama12);
void SetLayerSizeFcBias(Layer*,void* src,int* layerPrama12);
void SetLayerSizeSigmoid(Layer*,void* src,int* layerPrama12);
void SetLayerSizePool(Layer*,void* src,int* layerPrama12);
void SetLayerSizeFcmf(Layer*,void* src,int* layerPrama12);
void SetLayerSizeFcff(Layer*,void* src,int* layerPrama12);


//Ϊ����������ݵĿռ�
void** mallocNet(Net* net);

//Ϊ����ṹ����ռ�
void mallocNetStruct(Net* net);

//Ϊһ��ṹ����ռ�
void** mallocLayer(Layer* layer);

//��ʼ������ṹ
void InitialNetStruct(Net* net,int m,LayerType types[],int** tyu);

//Ϊ����ṹ����ռ�
void** InitialNetData(Net* net);


//Ϊȷ����һ������ָ���ռ�
void IndexLayer(Layer* layer,void** layerSpace3);

//Ϊ��������ָ���ռ�
void IndexNet(Net* net,void** layerSpace3);

//����һ�������ǰ��
void ForwardLyer(Layer* layer);

//ǰ���������
void ForwardNet(Net* net);

//���ļ��а���һ������
void SetDataLayer(Layer* layer,char* kernelname);

//���ļ��а�����������
void SetDataNet(Net* net,char** kernelname);
#endif