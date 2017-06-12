#include "net.h"



//设置一层的尺寸
void SetLayerSize(Layer* layer,void* src,int* sizeinfo){
	switch (layer->type)
	{		
	case data:
		SetLayerSizeData(layer,src,sizeinfo);break;
	case conv:
		SetLayerSizeConv(layer,src,sizeinfo);break;
	case bias:
		SetLayerSizeBias(layer,src,sizeinfo);break;
	case fcbias:
		SetLayerSizeFcBias(layer,src,sizeinfo);break;
	case sigmoid:
		SetLayerSizeSigmoid(layer,src,sizeinfo);break;
	case pool:
		SetLayerSizePool(layer,src,sizeinfo);break;
	case fcmf:
		SetLayerSizeFcmf(layer,src,sizeinfo);break;
	case fcff:
		SetLayerSizeFcff(layer,src,sizeinfo);break;
	default:
		break;
	}	
}

void SetLayerSizeData(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.dataLayer.srcFMLSize.width = sizeinfo[0];
	layer->layerprama.dataLayer.srcFMLSize.height = sizeinfo[1];
	layer->layerprama.dataLayer.srcFMLSize.channel = sizeinfo[2];
	layer->layerprama.dataLayer.datasrc = src;
}

void SetLayerSizeConv(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.convLayer.srcFMLSize.width = sizeinfo[0];
	layer->layerprama.convLayer.srcFMLSize.height = sizeinfo[1];
	layer->layerprama.convLayer.srcFMLSize.channel = sizeinfo[2];
	layer->layerprama.convLayer.KLSize.width = sizeinfo[3];
	layer->layerprama.convLayer.KLSize.height = sizeinfo[4];
	layer->layerprama.convLayer.KLSize.channel = sizeinfo[5];
	layer->layerprama.convLayer.KLSize.depth = sizeinfo[6];
	layer->layerprama.convLayer.destFMLSize.width = sizeinfo[7];
	layer->layerprama.convLayer.destFMLSize.height = sizeinfo[8];
	layer->layerprama.convLayer.destFMLSize.channel = sizeinfo[9];
	layer->layerprama.convLayer.CP.padding = sizeinfo[10];
	layer->layerprama.convLayer.CP.stride = sizeinfo[11];
}

void SetLayerSizeBias(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.biasLayer.srcFMLSize.width = sizeinfo[0];
	layer->layerprama.biasLayer.srcFMLSize.height = sizeinfo[1];
	layer->layerprama.biasLayer.srcFMLSize.channel = sizeinfo[2];
}

void SetLayerSizeFcBias(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.fcbiasLayer.srcFCSize.lenth = sizeinfo[0];
}

void SetLayerSizeSigmoid(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.sigmoidLayer.srcFMLSize.width = sizeinfo[0];
	layer->layerprama.sigmoidLayer.srcFMLSize.height = sizeinfo[1];
	layer->layerprama.sigmoidLayer.srcFMLSize.channel = sizeinfo[2];
}

void SetLayerSizePool(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.poolLayer.srcFMLSize.width = sizeinfo[0];
	layer->layerprama.poolLayer.srcFMLSize.height = sizeinfo[1];
	layer->layerprama.poolLayer.srcFMLSize.channel = sizeinfo[2];
	layer->layerprama.poolLayer.destFMLSize.width = sizeinfo[7];
	layer->layerprama.poolLayer.destFMLSize.height = sizeinfo[8];
	layer->layerprama.poolLayer.destFMLSize.channel = sizeinfo[9];
	layer->layerprama.poolLayer.PP.size = sizeinfo[10];
	layer->layerprama.poolLayer.PP.stride = sizeinfo[11];
}

void SetLayerSizeFcmf(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.fcmfLayer.srcFMLSize.width = sizeinfo[0];
	layer->layerprama.fcmfLayer.srcFMLSize.height = sizeinfo[1];
	layer->layerprama.fcmfLayer.srcFMLSize.channel = sizeinfo[2];
	layer->layerprama.fcmfLayer.KSize.width = sizeinfo[3];
	layer->layerprama.fcmfLayer.KSize.height = sizeinfo[4];
	layer->layerprama.fcmfLayer.destFCFSize.lenth = sizeinfo[7];
}

void SetLayerSizeFcff(Layer* layer,void* src,int* sizeinfo){
	layer->layerprama.fcffLayer.srcFCFSize.lenth = sizeinfo[0];
	layer->layerprama.fcffLayer.KSize.width = sizeinfo[3];
	layer->layerprama.fcffLayer.KSize.height = sizeinfo[4];
	layer->layerprama.fcffLayer.destFCFSize.lenth = sizeinfo[7];
}

//为网络配置层数
void SetLayerCount(Net* net,int m){
	net->LayerCount = m;
}

//为网络分配初始的空间
void mallocNetStruct(Net* net){
	net->layers = (Layer*)malloc(sizeof(Layer)*net->LayerCount);

}

//返回一个三维向量，每一维存放一个地址,只开辟 kernel 和 output的空间
void** mallocLayerData(Layer*layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = newFeaturemapLayer(layer->layerprama.dataLayer.srcFMLSize);
	layerspace[1] = layerspace[0];
	layerspace[2] = layerspace[0];
	return layerspace;
}

void** mallocLayerConv(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = newKernelLayer(layer->layerprama.convLayer.KLSize);
	layerspace[2] = newFeaturemapLayer(layer->layerprama.convLayer.destFMLSize);
	return layerspace;
}

void** mallocLayerBias(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = newBiasLayer(layer->layerprama.biasLayer.srcFMLSize);
	layerspace[2] = NULL;
	return layerspace;
}

void** mallocLayerFcBias(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = newFcBiasLayer(layer->layerprama.fcbiasLayer.srcFCSize);
	layerspace[2] = NULL;
	return layerspace;
}

void** mallocLayerSigmoid(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = NULL;
	layerspace[2] = NULL;
	return layerspace;
}

void** mallocLayerPool(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = NULL;
	layerspace[2] = newFeaturemapLayer(layer->layerprama.poolLayer.destFMLSize);
	return layerspace;
}

void** mallocLayerFcmf(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = newFCKernel(layer->layerprama.fcmfLayer.KSize);
	layerspace[2] = newFCfeature(layer->layerprama.fcmfLayer.destFCFSize);
	return layerspace;
}

void** mallocLayerFcff(Layer* layer){
	void** layerspace = (void**)malloc(sizeof(void*)*3);
	layerspace[0] = NULL;
	layerspace[1] = newFCKernel(layer->layerprama.fcmfLayer.KSize);
	layerspace[2] = newFCfeature(layer->layerprama.fcmfLayer.destFCFSize);
	return layerspace;
}

//为一层结构分配空间
void** mallocLayer(Layer* layer){
	void** layerspace;
	switch (layer->type)
	{		
	case data:
		layerspace = mallocLayerData(layer);break;
	case conv:
		layerspace = mallocLayerConv(layer);break;
	case bias:
		layerspace = mallocLayerBias(layer);break;
	case fcbias:
		layerspace = mallocLayerFcBias(layer);break;
	case sigmoid:
		layerspace = mallocLayerSigmoid(layer);break;
	case pool:
		layerspace = mallocLayerPool(layer);break;
	case fcmf:
		layerspace = mallocLayerFcmf(layer);break;
	case fcff:
		layerspace = mallocLayerFcff(layer);break;
	default:
		break;
	}
	return layerspace;
}

//为整个网络分配空间
void** mallocNet(Net* net){
	int i;
	void** netspace = (void**)malloc(sizeof(void*)*net->LayerCount);
	for(i=0;i<(net->LayerCount);i++){
		switch ((net->layers[i]).type)
		{		
		case data:
			netspace[i] = mallocLayer(&(net->layers[i]));break;
		case conv:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			break;
		case bias:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			((void**)netspace[i])[2] = ((void**)netspace[i-1])[2];
			break;
		case fcbias:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			((void**)netspace[i])[2] = ((void**)netspace[i-1])[2];
			break;
		case sigmoid:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			((void**)netspace[i])[2] = ((void**)netspace[i-1])[2];
			break;
		case pool:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			break;
		case fcmf:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			break;
		case fcff:
			netspace[i] = mallocLayer(&(net->layers[i]));
			((void**)netspace[i])[0] = ((void**)netspace[i-1])[2];
			break;
		default:
			break;
		}
	}
	return netspace;
}

//为网络各层定义类型
void SetLayerType(Net* net,LayerType types[]){
	int i = 0;
	for (i=0;i<net->LayerCount;i++){
		net->layers[i].type = types[i];
	}
}

//初始化网络结构
void InitialNetStruct(Net* net,int m,LayerType types[],int** tyu){
	int i =0;
	SetLayerCount(net,m);
	mallocNetStruct(net);
	SetLayerType(net,types);
	for(i=0;i<net->LayerCount;i++){
		SetLayerSize(&(net->layers[i]),NULL,tyu[i]);
	}
}

//为网络结构分配空间
void** InitialNetData(Net* net){
	void **Space;
	Space = mallocNet(net);
	IndexNet(net,Space);
	return Space;
}

//为确定的一层网络指定空间 src kernal dest

void IndexLayerData(Layer* layer,void** layerSpace3){
	layer->layerprama.dataLayer.datasource = (FeaturemapLayer)layerSpace3[0];
}

void IndexLayerConv(Layer* layer,void** layerSpace3){
	layer->layerprama.convLayer.srcFM = (FeaturemapLayer)layerSpace3[0];
	layer->layerprama.convLayer.KLsource = (KernelLayer)layerSpace3[1];
	layer->layerprama.convLayer.destFM = (KernelLayer)layerSpace3[2];
}

void IndexLayerBias(Layer* layer,void** layerSpace3){
	layer->layerprama.biasLayer.srcFM = (FeaturemapLayer)layerSpace3[0];
	layer->layerprama.biasLayer.biassource = (BiasLayer)layerSpace3[1];
}

void IndexLayerFcBias(Layer* layer,void** layerSpace3){
	layer->layerprama.fcbiasLayer.srcFC = (FCfeature)layerSpace3[0];
	layer->layerprama.fcbiasLayer.biassource = (BiasLayer)layerSpace3[1];
}

void IndexLayerSigmoid(Layer* layer,void** layerSpace3){
	layer->layerprama.sigmoidLayer.srcFM = (FeaturemapLayer)layerSpace3[0];
}

void IndexLayerPool(Layer* layer,void** layerSpace3){
	layer->layerprama.poolLayer.srcFM = (FeaturemapLayer)layerSpace3[0];
	layer->layerprama.poolLayer.destFM = (FeaturemapLayer)layerSpace3[2];
}

void IndexLayerFcmf(Layer* layer,void** layerSpace3){
	layer->layerprama.fcmfLayer.srcFM = (FeaturemapLayer)layerSpace3[0];
	layer->layerprama.fcmfLayer.KNEel = (FCKernel)layerSpace3[1];
	layer->layerprama.fcmfLayer.destFC = (FCfeature)layerSpace3[2];
}

void IndexLayerFcff(Layer* layer,void** layerSpace3){
	layer->layerprama.fcffLayer.srcFC = (FCfeature)layerSpace3[0];
	layer->layerprama.fcffLayer.KNEel = (FCKernel)layerSpace3[1];
	layer->layerprama.fcffLayer.destFC = (FCfeature)layerSpace3[2];
}

void IndexLayer(Layer* layer,void** layerSpace3){
	switch (layer->type)
	{		
	case data:
		IndexLayerData(layer,layerSpace3);break;
	case conv:
		IndexLayerConv(layer,layerSpace3);break;
	case bias:
		IndexLayerBias(layer,layerSpace3);break;
	case fcbias:
		IndexLayerFcBias(layer,layerSpace3);break;
	case sigmoid:
		IndexLayerSigmoid(layer,layerSpace3);break;
	case pool:
		IndexLayerPool(layer,layerSpace3);break;
	case fcmf:
		IndexLayerFcmf(layer,layerSpace3);break;
	case fcff:
		IndexLayerFcff(layer,layerSpace3);break;
	default:
		break;
	}
}

//从文件中搬数据
void SetDataLayerData(Layer* layer,char* kernelname){
	
}



void SetDataLayerConv(Layer* layer,char* kernelname){
	FILE *infp;
	int i,j;
	int sizestrid = layer->layerprama.convLayer.KLSize.height *layer->layerprama.convLayer.KLSize.width*layer->layerprama.convLayer.KLSize.channel;
	infp=fopen(kernelname,"rb");
	for(i = 0;i<layer->layerprama.convLayer.KLSize.depth;i++){
		fread( layer->layerprama.convLayer.KLsource[i], sizeof(FType), sizestrid, infp );
	}
	fclose(infp);
}


void SetDataLayerBias(Layer* layer,char* kernelname){
	FILE *infp;
	int i,j;
	int sizestrid = layer->layerprama.biasLayer.srcFMLSize.channel ;
	infp=fopen(kernelname,"rb");
	fread( layer->layerprama.biasLayer.biassource, sizeof(FType), sizestrid, infp );
	fclose(infp);
}

void SetDataLayerFcBias(Layer* layer,char* kernelname){
	FILE *infp;
	int i,j;
	int sizestrid = layer->layerprama.fcbiasLayer.srcFCSize.lenth ;
	infp=fopen(kernelname,"rb");
	fread( layer->layerprama.fcbiasLayer.biassource, sizeof(FType), sizestrid, infp );
	fclose(infp);
}

void SetDataLayerSigmoid(Layer* layer,char* kernelname){
	
}

void SetDataLayerPool(Layer* layer,char* kernelname){

}

void SetDataLayerFcmf(Layer* layer,char* kernelname){
	FILE *infp;
	int i,j;
	int sizestrid = layer->layerprama.fcmfLayer.KSize.width ;
	infp=fopen(kernelname,"rb");
	for(i=0;i<layer->layerprama.fcmfLayer.KSize.height;i++){
		fread( layer->layerprama.fcmfLayer.KNEel[i], sizeof(FType), sizestrid, infp );
	}
	fclose(infp);
}

void SetDataLayerFcff(Layer* layer,char* kernelname){
	FILE *infp;
	int i,j;
	int sizestrid = layer->layerprama.fcffLayer.KSize.width ;
	infp=fopen(kernelname,"rb");
	for(i=0;i<layer->layerprama.fcffLayer.KSize.height;i++){
		fread( layer->layerprama.fcffLayer.KNEel[i], sizeof(FType), sizestrid, infp );
	}
	fclose(infp);
}

//从文件中搬运一层数据
void SetDataLayer(Layer* layer,char* kernelname){
	switch (layer->type)
	{		
	case data:
		SetDataLayerData(layer,kernelname);break;
	case conv:
		SetDataLayerConv(layer,kernelname);break;
	case bias:
		SetDataLayerBias(layer,kernelname);break;
	case fcbias:
		SetDataLayerFcBias(layer,kernelname);break;
	case sigmoid:
		SetDataLayerSigmoid(layer,kernelname);break;
	case pool:
		SetDataLayerPool(layer,kernelname);break;
	case fcmf:
		SetDataLayerFcmf(layer,kernelname);break;
	case fcff:
		SetDataLayerFcff(layer,kernelname);break;
	default:
		break;
	}
}

//从文件中搬运整个网络
void SetDataNet(Net* net,char** kernelname){
	int i;
	for(i=0;i<net->LayerCount;i++){
		SetDataLayer(&net->layers[i],kernelname[i]);
	}
}


//为整个网络指定空间
void IndexNet(Net* net,void** layerSpace3){
	int i = 0;
	for(i=0;i<net->LayerCount;i++){
		IndexLayer(&(net->layers[i]),(void**)layerSpace3[i]);
	}
}

void ForwardLyerData(Layer* layer){

}

void ForwardLyerConv(Layer* layer){
	ConvLayer(layer->layerprama.convLayer.KLsource,\
		layer->layerprama.convLayer.KLSize,\
		layer->layerprama.convLayer.srcFM,\
		layer->layerprama.convLayer.srcFMLSize,\
		layer->layerprama.convLayer.destFM,\
		layer->layerprama.convLayer.destFMLSize,\
		layer->layerprama.convLayer.CP);
}

void ForwardLyerBias(Layer* layer){
	AddbiasFeaturemapLayer( \
		layer->layerprama.biasLayer.srcFM,\
		layer->layerprama.biasLayer.srcFMLSize,\
		layer->layerprama.biasLayer.biassource);
}

void ForwardLyerFcBias(Layer* layer){
	AddbiasFC( \
		layer->layerprama.fcbiasLayer.srcFC,\
		layer->layerprama.fcbiasLayer.srcFCSize,\
		layer->layerprama.fcbiasLayer.biassource);
}

void ForwardLyerSigmoid(Layer* layer){
	SigmoidFeaturemapLayer( \
		layer->layerprama.sigmoidLayer.srcFM,\
		layer->layerprama.sigmoidLayer.srcFMLSize);
}
void ForwardLyerPool(Layer* layer){
	PoolFeaturemapLayer(layer->layerprama.poolLayer.srcFM,\
		layer->layerprama.poolLayer.srcFMLSize,\
		layer->layerprama.poolLayer.destFM,\
		layer->layerprama.poolLayer.destFMLSize,\
		layer->layerprama.poolLayer.PP);


}
void ForwardLyerFcmf(Layer* layer){
	FCmapfeature(layer->layerprama.fcmfLayer.srcFM,\
		layer->layerprama.fcmfLayer.srcFMLSize,\
		layer->layerprama.fcmfLayer.KNEel,\
		layer->layerprama.fcmfLayer.KSize,\
		layer->layerprama.fcmfLayer.destFC,\
		layer->layerprama.fcmfLayer.destFCFSize);
}
void ForwardLyerFcff(Layer* layer){
	FCfeaturefeature(layer->layerprama.fcffLayer.srcFC,\
		layer->layerprama.fcffLayer.srcFCFSize,\
		layer->layerprama.fcffLayer.KNEel,\
		layer->layerprama.fcffLayer.KSize,\
		layer->layerprama.fcffLayer.destFC,\
		layer->layerprama.fcffLayer.destFCFSize);
}

//计算一层网络的前向
void ForwardLyer(Layer* layer){
	switch (layer->type)
	{		
	case data:
		ForwardLyerData(layer);break;
	case conv:
		ForwardLyerConv(layer);break;
	case bias:
		ForwardLyerBias(layer);break;
	case fcbias:
		ForwardLyerFcBias(layer);break;
	case sigmoid:
		ForwardLyerSigmoid(layer);break;
	case pool:
		ForwardLyerPool(layer);break;
	case fcmf:
		ForwardLyerFcmf(layer);break;
	case fcff:
		ForwardLyerFcff(layer);break;
	default:
		break;
	}	

}

void SetDateFromMemory(Net* net,FeaturemapLayer oridata){
	int incha = net->layers[0].layerprama.dataLayer.srcFMLSize.channel;
	int inlen = net->layers[0].layerprama.dataLayer.srcFMLSize.width*net->layers[0].layerprama.dataLayer.srcFMLSize.height;
	int i;
	for (i=0;i<incha;i++){
		memcpy(net->layers[0].layerprama.dataLayer.datasource[i],oridata[i],sizeof(FType)*inlen);
	}
}


//前向整个络的
void ForwardNet(Net* net){
	int i = 0;
	for(i=0;i<net->LayerCount;i++){
		ForwardLyer(&(net->layers[i]));
	}
}