#pragma once
#include <torch/torch.h>
#include <iostream>
#include <string>

enum class EncodeMode : uint32_t
{
	None,
	BC = 0b1,
	DTBC = 0b10,
	All = 0b11,
};

torch::Tensor GumbelSample(const torch::Tensor& logits, double noisy);

torch::Tensor GumbelMax(const torch::Tensor& logits, double noisy, int dim = -1);

torch::Tensor MoPSelect(const torch::Tensor& logits, int Ns, int Nr, int dim = 0);

class Compressor : torch::nn::Module
{
public:
	enum class QuantizeMode : uint32_t
	{
		None = 0,
		Mask = 0b1,
		Color = 0b10,
		All = 0b11,

		Default = 0b11,
	};
	enum class OptimizeMode : uint32_t
	{
		FixConfig,
		DTBC,
	};
	Compressor(at::DeviceType device, int epoch = 10, float lr = 0.1, QuantizeMode quantizeMode = QuantizeMode::None, OptimizeMode optmizeMode = OptimizeMode::DTBC);
	virtual ~Compressor() {}

	bool _QuantizeColor, _QuantizeMask;
	OptimizeMode _optimizeMode;
	torch::Tensor _src; //[n,b*b,c]
	torch::Tensor _dest;
	torch::Tensor _mode7_learned_weight; //[n,p]
	torch::Tensor _mode7_subset0_weight; //[k+h,n,b*b]
	torch::Tensor _mode7_subset1_weight; //[k+h,n,b*b]
	torch::Tensor _bc6_mode1To10_learned_weight; //[n,p]
	torch::Tensor _bc6_mode1To10_subset0_weight; //[k+h,n,b*b]
	torch::Tensor _bc6_mode1To10_subset1_weight; //[k+h,n,b*b]
	bool _init_MoP_weight = false;
	bool _updateMoPweight = true;
	int _epoch;
	float _lr;
	at::DeviceType _device = at::kCPU;

	torch::Tensor forward(const torch::Tensor &src, double noisy = 1.0);
	torch::Tensor backward(const torch::Tensor &gradinput);
	bool DTBCLRScheduler(double cost,double& histcost,int& lr_interval,int lr_patience);


	virtual void encode() = 0;
	virtual std::vector<torch::Tensor> getcode() = 0;
	virtual torch::Tensor decode(const std::vector<torch::Tensor>& code, double noisy = 1.0) = 0;
	torch::Tensor QuantizeMask(const torch::Tensor& mask, int qmax);
	torch::Tensor QuantizeAlphaMask(const torch::Tensor& Alphamask, int Alphaqmax, int Alphamaskqmax);

	// src => mask * v + mu
	void subset_encode(const torch::Tensor& src /*[x,b*b,c]*/, torch::Tensor& v /*[x,c]*/, torch::Tensor& mu /*[x,c]*/, torch::Tensor& mask /*[x,b*b]*/, const torch::Tensor* weight /*[x,b*b]*/ = nullptr);
	torch::Tensor subset_decode(const torch::Tensor& src, const torch::Tensor& c0, const torch::Tensor& c1, const torch::Tensor& mask, int QuantizeMaskMaxValue = 0, int QuantizeColorMaxValue = 0);
	torch::Tensor DTBCcodec(const torch::Tensor blockfeature, double noisy);
};
