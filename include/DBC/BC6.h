#pragma once

#include <DBC/Compressor.h>

class BC6 : public Compressor
{
public:
	enum class Mode1To10Type
	{
		BruteForce,
		MoP, // Mixture of Partitions
	};

	BC6(at::DeviceType device, bool* use_mode = nullptr, QuantizeMode quantizeMode = QuantizeMode::None, OptimizeMode optimizeMode = OptimizeMode::DBC, Mode1To10Type mode1To10Type = Mode1To10Type::MoP, int Ns = 2, int Nr = 2);

	virtual ~BC6()
	{
		delete _modeweight;
		_modeweight = nullptr;
	}

	torch::Tensor* _modeweight = nullptr;
	//0: mode11To14, v, [1,n,3]
	//1: mode11To14, mu, [1,n,3]
	//2: mode11To14, mask, [n,b*b]
	std::vector <torch::Tensor> _code11To14;

	// c0+c1+mask, 2 subset
	torch::Tensor _code1To10[3][2];
	// mode1To10, mode11To14
	bool _use_mode[2];
	Mode1To10Type _mode1To10Type;
	int _Ns = 1, _Nr = 0;
	torch::Tensor _MoPIndices;
	// reshape to [p,b*b]
	torch::Tensor _bc6_partition2 = torch::tensor(
		{
			0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,		0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,		0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,		0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,		0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
			0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,		0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,		0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
			0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,		0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,		0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,		0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
			0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,		0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,		0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,		0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,		0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,		0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,		0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0
		}, torch::TensorOptions().requires_grad(false)).reshape({ 32,16 });
	torch::Tensor _origin_bc6_partition2_subset[32 * 2];
	torch::Tensor _origin_bc6_partition2_subset_repermute = torch::zeros({ 32,16 }, torch::TensorOptions().dtype(torch::kInt64));

	virtual void encode() override;
	virtual std::vector<torch::Tensor> getcode() override;
	virtual torch::Tensor decode(const std::vector<torch::Tensor>& code, double noisy = 1.0) override;
};