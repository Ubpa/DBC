#pragma once

#include <NeuralAidedMBD/Compressor.h>

class BC1 : public Compressor
{
public:
	BC1(at::DeviceType device, int refinecount = 2, int epoch = 10, float lr = 0.1, QuantizeMode quantizeMode = QuantizeMode::None);
	virtual ~BC1() {}

	torch::Tensor _c0; //[n,c]
	torch::Tensor _c1; //[n,c]
	torch::Tensor _mask; //[n,b*b]

	virtual void encode(float roundc, double tau = 1.0, double noisy = 1.0) override;
	virtual std::vector<torch::Tensor> getcode() override;
	virtual torch::Tensor decode(const std::vector<torch::Tensor>& code) override;
	virtual torch::Tensor qdecode(const std::vector<torch::Tensor>& code, float roundc, double tau = 1.0, double noisy = 1.0) override;
	virtual torch::Tensor forward(const torch::Tensor& src);
	virtual torch::Tensor backward(const torch::Tensor& gradinput);
	//virtual void init(const torch::Tensor& src);
};
