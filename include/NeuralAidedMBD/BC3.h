#pragma once

#include <NeuralAidedMBD/Compressor.h>

class BC3 : public Compressor
{
public:
	BC3(at::DeviceType device, int epoch = 10, float lr = 0.1, QuantizeMode quantizeMode = QuantizeMode::None);
	virtual ~BC3() {}

	torch::Tensor _v; //[n,3]
	torch::Tensor _mu; //[n,3]
	torch::Tensor _mask; //[n,b*b]
	torch::Tensor _alphamask; //[n,b*b]
	torch::Tensor _srcRGB; //[n,b*b,3]

	virtual void encode() override;
	virtual std::vector<torch::Tensor> getcode() override;
	virtual torch::Tensor decode(const std::vector<torch::Tensor>& code, double noisy = 1.0) override;
};
