//#pragma once
//
//#include <NeuralAidedMBD/Compressor.h>
//
//class BC1 : public Compressor
//{
//public:
//	BC1(at::DeviceType device, int epoch = 10, float lr = 0.1);
//	virtual ~BC1() {}
//
//	torch::Tensor _c0; //[n,c]
//	torch::Tensor _c1; //[n,c]
//	torch::Tensor _mask; //[n,b*b]
//
//	virtual void encode() override;
//	virtual std::vector<torch::Tensor> getcode() override;
//	virtual torch::Tensor decode(const std::vector<torch::Tensor>& code, double noisy = 1.0) override;
//};
