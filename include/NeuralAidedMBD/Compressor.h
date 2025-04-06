#pragma once
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <UMBD_ext/UMBDCeresSolver.h>

class CustomRound : public torch::autograd::Function<CustomRound> {
public:
	static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& input);
	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, const torch::autograd::tensor_list& grad_outputs);
};

class CustomRound2 : public torch::autograd::Function<CustomRound2> {
public:
	static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x);
	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_output);
};

class CustomClamp : public torch::autograd::Function<CustomClamp> {
public:
	static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x);
	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_output);
};

torch::Tensor GumbelSample(const torch::Tensor& logits, double noisy);

torch::Tensor GumbelSoftmaxSample(const torch::Tensor& logits, double tau, double noisy, int dim = -1);

torch::Tensor GumbelSoftmax(const torch::Tensor& logits, double tau, double noisy, bool hard, int dim = -1);

torch::Tensor GumbelMax(const torch::Tensor& logits, double noisy, int dim = -1);

torch::Tensor STESoftmax(const torch::Tensor& logits, double tau, bool hard, int dim = -1);

torch::Tensor ArgMax(const torch::Tensor& logits, int dim = -1);

torch::Tensor AutoMax(const torch::Tensor& logits, double tau, double noisy, bool hard, int dim = -1, double delta = 1e-4);

torch::Tensor MoPSelect(const torch::Tensor& logits, double noisy, int Ns, int Nr, int dim = 0, double delta = 1e-4);

class Compressor : torch::nn::Module
{
public:
	enum class QuantizeMode : uint32_t
	{
		None = 0,
		Mask = 0b1,
		Color = 0b10,
		LeakyClamp = 0b100,
		TwoStage = 0b1000,
		Default = 0b0100,
		Best = 0b1101,
		All = 0b1111,
	};
	enum class OptimizeMode : uint32_t
	{
		FixConfig,
		DTBC,
	};
	Compressor(at::DeviceType device, int refinecount = 2, int epoch = 10, float lr = 0.1, QuantizeMode quantizeMode = QuantizeMode::None, OptimizeMode optmizeMode = OptimizeMode::DTBC);
	virtual ~Compressor() {}

	bool _TwoStage = false;
	float _leaky = 1.f;
	bool _QuantizeColor,_QuantizeMask;
	OptimizeMode _optimizeMode;
	torch::Tensor _src; //[n,b*b,c]
	torch::Tensor _dest;
	torch::Tensor _mode7_learned_weight; //[n,p]
	torch::Tensor _mode7_subset0_weight; //[k+h,n,b*b]
	torch::Tensor _mode7_subset1_weight; //[k+h,n,b*b]
	bool _init_mode7_weight = false;
	bool _updateMoPweight = true;
	int _refinecount;
	int _epoch;
	float _lr;
	float _lossqsum;
	at::DeviceType _device = at::kCPU;
	torch::Tensor _eps3 = torch::tensor({ 0.001 });
	torch::Tensor _eps6 = torch::tensor({ 0.000001 });
	torch::Tensor _eps7 = torch::tensor({ 0.0000001 });
	torch::Tensor _eps8 = torch::tensor({ 0.00000001 });
	torch::Tensor _eps9 = torch::tensor({ 0.000000001 });
	torch::Tensor _eps10 = torch::tensor({ 0.0000000001 });
	torch::Tensor _eps12 = torch::tensor({ 1e-12 });
	torch::Tensor _eps14 = torch::tensor({ 0.00000000000001 });
	torch::Tensor _eps16 = torch::tensor({ 1e-16 });
	torch::Tensor _eps20 = torch::tensor({ 1e-20 });

	virtual torch::Tensor forward(const torch::Tensor &src) = 0;
	virtual torch::Tensor backward(const torch::Tensor &gradinput) = 0;
	virtual void MBDOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f);
	virtual void BC1DOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f);
	virtual void BC7DOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f);
	virtual void BCOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f);
	static void GetCTex(UMBD::FCompressedData& compressedData, std::string name);
	int DTBCLRScheduler(float& roundc, const double& qcost,double& histqcost,int& cooldown,int& lr_interval,int& lr_patience,int& enter_nextpass,const int& enter_nextpass_patience,int& pass,const int& maxpass);


	virtual void encode(float roundc, double tau = 1.0, double noisy = 1.0) = 0;
	virtual std::vector<torch::Tensor> getcode() = 0;
	virtual torch::Tensor decode(const std::vector<torch::Tensor>& code) = 0;
	virtual torch::Tensor qdecode(const std::vector<torch::Tensor>& code, float roundc, double tau = 1.0, double noisy = 1.0) = 0;
	torch::Tensor QuantizeMask(const torch::Tensor& mask, int qmax, float roundc);
	torch::Tensor QuantizeAlphaMask(const torch::Tensor& Alphamask, int Alphaqmax, int Alphamaskqmax, float roundc);

	/**
	 * src => mask * v + mu
	 * 
	 * @param src [n,b*b,c]
	 * @param v [n,c]
	 * @param mu [n,c]
	 * @param mask [n,b*b]
	 */
	void OptimizeColorsBlock(const torch::Tensor& src, torch::Tensor& max16, torch::Tensor& min16, torch::Tensor& mask, const torch::Tensor* weight = nullptr);
	void OptimizeAlphaBlock(const torch::Tensor& srcA, torch::Tensor& Alphamax16, torch::Tensor& Alphamin16, torch::Tensor& Alphamask);
	torch::Tensor MatchColorsBlock(const torch::Tensor& srcRGB, const torch::Tensor& c0, const torch::Tensor& c1, const torch::Tensor* weight = nullptr/*[m,n,b*b]*/);
	void RefineBlock(const torch::Tensor& src, const torch::Tensor& mask, torch::Tensor& max16, torch::Tensor& min16, const torch::Tensor* weight = nullptr/*[m,n,b*b]*/);
	void subset_encode(const torch::Tensor& subset_src, torch::Tensor& c0, torch::Tensor& c1, torch::Tensor& mask, torch::Tensor* weight = nullptr);
	torch::Tensor subset_decode(const torch::Tensor& src, const torch::Tensor& c0, const torch::Tensor& c1, const torch::Tensor& mask, float roundc, int maskBits = 0, int colorBits = 0);
};
