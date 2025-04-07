#include <NeuralAidedMBD/Compressor.h>
#include <NeuralAidedMBD/Utils.h>
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>
#include <chrono>
#include <thread>
#include <mutex>

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;
using namespace torch::autograd;

//C:\Program Files\NVIDIA Corporation\NvToolsExt\\lib\x64\nvToolsExt64_1.lib

static Tensor AutoRound(Tensor value, float roundc = 0)
{
	if (abs(roundc) < 1e-8)
	{
		return (torch::round(value) - value).detach() + value; //round STE
	}
	else if (abs(roundc + 1) < 1e-8)
	{
		return torch::round(value); // no gradient
	}
	else
	{
		return value - torch::atan(-roundc * torch::sin(2 * M_PI * value) / (1 - roundc * torch::cos(2 * M_PI * value))) / M_PI; // soft round
	}
}

static Tensor LeakyClamp(Tensor value, float minv, float maxv, float leaky = 1.f)
{
	if (leaky == 0.f)
	{
		return torch::clamp(value, minv, maxv);
	}
	else if (leaky == 1.f)
	{
		return value;
	}
	else
	{
		return torch::clamp(value, minv, maxv)
			+ torch::clamp(value - maxv, 0, {}) * leaky
			+ torch::clamp(value - minv, {}, 0) * leaky;
	}
}

Tensor CustomRound::forward(AutogradContext* ctx, const Tensor& input)
{
	// Save the input for use in the backward pass
	ctx->save_for_backward({ input });
	// Use torch::round for rounding
	return torch::round(input);
}

tensor_list CustomRound::backward(AutogradContext* ctx, const torch::autograd::tensor_list& grad_outputs) {
	// Retrieve the saved input from the context
	auto saved = ctx->get_saved_variables();
	auto input = saved[0];

	// Pass-through gradient
	return { grad_outputs[0] }; // Return the same gradient as the input
}

Tensor CustomRound2::forward(AutogradContext* ctx, Tensor x) 
{
	Tensor c = torch::tensor(-0.95).to(x.device());
	Tensor pi = torch::tensor(M_PI).to(x.device());
	{
		torch::AutoGradMode enable_grad(true);
		//Tensor output = x + torch::rand_like(x) - 0.5;
		Tensor output = x - torch::atan(-c * torch::sin(2 * pi * x) / (1 - c * torch::cos(2 * pi * x))) / pi;
		//Tensor output = x;
		//Tensor output = torch::round(x) - x.detach() + x;
		//Tensor output = torch::round(x);
		//Tensor output = torch::round(x) + torch::pow(x - torch::round(x), 3);
		ctx->save_for_backward({ x, output });
	}
	//return x;
	//return torch::round(x) - x.detach() + x;
	return torch::round(x);
	//return torch::round(x) + torch::pow(x - torch::round(x), 3);
	//return x - torch::atan(-c * torch::sin(2 * pi * x) / (1 - c * torch::cos(2 * pi * x))) / pi;
}

tensor_list CustomRound2::backward(AutogradContext* ctx, tensor_list grad_output) {
	auto saved = ctx->get_saved_variables();
	auto x = saved[0];
	auto output = saved[1];
	tensor_list grad = torch::autograd::grad({ output }, { x }, grad_output, c10::optional<bool>(false));
	return grad;
}
Tensor CustomClamp::forward(AutogradContext* ctx, Tensor x)
{
	c10::Scalar c(0.00001);
	{
		torch::AutoGradMode enable_grad(true);
		//Tensor g = torch::tensor(200.0);
		Tensor output = x;
		//Tensor output = torch::clamp(x, 0, 1);
		//Tensor output = torch::max(x, torch::zeros_like(x)) + c * torch::min(x, torch::zeros_like(x));
		//output = torch::min(output - 1, torch::zeros_like(output)) + c * torch::max(output - 1, torch::zeros_like(output)) + 1;
		//Tensor output = 1 / (1 + torch::pow(5.0, 0.5 - x));
		ctx->save_for_backward({ x, output });
	}
	return x;
	//return torch::clamp(x, 0, 1);
	//return 1 / (1 + torch::pow(200.0, 0.5 - x));
	//x = torch::max(x, torch::zeros_like(x)) + c * torch::min(x, torch::zeros_like(x));
	//return torch::min(x - 1, torch::zeros_like(x)) + c * torch::max(x - 1, torch::zeros_like(x)) + 1;
}

tensor_list CustomClamp::backward(AutogradContext* ctx, tensor_list grad_output) {
	auto saved = ctx->get_saved_variables();
	auto x = saved[0];
	auto output = saved[1];
	tensor_list grad = torch::autograd::grad({ output }, { x }, grad_output, c10::optional<bool>(false));
	return grad;
}

Tensor GumbelSample(const Tensor& logits, double noisy)
{
	//Tensor probabilities = torch::log(torch::softmax(logits, 0));
	auto gumbel_noise = -torch::log(-torch::log(torch::rand_like(logits)));
	return logits + noisy * gumbel_noise;
}

Tensor GumbelSoftmaxSample(const Tensor& logits, double tau, double noisy, int dim)
{
	return torch::softmax(GumbelSample(logits, noisy) / tau, dim);
}

Tensor GumbelSoftmax(const Tensor& logits, double tau, double noisy, bool hard, int dim)
{
	auto y = GumbelSoftmaxSample(logits, tau, noisy, dim);

	if (hard) {
		auto y_hard = torch::zeros_like(y);
		auto index = std::get<1>(y.max(dim, true));
		y_hard.scatter_(dim, index, 1.0);
		y_hard = y_hard - y.detach() + y; // 保持梯度
		return y_hard;
	}
	return y;
}

Tensor GumbelMax(const Tensor& logits, double noisy, int dim)
{
	auto noisy_logits = GumbelSample(logits, noisy);
	auto y = torch::zeros_like(logits);
	auto index = std::get<1>(noisy_logits.max(dim, true));
	y.scatter_(dim, index, 1.0);
	return y;
}

Tensor RandomMax(const Tensor& logits)
{
	//Tensor indices = torch::randint(0, logits.size(0), { logits.size(1) },c10::TensorOptions().device(logits.device()).dtype(torch::kInt64)).unsqueeze(0);
	Tensor probabilities = torch::ones({ logits.size(1), logits.size(0) }, c10::TensorOptions().device(logits.device()));//[n,Ns+Nr]
	//Tensor probabilities = logits.transpose(0, 1);//[n,Ns+Nr]
	//probabilities = torch::softmax(probabilities, 0);
	Tensor indices = torch::multinomial(probabilities, 1).transpose(0, 1);//[1,n]
	auto y = torch::zeros_like(logits);
	y.scatter_(0, indices, 1);
	return y;
}

Tensor STESoftmax(const Tensor& logits, double tau, bool hard, int dim)
{
	auto y = torch::softmax(logits / tau, dim);
	if (hard) {
		auto y_hard = torch::zeros_like(y);
		auto index = std::get<1>(y.max(dim, true));
		y_hard.scatter_(dim, index, 1.0);
		y_hard = y_hard - y.detach() + y; // 保持梯度
		return y_hard;
	}
	return y;
}

Tensor ArgMax(const Tensor& logits, int dim)
{
	Tensor y = torch::zeros_like(logits);
	auto index = std::get<1>(logits.max(dim, true));
	y.scatter_(dim, index, 1.0);
	return y;
}

Tensor AutoMax(const Tensor& logits, double tau, double noisy, bool hard, int dim, double delta)
{
	if (noisy >= 1e10)
	{
		return RandomMax(logits);//[m,n]
	}
	else if (tau <= delta && noisy <= delta)
	{
		return ArgMax(logits, dim);//[m,n]
	}
	else if (tau <= delta)
	{
		return GumbelMax(logits, noisy, dim);//[m,n]
	}
	else if (noisy <= delta)
	{
		return STESoftmax(logits, tau, hard, dim);//[m,n]
	}
	else
	{
		return GumbelSoftmax(logits, tau, noisy, hard, dim);//[m,n]
	}
}

Tensor MoPSelect(const Tensor& logits, int Ns, int Nr, int dim)
{
	Tensor indices_stable = std::get<1>(logits.topk(Ns, dim, true, false)); //[Ns,n]
	Tensor noisy_logits = GumbelSample(logits, /*noisy*/ 1.f);
	noisy_logits.scatter_(dim, indices_stable, -std::numeric_limits<float>::max());
	Tensor indices_random = std::get<1>(noisy_logits.topk(Nr, dim, true, false)); //[Nr,n]
	return torch::cat({ indices_stable,indices_random }); //[Ns+Nr,n]

	//Tensor probabilities = torch::ones({ logits.size(1), logits.size(0) }, c10::TensorOptions().device(logits.device()));//[n,64]
	//Tensor probabilities = logits.transpose(0, 1);//[n,64]
	//Tensor indices = torch::multinomial(probabilities, Ns + Nr).transpose(0, 1);//[Ns+Nr,n]
	//return indices;
}

Compressor::Compressor(at::DeviceType device,int epoch,float lr, QuantizeMode quantizeMode, OptimizeMode optmizeMode)
{
	_QuantizeColor = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::Color) > 0;
	_QuantizeMask = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::Mask) > 0;
	_optimizeMode = optmizeMode;
	_lr = lr;
	_epoch = epoch;
	_device = device;
	_eps3 = _eps3.to(device);
	_eps6 = _eps6.to(device);
	_eps7 = _eps7.to(device);
	_eps8 = _eps8.to(device);
	_eps9 = _eps9.to(device);
	_eps10 = _eps10.to(device);
	_eps12 = _eps12.to(device);
	_eps14 = _eps14.to(device);
	_eps16 = _eps16.to(device);
	_eps20 = _eps20.to(device);
}

void Compressor::subset_encode(const Tensor& subset_src/*[m,n,b*b,c]*/, Tensor& c0/*[m,n,c]*/, Tensor& c1/*[m,n,c]*/, Tensor& mask/*[m,n,b*b]*/, Tensor* weight/*[m,n,b*b]*/)
{
	OptimizeColorsBlock(subset_src, c0, c1, mask,weight);
}

Tensor Compressor::subset_decode(const torch::Tensor& src/*[m,n,b*b,c] or [n,b*b,c]*/, const Tensor& c0/*[m,n,c]*/, const Tensor& c1/*[m,n,c]*/, const Tensor& mask/*[m,n,b*b]*/, int QuantizeMaskMaxValue, int QuantizeColorMaxValue)
{
	if (_QuantizeColor)
	{
		Tensor max_mask = std::get<0>(torch::max(mask, -1, true)); //[m,n,1]
		Tensor min_mask = std::get<0>(torch::min(mask, -1, true)); //[m,n,1]
		Tensor max_color = c0 * max_mask + c1; //[m,n,c]
		Tensor min_color = c0 * min_mask + c1; //[m,n,c]
		if (_QuantizeColor)
		{
			max_color = AutoRound((max_color + 1.f) / 2.f * QuantizeColorMaxValue) / QuantizeColorMaxValue * 2.f - 1.f;
			min_color = AutoRound((min_color + 1.f) / 2.f * QuantizeColorMaxValue) / QuantizeColorMaxValue * 2.f - 1.f;
		}
		// remap mask
		Tensor diff = max_color - min_color; //[m,n,c]
		Tensor norm = torch::norm(diff, 2, -1, true) + 1e-8; //[m,n,1]
		Tensor v = diff / norm; //[m,n,c]
		Tensor normed_mask = torch::matmul(src - min_color.unsqueeze(-2), v.unsqueeze(-1)).squeeze(-1) / norm;//[m,n,b*b]
		if (_QuantizeMask)
			normed_mask = AutoRound(normed_mask * QuantizeMaskMaxValue) / QuantizeMaskMaxValue;
		return (max_color - min_color).unsqueeze(-2) * normed_mask.unsqueeze(-1) + min_color.unsqueeze(-2);//[m,n,b*b,c]
	}
	else
	{
		Tensor decmask = mask;//[m,n,b*b]
		if (_QuantizeMask)
			decmask = QuantizeMask(decmask, QuantizeMaskMaxValue);
		return decmask.unsqueeze(-1) * c0.unsqueeze(-2) + c1.unsqueeze(-2);//[m,n,b*b,c]
	}
}

void Compressor::OptimizeColorsBlock(const Tensor& src /*[x,b*b,c]*/, Tensor& v /*[x,c]*/, Tensor& mu /*[x,c]*/, Tensor& mask /*[x,b*b]*/, const Tensor* weight /*[x,b*b]*/)
{
	Tensor center_src /*[x,b*b,c]*/;
	if (weight)
	{
		Tensor norm_weight = *weight / (weight->sum(-1, true) + 1e-8); //[x,b*b]
		mu = torch::matmul(norm_weight.unsqueeze(-2), src).squeeze(-2); //[x,c]<-[x,1,c] = [x,1,b*b] x [x,b*b,c]
		center_src = src - mu.unsqueeze(-2); //[x,b*b,c]
		center_src = torch::multiply(center_src, weight->unsqueeze(-1));
	}
	else
	{
		mu = torch::mean(src, -2); //[x,c]
		center_src = src - mu.unsqueeze(-2); //[x,b*b,c]
	}
	Tensor eigenvectors = get<2>(torch::linalg::svd(center_src + 1e-5, true, {})); //[x,c,c]
	v = eigenvectors.index({ Slice(),Slice(),0 }); //[x,c]
	mask = torch::matmul(center_src, v.unsqueeze(-1)).squeeze(-1); //[x,b*b]<-[x,b*b,1] = [x,b*b,c] x [x,c,1]
}

void Compressor::OptimizeAlphaBlock(const Tensor& srcA /*[m,n,b*b,1]*/, Tensor& Alphamax16 /*[m,n,1]*/, Tensor& Alphamin16 /*[m,n,1]*/, Tensor& Alphamask /*[m,n,b*b]*/)
{
	Alphamin16 = torch::zeros_like(torch::mean(srcA, 2));//[m,n,1]
	Alphamax16 = torch::ones_like(Alphamin16);//[m,n,1]
	Alphamask = srcA.squeeze(-1);//[m,n,b*b]
}
Tensor Compressor::MatchColorsBlock(const Tensor& srcRGB/*[m,n,b*b,c]*/,const Tensor& c0/*[m,n,c]*/, const Tensor& c1/*[m,n,c]*/, const Tensor* weight /*[m,n,b*b]*/)
{
	Tensor dir = c0;//[m,n,c]
	Tensor mu = c1;//[m,n,c]
	Tensor center_src = srcRGB - mu.unsqueeze(2);//[m,n,b*b,c]
	if (weight)
		center_src = torch::multiply(center_src, weight->unsqueeze(3));//[m,n,b*b,c]
	Tensor dots = torch::matmul(center_src, dir.unsqueeze(3)); //[m,n,b*b,1] = [m,n,b*b,c] x [m,n,c,1]
	Tensor mask = dots.squeeze(3);//[m,n,b*b]
	return mask;
}

void Compressor::RefineBlock(const Tensor& src/*[m,n,b*b,c]*/, const Tensor& mask/*[m,n,b*b]*/, Tensor& max16/*[m,n,c]*/, Tensor& min16/*[m,n,c]*/,const Tensor* weight/*[m,n,b*b]*/)
{
	Tensor A = torch::stack({ torch::ones_like(mask), mask }, -1);//[m,n,b*b,2]
	Tensor x;
	if (weight == nullptr)
	{
		x = std::get<0>(torch::linalg::lstsq(A, src, {}, {}));//[m,n,2,c]
	}
	else
	{
		Tensor weighted_src = torch::multiply(src, weight->unsqueeze(3));//[m,n,b*b,c]
		A = torch::multiply(A, weight->unsqueeze(3));//[m,n,b*b,2]
		x = std::get<0>(torch::linalg::lstsq(A, weighted_src, {}, {}));//[m,n,2,c]
	}
	min16 = x.index({ Slice(),Slice(),0 });//[m,n,c]
	max16 = x.index({ Slice(),Slice(),1 });//[m,n,c]
}

bool Compressor::DTBCLRScheduler(double cost,double& histcost,int& lr_interval,int lr_patience)
{
	/**
		* 1. cost < histcost => good
		* 2. no patience => decay lr
		* 3. decay patience
		*/
	if (cost < histcost)
	{
		lr_interval = 0;
		histcost = cost;
		return false;
	}
	else if (lr_interval > lr_patience)
	{
		lr_interval = 0;
		return true;
	}
	else
	{
		lr_interval++;
		return false;
	}
}

Tensor Compressor::DTBCcodec(const Tensor blockfeature, double noisy)
{
	_src = blockfeature;//[N,b*b,c]
	encode();//[N,b*b,c]
	Tensor edC = decode(getcode(), noisy);//[N,b*b,c]
	return edC;//[N,b*b,c]
}

Tensor Compressor::forward(const Tensor& src, double noisy)
{
	_src = src.clone().detach().to(_device).requires_grad_(true);
	encode();
	_dest = decode(getcode(), noisy);
	return _dest;
}
Tensor Compressor::backward(const Tensor& gradinput)
{
	Tensor grad = gradinput.to(_device);
	_dest.backward(grad);
	return _src.grad();
}

Tensor Compressor::QuantizeMask(const Tensor& mask /*[m,n,b*b]*/, int qmax)
{
	Tensor min_mask = get<0>(torch::min(mask, 2, true)); //[m,n,1]
	Tensor max_mask = get<0>(torch::max(mask, 2, true)); //[m,n,1]
	Tensor mask_range = max_mask - min_mask + _eps6; //[m,n,1]
	Tensor normed_mask = (mask - min_mask) / mask_range; //[m,n,b*b]
	Tensor normed_qmask = AutoRound(normed_mask * qmax) / qmax;
	return mask_range * normed_qmask + min_mask;
}

Tensor Compressor::QuantizeAlphaMask(const Tensor& Alphamask/*[m,n,b*b]*/, int Alphaqmax, int Alphamaskqmax)
{
	if (_QuantizeColor || _QuantizeMask)
	{
		Tensor minmum = get<0>(torch::min(Alphamask, -1, true)); //[m,n,1]
		Tensor maxmum = get<0>(torch::max(Alphamask, -1, true)); //[m,n,1]
		if (_QuantizeColor)
		{
			minmum = AutoRound((minmum + 1.f) / 2.f * Alphaqmax) / Alphaqmax * 2.f - 1.f;
			maxmum = AutoRound((maxmum + 1.f) / 2.f * Alphaqmax) / Alphaqmax * 2.f - 1.f;
		}
		Tensor scale = maxmum - minmum + _eps6; //[m,n,1]
		Tensor normed_mask = (Alphamask - minmum) / scale; //[m,n,b*b]
		if (_QuantizeMask)
			normed_mask = AutoRound(normed_mask * Alphamaskqmax) / Alphamaskqmax;
		return normed_mask * scale + minmum; //[m,n,b*b]
	}
	else
	{
		return Alphamask;
	}
}
