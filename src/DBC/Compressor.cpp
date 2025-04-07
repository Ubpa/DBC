#include <DBC/Compressor.h>
#include <DBC/Utils.h>
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

static Tensor STERound(Tensor value)
{
	return (torch::round(value) - value).detach() + value;
}

Tensor GumbelSample(const Tensor& logits, double noisy)
{
	auto gumbel_noise = -torch::log(-torch::log(torch::rand_like(logits)));
	// this foluma is more robust, equal to "logits / noisy + gumbel_noise"
	return logits + noisy * gumbel_noise;
}

Tensor GumbelMax(const Tensor& logits, double noisy, int dim)
{
	auto noisy_logits = GumbelSample(logits, noisy);
	auto y = torch::zeros_like(logits);
	auto index = std::get<1>(noisy_logits.max(dim, true));
	y.scatter_(dim, index, 1.0);
	return y;
}

Tensor MoPSelect(const Tensor& logits, int Ns, int Nr, int dim)
{
	Tensor indices_stable = std::get<1>(logits.topk(Ns, dim, true, false)); //[Ns,n]
	Tensor noisy_logits = GumbelSample(logits, /*noisy*/ 1.f);
	noisy_logits.scatter_(dim, indices_stable, -std::numeric_limits<float>::max());
	Tensor indices_random = std::get<1>(noisy_logits.topk(Nr, dim, true, false)); //[Nr,n]
	return torch::cat({ indices_stable,indices_random }); //[Ns+Nr,n]
}

Compressor::Compressor(at::DeviceType device,int epoch,float lr, QuantizeMode quantizeMode, OptimizeMode optmizeMode)
{
	_QuantizeColor = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::Color) > 0;
	_QuantizeMask = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::Mask) > 0;
	_optimizeMode = optmizeMode;
	_lr = lr;
	_epoch = epoch;
	_device = device;
}

void Compressor::subset_encode(const Tensor& src /*[x,b*b,c]*/, Tensor& v /*[x,c]*/, Tensor& mu /*[x,c]*/, Tensor& mask /*[x,b*b]*/, const Tensor* weight)
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
			max_color = STERound((max_color + 1.f) / 2.f * QuantizeColorMaxValue) / QuantizeColorMaxValue * 2.f - 1.f;
			min_color = STERound((min_color + 1.f) / 2.f * QuantizeColorMaxValue) / QuantizeColorMaxValue * 2.f - 1.f;
		}
		// remap mask
		Tensor diff = max_color - min_color; //[m,n,c]
		Tensor norm = torch::norm(diff, 2, -1, true) + 1e-8; //[m,n,1]
		Tensor v = diff / norm; //[m,n,c]
		Tensor normed_mask = torch::matmul(src - min_color.unsqueeze(-2), v.unsqueeze(-1)).squeeze(-1) / norm;//[m,n,b*b]
		if (_QuantizeMask)
			normed_mask = STERound(normed_mask * QuantizeMaskMaxValue) / QuantizeMaskMaxValue;
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
	Tensor mask_range = max_mask - min_mask + 1e-6; //[m,n,1]
	Tensor normed_mask = (mask - min_mask) / mask_range; //[m,n,b*b]
	Tensor normed_qmask = STERound(normed_mask * qmax) / qmax;
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
			minmum = STERound((minmum + 1.f) / 2.f * Alphaqmax) / Alphaqmax * 2.f - 1.f;
			maxmum = STERound((maxmum + 1.f) / 2.f * Alphaqmax) / Alphaqmax * 2.f - 1.f;
		}
		Tensor scale = maxmum - minmum + 1e-6; //[m,n,1]
		Tensor normed_mask = (Alphamask - minmum) / scale; //[m,n,b*b]
		if (_QuantizeMask)
			normed_mask = STERound(normed_mask * Alphamaskqmax) / Alphamaskqmax;
		return normed_mask * scale + minmum; //[m,n,b*b]
	}
	else
	{
		return Alphamask;
	}
}
