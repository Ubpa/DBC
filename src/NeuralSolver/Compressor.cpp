#include <NeuralAidedMBD/Compressor.h>
#include <NeuralAidedMBD/NeuralSolver.h>
#include <NeuralAidedMBD/Utils.h>
#include <UMBDIO.h>
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>
#include "rdo_bc_encoder.h"
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

static Tensor AutoRound(Tensor value, float roundc)
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
	if (tau <= delta && noisy <= delta)
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

Tensor MoPSelect(const Tensor& logits, double noisy, int Ns, int Nr, int dim, double delta)
{
	Tensor indices_stable = std::get<1>(logits.topk(Ns, dim, true, false)); //[Ns,n]
	Tensor noisy_logits = GumbelSample(logits, noisy);
	noisy_logits.scatter_(dim, indices_stable, -std::numeric_limits<float>::max());
	Tensor indices_random = std::get<1>(noisy_logits.topk(Nr, dim, true, false)); //[Nr,n]
	return torch::cat({ indices_stable,indices_random }); //[Ns+Nr,n]
}

Compressor::Compressor(at::DeviceType device, int refinecount,int epoch,float lr, QuantizeMode quantizeMode, OptimizeMode optmizeMode)
{
	_TwoStage = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::TwoStage) > 0;
	_QuantizeColor = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::Color) > 0;
	_QuantizeMask = ((uint32_t)quantizeMode & (uint32_t)QuantizeMode::Mask) > 0;
	_leaky = (((uint32_t)quantizeMode & (uint32_t)QuantizeMode::LeakyClamp) > 0) ? 0.01f : 1.f;
	_optimizeMode = optmizeMode;
	_lossqsum = 0.f;
	_lr = lr;
	_epoch = epoch;
	_refinecount = refinecount;
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

	for (int j = 0; j < _refinecount; ++j)
	{
		RefineBlock(subset_src, mask, c0, c1, weight);
		mask = MatchColorsBlock(subset_src, c0, c1, weight);
	}
}

Tensor Compressor::subset_decode(const torch::Tensor& src/*[m,n,b*b,c] or [n,b*b,c]*/, const Tensor& c0/*[m,n,c]*/, const Tensor& c1/*[m,n,c]*/, const Tensor& mask/*[m,n,b*b]*/, float roundc, int QuantizeMaskMaxValue, int QuantizeColorMaxValue)
{
	Tensor decmask = mask;//[m,n,b*b]
	if (_QuantizeColor || _leaky != 1.f)
	{
		Tensor max_mask = std::get<0>(torch::max(decmask, 2, true)); //[m,n,1]
		Tensor min_mask = std::get<0>(torch::min(decmask, 2, true)); //[m,n,1]
		Tensor max_color = c0 * max_mask + c1; //[m,n,c]
		Tensor min_color = c0 * min_mask + c1; //[m,n,c]
		if (_QuantizeColor)
		{
			max_color = AutoRound((max_color + 1.f) / 2.f * QuantizeColorMaxValue, roundc) / QuantizeColorMaxValue * 2.f - 1.f;
			min_color = AutoRound((min_color + 1.f) / 2.f * QuantizeColorMaxValue, roundc) / QuantizeColorMaxValue * 2.f - 1.f;
		}
		max_color = LeakyClamp(max_color, -1.f, 1.f, _leaky);
		min_color = LeakyClamp(min_color, -1.f, 1.f, _leaky);
		// remap mask
		Tensor diff = max_color - min_color; //[m,n,c]
		Tensor norm = torch::norm(diff, 2, -1, true) + _eps8; //[m,n,1]
		Tensor v = diff / norm; //[m,n,c]
		Tensor normed_mask = torch::matmul(src - min_color.unsqueeze(-2), v.unsqueeze(-1)).squeeze(3) / norm;//[m,n,b*b]
		if (_QuantizeMask)
			normed_mask = AutoRound(normed_mask * QuantizeMaskMaxValue, roundc) / QuantizeMaskMaxValue;
		normed_mask = LeakyClamp(normed_mask, 0.f, 1.f, _leaky);
		return (max_color - min_color).unsqueeze(2) * normed_mask.unsqueeze(3) + min_color.unsqueeze(2);//[m,n,b*b,c]
	}
	else
	{
		if (_QuantizeMask)
			decmask = QuantizeMask(decmask, QuantizeMaskMaxValue, roundc);
		decmask = decmask.unsqueeze(3);//[m,n,b*b,1]
		return decmask * c0.unsqueeze(2) + c1.unsqueeze(2);//[m,n,b*b,c]
	}
}

void Compressor::OptimizeColorsBlock(const Tensor& src /*[m,n,b*b,c]*/, Tensor& v /*[m,n,c]*/, Tensor& mu /*[m,n,c]*/, Tensor& mask /*[m,n,b*b]*/, const Tensor* weight /*[m,n,b*b]*/)
{
	Tensor center_src /*[m,n,b*b,c]*/;
	if (weight)
	{
		Tensor norm_weight = *weight / (weight->sum(-1, true) + _eps8); //[m,n,b*b]
		mu = torch::matmul(norm_weight.unsqueeze(2), src).squeeze(2); //[m,n,c]
		center_src = src - mu.unsqueeze(2); //[m,n,b*b,c]
		center_src = torch::multiply(center_src, weight->unsqueeze(3));//[m,n,b*b,c]
	}
	else
	{
		mu = torch::mean(src, -2); //[m,n,c]
		center_src = src - mu.unsqueeze(2); //[m,n,b*b,c]
	}
	Tensor eigenvectors = get<2>(torch::linalg::svd(center_src + _eps8, true, {})); //[m,n,c,c]
	v = eigenvectors.index({ Slice(),Slice(),0 }); //[m,n,c]
	mask = torch::matmul(center_src, v.unsqueeze(3)).squeeze(3); //[m,n,b*b]<-[m,n,b*b,1] = [m,n,b*b,c] x [m,n,c,1]
}

void Compressor::OptimizeAlphaBlock(const Tensor& srcA /*[m,n,b*b,1]*/, Tensor& Alphamax16 /*[m,n,1]*/, Tensor& Alphamin16 /*[m,n,1]*/, Tensor& Alphamask /*[m,n,b*b]*/)
{
	Alphamin16 = torch::zeros_like(torch::mean(srcA, 2));//[m,n,1]
	Alphamax16 = torch::ones_like(Alphamin16);//[m,n,1]
	Alphamask = srcA.squeeze(3);//[m,n,b*b]
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

void Compressor::MBDOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f)
{
	const uint64_t D = compressedData.GetD();
	const uint64_t L = compressedData.GetL();
	const UMBD::FGrid GridB = compressedData.GetB().GetGrid();
	const UMBD::FGrid GridC = compressedData.GetC().GetGrid();

	UMBD::FTex3D& ParamTexB = compressedData.GetB();
	UMBD::FTex3D& ParamTexC = compressedData.GetC();

	double* newParameters = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	double* gradient = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	UMBD::FTex3D newParamTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters);
	UMBD::FTex3D GradTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient);
	UMBD::FTex3D newParamTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters + ParamTexB.GetNumElements());
	UMBD::FTex3D GradTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient + ParamTexB.GetNumElements());

	for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
	{
		newParamTexB.At<double>(i) = float(ParamTexB.At<UMBD::FHalf>(i)) * compressedData.GetFScales()[i % D];
	}
	for (uint64_t i = 0; i < ParamTexC.GetNumElements(); i++)
	{
		newParamTexC.At<double>(i) = 2.f * ParamTexC.At<float>(i) - 1.f;
	}

	Tensor B = torch::from_blob(newParamTexB.GetStorage(), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, torch::TensorOptions().dtype<double>().requires_grad(true));
	Tensor C = torch::from_blob(newParamTexC.GetStorage(), { (int64_t)GridC.GetOuterVolume(), int64_t(L) }, torch::TensorOptions().dtype<double>().requires_grad(true));
	Tensor tensorGradB = torch::from_blob(const_cast<void*>(GradTexB.GetStorage()), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, c10::TensorOptions().dtype<double>());
	Tensor tensorGradC = torch::from_blob(const_cast<void*>(GradTexC.GetStorage()), { (int64_t)GridC.GetOuterVolume(), int64_t(L) }, c10::TensorOptions().dtype<double>());

	torch::optim::Adam optimizer({ B,C }, torch::optim::AdamOptions(_lr));
	torch::optim::ReduceLROnPlateauScheduler lrs(optimizer, torch::optim::ReduceLROnPlateauScheduler::min, 0.8f, 10, 1e-6, torch::optim::ReduceLROnPlateauScheduler::abs, 0, { 0.01f }, 1e-8, true);
	auto start = std::chrono::system_clock::now();
	float histcost = 0;
	for (int i = 0; i < _epoch; ++i)
	{
		auto epochstart = std::chrono::system_clock::now();
		optimizer.zero_grad();

		double cost = 0.;
		bool bBaseEvaluateSuccess = mbdf->Evaluate(newParameters, &cost, gradient);

		B.backward(tensorGradB);
		C.backward(tensorGradC);
		optimizer.step();
		lrs.step((float)cost);
		cout
			<< i<< ": "
			<< "cost["<< cost << "], "
			<< "time[" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - epochstart).count() << "ms]" << endl;
#if 0
		if (i % 50 == 0 || i == _epoch - 1)
		{
			for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
			{
				ParamTexB.At<UMBD::FHalf>(i) = (float)newParamTexB.At<double>(i) / compressedData.GetFScales()[i % D];
			}
			for (uint64_t i = 0; i < ParamTexC.GetNumElements(); i++)
			{
				ParamTexC.At<float>(i) = (newParamTexC.At<double>(i) + 1.f) / 2.f;
			}
			Bc7e(ParamTexC);
			UMBD::FTex3D approx_f = compressedData.GetApproxF(f.GetGrid());

			double SumSize = 0.;
			double SE = 0.;
			for (const UMBD::FUint64Vector& Point : f.GetGrid())
			{
				double SquaredSize = 0.;
				for (size_t c = 0; c < f.GetNumChannel(); c++)
				{
					double diff = (double)approx_f.At<float>(Point, c) - (double)f.At<float>(Point, c);
					SquaredSize += UCommon::Pow2(f.At<float>(Point, c));
					SE += UCommon::Pow2(diff);
				}
				SumSize += std::sqrt(SquaredSize);
			}
			double Mean = SumSize / f.GetGrid().GetOuterVolume();
			double MSE = SE / f.GetGrid().GetOuterVolume();
			double RMSE = std::sqrt(MSE);
			double RRMSE = RMSE / Mean;

			std::cout
				<< "Mean : " << Mean << std::endl
				<< "SE   : " << SE << std::endl
				<< "MSE  : " << MSE << std::endl
				<< "RMSE : " << RMSE << std::endl
				<< "RRMSE: " << RRMSE * 100. << "%" << std::endl;
		}
#endif
		if (IsFileExist("close.txt"))
			break;

		float costchange = histcost - (float)cost;
		if (abs(costchange) / cost <= 1e-6)
		{
			cout << "abs(costchange)/(*cost)= " << abs(costchange) / cost << endl;
			break;
		}
		histcost = (float)cost;
	}

	for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
	{
		ParamTexB.At<UMBD::FHalf>(i) = (float)newParamTexB.At<double>(i) / compressedData.GetFScales()[i % D];
	}
	for (uint64_t i = 0; i < ParamTexC.GetNumElements(); i++)
	{
		ParamTexC.At<float>(i) = ((float)newParamTexC.At<double>(i) + 1.f) / 2.f;
	}

	auto time = std::chrono::system_clock::now() - start;
	cout << "Run time: " << std::chrono::duration_cast<std::chrono::seconds>(time).count() << 's'<<endl;
}

void Compressor::BC1DOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f)
{
	const uint64_t D = compressedData.GetD();
	const uint64_t L = compressedData.GetL();
	const UMBD::FGrid GridB = compressedData.GetB().GetGrid();
	const UMBD::FGrid GridC = compressedData.GetC().GetGrid();

	UMBD::FTex3D& ParamTexB = compressedData.GetB();
	UMBD::FTex3D& ParamTexC = compressedData.GetC();

	double* newParameters = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	double* gradient = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	UMBD::FTex3D newParamTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters);
	UMBD::FTex3D GradTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient);
	UMBD::FTex3D newParamTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters + ParamTexB.GetNumElements());
	UMBD::FTex3D GradTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient + ParamTexB.GetNumElements());

	for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
	{
		newParamTexB.At<double>(i) = (float)ParamTexB.At<UMBD::FHalf>(i) * compressedData.GetFScales()[i % D];
	}

	c10::ScalarType tensorType = torch::kFloat;

	Tensor B = torch::from_blob(newParamTexB.GetStorage(), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, torch::TensorOptions().dtype<double>().requires_grad(true));
	_src = (TexToBlock(ParamTexC, 4, L, _device) * 2 - 1).to(tensorType).to(_device).requires_grad_(false);
	encode(-1.f);
	Tensor tensorGradB = torch::from_blob(const_cast<void*>(GradTexB.GetStorage()), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, torch::TensorOptions().dtype<double>());

	torch::optim::Adam optimizer({ B },torch::optim::AdamOptions(_lr));
	std::vector<Tensor> code = getcode();
	for (int i = 0; i < code.size(); ++i)
		code[i] = code[i].clone().detach().requires_grad_(true);
	optimizer.add_param_group(torch::optim::OptimizerParamGroup(code));
	//torch::optim::StepLR lrs(optimizer, 100, 0.1);
	torch::optim::ReduceLROnPlateauScheduler lrs(optimizer, torch::optim::ReduceLROnPlateauScheduler::min, 0.8f, 10, 1e-6, torch::optim::ReduceLROnPlateauScheduler::abs, 0, { 0.01f,0.01f }, 1e-8, true);
	auto start = std::chrono::system_clock::now();
	float histcost = 0;
	for (int i = 0; i < _epoch; ++i)
	{
		auto tmp_start = std::chrono::system_clock::now();
		optimizer.zero_grad();

		Tensor dC = decode(optimizer.param_groups()[1].params());
		BlockToTex(newParamTexC, dC, 4, L);
		double cost = 0.;
		auto MBDstart = std::chrono::system_clock::now();
		bool bBaseEvaluateSuccess = mbdf->Evaluate(newParameters, &cost, gradient);
		auto MBDend = std::chrono::system_clock::now();
		cost += _lossqsum;

		Tensor tensorGradC = TexToBlock(GradTexC, 4, L, _device).to(tensorType).to(_device);

		B.backward(tensorGradB);
		dC.backward(tensorGradC);
		optimizer.step();
		lrs.step((float)cost);

		cout << "cost : " << cost << endl;
		auto MBDtime = MBDend - MBDstart;
		auto time = std::chrono::system_clock::now() - tmp_start;
		cout << "epoch: " << i << " "
			<< "cost : " << cost << ' '
			<< "MBD time: " << std::chrono::duration_cast<std::chrono::milliseconds>(MBDtime).count() << " ms"
			<< "whole time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << " ms" << endl;

		if (i % 50 == 0 || i == _epoch - 1)
		{
			for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
			{
				ParamTexB.At<UMBD::FHalf>(i) = (float)newParamTexB.At<double>(i) / compressedData.GetFScales()[i % D];
			}
			Tensor tmp_C = (dC + 1) / 2;
			float minmum = tmp_C.min().item().toFloat();
			float maxmum = tmp_C.max().item().toFloat();
			BlockToTex(ParamTexC, tmp_C, 4, L);
			Bc7e(ParamTexC);
			UMBD::FTex3D approx_f = compressedData.GetApproxF(f.GetGrid());
			EvaluateError(f, approx_f);
		}

		if (IsFileExist("close.txt"))
			break;

		float costchange = histcost - (float)cost;
		if (i > 2000 && abs(costchange) / cost <= 1e-6)
		{
			cout << "abs(costchange)/(*cost)= " << abs(costchange) / cost << endl;
			break;
		}
		histcost = (float)cost;
	}

	for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
	{
		ParamTexB.At<UMBD::FHalf>(i) = (float)newParamTexB.At<double>(i) / compressedData.GetFScales()[i % D];
	}

	Tensor tmp_C = (decode(optimizer.param_groups()[1].params()) + 1) / 2;
	BlockToTex(ParamTexC, tmp_C, 4, L);

	auto time = std::chrono::system_clock::now() - start;
	cout << "Run time: " << std::chrono::duration_cast<std::chrono::seconds>(time).count() << 's'<<endl;
}

void Compressor::BC7DOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f)
{
	const uint64_t D = compressedData.GetD();
	const uint64_t L = compressedData.GetL();
	const UMBD::FGrid GridB = compressedData.GetB().GetGrid();
	const UMBD::FGrid GridC = compressedData.GetC().GetGrid();

	c10::ScalarType tensorType = c10::kFloat;
	UMBD::FTex3D& ParamTexB = compressedData.GetB();
	UMBD::FTex3D& ParamTexC = compressedData.GetC();
	for (uint64_t l = 0; l < L; l++)
		for (uint64_t i = 0; i < GridB.GetOuterVolume(); i++)
			for (uint64_t k = 0; k < D; k++)
				ParamTexB.At<UMBD::FHalf>(i * L * D + l * D + k) *= compressedData.GetFScales()[k];
	for (int i = 0; i < ParamTexC.GetGrid().GetOuterVolume() * ParamTexC.GetNumChannel(); ++i)
		ParamTexC.At<float>(i) = ParamTexC.At<float>(i) * 2 - 1;

	Tensor B = torch::from_blob(const_cast<void*>(ParamTexB.GetStorage()), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, torch::TensorOptions().dtype(torch::kFloat16));
	B = B.to(_device).to(tensorType).clone().detach().requires_grad_(true);
	_src = TexToBlock(ParamTexC, 4, L, _device).to(tensorType).to(_device).requires_grad_(false);
	encode(-1.f);

	double* newParameters = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	double* gradient = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	UMBD::FTex3D newParamTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double,newParameters);
	UMBD::FTex3D GradTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient);
	UMBD::FTex3D newParamTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters + ParamTexB.GetNumElements());
	UMBD::FTex3D GradTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient + ParamTexB.GetNumElements());

	//Tensor roundc1 = (torch::ones({ _src.size(0) * 4 }, torch::TensorOptions().dtype(tensorType).device(_device))*-0.9).clone().detach().requires_grad_(true);
	//Tensor roundc2 = (torch::ones({ _src.size(0) * 4 }, torch::TensorOptions().dtype(tensorType).device(_device))*-0.9).clone().detach().requires_grad_(true);
	//std::vector<Tensor> roundc({ roundc1,roundc2 });
	torch::optim::Optimizer *Adamoptimizer=new torch::optim::Adam({ B }, torch::optim::AdamOptions(_lr));
	//torch::optim::Adam SGDoptimizer({ torch::tensor({ -0.9 }, torch::TensorOptions().dtype(tensorType).device(_device).requires_grad(true)) }, torch::optim::AdamOptions(1.0));
	std::vector<Tensor> code = getcode();
	for (int i = 0; i < code.size(); ++i)
	{
		if ((code[i].dtype() != torch::kFloat && code[i].dtype() != torch::kDouble) || i == 2 || i == 5)
			code[i] = code[i].clone().detach().requires_grad_(false);
		else
			code[i] = code[i].clone().detach().requires_grad_(true);
	}
	(*Adamoptimizer).add_param_group(torch::optim::OptimizerParamGroup(code));
	torch::optim::ReduceLROnPlateauScheduler* multipasslrs = new torch::optim::ReduceLROnPlateauScheduler(*Adamoptimizer, torch::optim::ReduceLROnPlateauScheduler::min, 0.95f, 20, 1e-6, torch::optim::ReduceLROnPlateauScheduler::abs, 100, { _lr / 10000.f,_lr / 10000.f }, 1e-8, true);
	auto start = std::chrono::system_clock::now();
	float histcost = 0;
	double histqcost = std::numeric_limits<double>::max();
	int patience = 40;
	int cooldown = 0;
	bool pass2 = true, pass3 = false;
	float roundc = 0.f;
	int interval = 0;
	float t = 1.0f;
	for (int i = 0; i < _epoch; ++i)
	{
		(*Adamoptimizer).zero_grad();
		Tensor tmp_B = (*Adamoptimizer).param_groups()[0].params()[0].to(c10::kDouble).to(at::kCPU).contiguous();
		memcpy_s(newParamTexB.GetStorage(), sizeof(double) * tmp_B.numel(), tmp_B.data_ptr(), sizeof(double) * tmp_B.numel());

		Tensor multiplemodeC,tmp_C,tmp_qC;
		float realroundc = -1.f;
		tmp_qC = qdecode((*Adamoptimizer).param_groups()[1].params(), realroundc);
		tmp_C = qdecode((*Adamoptimizer).param_groups()[1].params(), roundc);
		BlockToTex(newParamTexC, tmp_C, 4, L);
		Tensor weight = (*Adamoptimizer).param_groups()[1].params()[5];
		weight = torch::softmax(weight, 0);
		Tensor loss2 = torch::mean(torch::pow(torch::sum(weight * weight, 0) - 1, 2), -1) * 1;
		float loss2_val = loss2.item().toFloat();
		double cost = 0.;
		bool bBaseEvaluateSuccess = mbdf->Evaluate(newParameters, &cost, gradient);

		Tensor tensorGradB = torch::from_blob(const_cast<void*>(GradTexB.GetStorage()), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, c10::TensorOptions().dtype(c10::kDouble)).to(tensorType).to(_device);
		Tensor tensorGradC = TexToBlock(GradTexC, 4, L, _device).to(tensorType).to(_device);

		(*Adamoptimizer).param_groups()[0].params()[0].backward(tensorGradB);
		tmp_C.backward(tensorGradC);
		(*Adamoptimizer).step();

		double qcost = 0.;
		BlockToTex(newParamTexC, tmp_qC, 4, L);
		mbdf->Evaluate(newParameters, &qcost, gradient);
		cout <<"cost : "<< cost << ' '<<"qcost : "<<qcost << ' '<<"c : "<<roundc<<' '<<"lr : "<<(*Adamoptimizer).param_groups()[0].options().get_lr()<<endl;
		if (qcost < histqcost && !cooldown)
		{
			interval = 0;
			cooldown = 0;
			histqcost = qcost;
		}
		else if (interval > patience && !cooldown)
		{
			interval = 0;
			float lr_now = (float)(*Adamoptimizer).param_groups()[0].options().get_lr();
			if (lr_now < 0.008 && abs(roundc + 0.95f) > 1e-4)
			{
				cout << "pass2" << endl;
				cooldown = 20;
				//torch::save(optimizer->param_groups()[0].params(), "pass1_(-2).pt");
				torch::optim::Adam* tmp = new torch::optim::Adam((*Adamoptimizer).param_groups(), torch::optim::AdamOptions(_lr));
				delete Adamoptimizer;
				Adamoptimizer = tmp;
				//optimizer->param_groups()[0].options().set_lr(0.01);
				roundc = -0.95f;
			}
			(*Adamoptimizer).param_groups()[0].options().set_lr(lr_now * 0.9);
			(*Adamoptimizer).param_groups()[1].options().set_lr(lr_now * 0.9);
		}
		else
		{
			if (cooldown)
				cooldown--;
			else
			{
				interval++;
			}
		}
		if (i % 50 == 0 || i == _epoch - 1)
		{
			tmp_B = (*Adamoptimizer).param_groups()[0].params()[0].to(c10::kHalf).to(at::kCPU).contiguous();
			memcpy_s(ParamTexB.GetStorage(), 2 * tmp_B.numel(), tmp_B.data_ptr(), 2 * tmp_B.numel());
			for (uint64_t l = 0; l < L; l++)
				for (uint64_t i = 0; i < GridB.GetOuterVolume(); i++)
					for (uint64_t k = 0; k < D; k++)
						ParamTexB.At<UMBD::FHalf>(i * L * D + l * D + k) /= compressedData.GetFScales()[k];
			tmp_C = (tmp_C + 1) / 2;
			float minC = tmp_C.min().item().toFloat();
			float maxC = tmp_C.max().item().toFloat();
			tmp_C = torch::clamp(tmp_C, 0, 1);
			BlockToTex(ParamTexC, tmp_C, 4, L);
			Bc7e(compressedData.GetC());
			UMBD::FTex3D approx_f = compressedData.GetApproxF(f.GetGrid());
			double SumSize = 0.;
			double SE = 0.;
			for (const UMBD::FUint64Vector& Point : f.GetGrid())
			{
				double SquaredSize = 0.;
				for (size_t c = 0; c < f.GetNumChannel(); c++)
				{
					double diff = (double)approx_f.At<float>(Point, c) - (double)f.At<float>(Point, c);
					SquaredSize += UCommon::Pow2(f.At<float>(Point, c));
					SE += UCommon::Pow2(diff);
				}
				SumSize += std::sqrt(SquaredSize);
			}
			double Mean = SumSize / f.GetGrid().GetOuterVolume();
			double MSE = SE / f.GetGrid().GetOuterVolume();
			double RMSE = std::sqrt(MSE);
			double RRMSE = RMSE / Mean;

			std::cout
				<< "Mean : " << Mean << std::endl
				<< "SE   : " << SE << std::endl
				<< "MSE  : " << MSE << std::endl
				<< "RMSE : " << RMSE << std::endl
				<< "RRMSE: " << RRMSE * 100. << "%" << std::endl;
			std::cout << "Epoch : " << i << std::endl;
			std::cout << "minC : " << minC << std::endl;
			std::cout << "maxC : " << maxC << std::endl;
		}

		if (IsFileExist("close.txt"))
			break;

		float costchange = histcost - (float)cost;
		//if (i > 5000 && abs(costchange) / (*cost) <= 1e-6)
		//{
		//	cout << "abs(costchange)/(*cost)= " << abs(costchange) / (*cost) << endl;
		//	break;
		//}
		histcost = (float)cost;
	}
	Tensor tmp_B = (*Adamoptimizer).param_groups()[0].params()[0].to(c10::kHalf).to(at::kCPU).contiguous();
	memcpy_s(ParamTexB.GetStorage(), 2 * tmp_B.numel(), tmp_B.data_ptr(), 2 * tmp_B.numel());
	for (uint64_t l = 0; l < L; l++)
		for (uint64_t i = 0; i < GridB.GetOuterVolume(); i++)
			for (uint64_t k = 0; k < D; k++)
				ParamTexB.At<UMBD::FHalf>(i * L * D + l * D + k) /= compressedData.GetFScales()[k];

	Tensor tmp_C = (qdecode((*Adamoptimizer).param_groups()[1].params(), roundc) + 1) / 2;
	BlockToTex(ParamTexC, tmp_C, 4, L);

	auto time = std::chrono::system_clock::now() - start;
	cout << "Run time: " << std::chrono::duration_cast<std::chrono::seconds>(time).count() << 's'<<endl;
}

int Compressor::DTBCLRScheduler(float& roundc, const double& qcost,double& histqcost,int& cooldown,int& lr_interval,int& lr_patience,int& enter_nextpass,const int& enter_nextpass_patience,int& pass,const int& maxpass)
{
	/**
		* 1. qcost < histqcost => good
		* 2. no patience => try enter pass 2 || decay lr
		* 3. cool down || decay patience
		*/
	if ((qcost < histqcost) && !cooldown)
	{
		if (lr_interval < lr_patience / 2)
		{
			enter_nextpass = std::max(0, enter_nextpass - 1);
		}
		lr_interval = 0;
		histqcost = qcost;
		return 0;
	}
	else if (lr_interval > lr_patience && !cooldown)
	{
		lr_interval = 0;
		enter_nextpass++;
		if (_TwoStage && pass < maxpass && enter_nextpass >= enter_nextpass_patience)
		{
			enter_nextpass = 0;
			pass++;
			roundc = -0.95f;
			return 1;
		}
		else
		{
			return 2;
		}
	}
	else
	{
		if (cooldown)
			cooldown--;
		else
		{
			lr_interval++;
		}
		return 0;
	}
}

void Compressor::BCOptim(UMBD::FCompressedData& compressedData, const UMBD::FFirstOrderFunction* mbdf, const UMBD::FTex3D &f)
{
	const uint64_t D = compressedData.GetD();
	const uint64_t L = compressedData.GetL();
	const UMBD::FGrid GridB = compressedData.GetB().GetGrid();
	const UMBD::FGrid GridC = compressedData.GetC().GetGrid();

	UMBD::FTex3D& ParamTexB = compressedData.GetB();
	UMBD::FTex3D& ParamTexC = compressedData.GetC();

	double* newParameters = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	double* gradient = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	UMBD::FTex3D newParamTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters);
	UMBD::FTex3D GradTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient);
	UMBD::FTex3D newParamTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, newParameters + ParamTexB.GetNumElements());
	UMBD::FTex3D GradTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient + ParamTexB.GetNumElements());

	for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
	{
		newParamTexB.At<double>(i) = (float)ParamTexB.At<UMBD::FHalf>(i) * compressedData.GetFScales()[i % D];
	}

	c10::ScalarType tensorType = torch::kFloat32;

	Tensor B = torch::from_blob(newParamTexB.GetStorage(), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, torch::TensorOptions().dtype<double>().requires_grad(true));
	Tensor C = (TexToBlock(ParamTexC, 4, L, _device) * 2 - 1).to(tensorType).to(_device).requires_grad_(true);
	Tensor tensorGradB = torch::from_blob(const_cast<void*>(GradTexB.GetStorage()), { (int64_t)GridB.GetOuterVolume(), int64_t(L * D) }, c10::TensorOptions().dtype<double>());

	//C->BC7->C'->MBD->L2
	torch::optim::Adam *optimizer=new torch::optim::Adam({ B,C }, torch::optim::AdamOptions(_lr));
	auto start = std::chrono::system_clock::now();
	float DTBC_time = 0;
	double histqcost = 1e20;
	int lr_patience = 40;
	int cooldown = 0;
	int lr_interval = 0;
	int enter_nextpass = 0;
	int enter_nextpass_patience = 3;
	double tau = 0;
	double noisy = 100;
	std::vector<float> RMSEs;
	float RMSE_minmum = 1e15f;
	string cost_path = "cost.txt";
	string qcost_path = "qcost.txt";
	string RMSE_path = "RMSE.txt";
	TxTClear(cost_path);
	TxTClear(qcost_path);
	TxTClear(RMSE_path);
	_init_mode7_weight = true;
	float origLeaky = _leaky;
	int rmse_patience = 5;
	int rmse_count = 0;
	int maxpass = 2;
	//pass 1
	int pass = 1;
	float roundc = 0.f;

	for (int i = 0; i < _epoch; ++i)
	{
		tau *= std::pow(1e-4 / 1, 1.0 / 2000.0);
		noisy *= std::pow(1e-4 / 100, 1.0 / 2000.0);
		auto tmp_start = std::chrono::system_clock::now();

		optimizer->zero_grad();

		// encode C
		_src = C;
		//if (i % 50 == 0 && i < 2000)
		//	_init_mode7_weight = true;
		encode(roundc, 0.f, 1.f);
		_init_mode7_weight = false;
		// C: torch -> MBD
		Tensor edC = qdecode(getcode(), roundc, 0.f, noisy); // e: encode, d: decode, [-1,1]
		auto MBD_start = std::chrono::system_clock::now();
		BlockToTex(newParamTexC, edC, 4, L);

		double cost = 0.;
		bool bBaseEvaluateSuccess = mbdf->Evaluate(newParameters, &cost, gradient);

		Tensor tensorGradC = TexToBlock(GradTexC, 4, L, _device).to(tensorType).to(_device);
		auto MBD_time = std::chrono::system_clock::now() - MBD_start;
		B.backward(tensorGradB);
		edC.backward(tensorGradC);
		optimizer->step();

		double qcost = 0.;
		if (roundc != 0.f)
		{
			float realroundc = -1.f;
			_leaky = 0.f;
			Tensor tmp_qC = qdecode(getcode(), realroundc, tau, noisy);
			_leaky = origLeaky;
			BlockToTex(newParamTexC, tmp_qC, 4, L);
			mbdf->Evaluate(newParameters, &qcost, nullptr);
		}
		else
			qcost = cost;
		auto time = std::chrono::system_clock::now() - tmp_start;
		auto DTBC_curr_time = std::chrono::duration_cast<std::chrono::milliseconds>(time-MBD_time).count();
		cout
			<< " e: " << pass << "|" << i
			//<< " c: " << cost
			<< " rmse: " << std::sqrt(2. * cost / GridC.GetOuterVolume())
			//<< " qc: " << qcost
			<< " qrmse: " << std::sqrt(2. * qcost / GridC.GetOuterVolume())
			<< " lr: " << optimizer->param_groups()[0].options().get_lr()
			<< " t: " << DTBC_curr_time << " ms"
			<< " t: " << tau
			<< " n: " << noisy
			<< endl;
		DTBC_time += DTBC_curr_time;
		WriteFloat(i, (float)cost, cost_path);
		WriteFloat(i, (float)qcost, qcost_path);
		
		int LRstep = DTBCLRScheduler(roundc, qcost, histqcost, cooldown, lr_interval, lr_patience, enter_nextpass, enter_nextpass_patience, pass, maxpass);
		if (LRstep == 1)
		{
			lr_patience = 40;
			cooldown = 20;
			double lr_now = optimizer->param_groups()[0].options().get_lr();
			std::cout << "reach patience in first pass" << endl;
			torch::optim::Adam* tmp = new torch::optim::Adam({ B,C }, torch::optim::AdamOptions(std::min(lr_now, 0.01)));
			delete optimizer;
			optimizer = tmp;
		}
		else if (LRstep == 2)
		{
			double lr_now = optimizer->param_groups()[0].options().get_lr();
			optimizer->param_groups()[0].options().set_lr(lr_now * 0.9);
		}

		if (i % 50 == 0 || i == _epoch - 1 || IsFileExist("close.txt"))
		{
			UMBD::FCompressedData tmpCompressedData = compressedData;
			for (uint64_t i = 0; i < ParamTexB.GetNumElements(); i++)
			{
				tmpCompressedData.GetB().At<UMBD::FHalf>(i) = (float)newParamTexB.At<double>(i) / compressedData.GetFScales()[i % D];
			}
			Tensor tmp_C = (edC + 1) / 2; //[0,1]
			//edC = (varC + 1) / 2;
			float minmum = tmp_C.min().item().toFloat();
			float maxmum = tmp_C.max().item().toFloat();
			BlockToTex(tmpCompressedData.GetC(), tmp_C, 4, L);
			UMBD::FTex3D texC = tmpCompressedData.GetC();
			Bc7e(tmpCompressedData.GetC());
			UMBD::FTex3D approx_f = tmpCompressedData.GetApproxF(f.GetGrid());
			tmpCompressedData.GetC() = std::move(texC);
			float RMSE = EvaluateError(f, approx_f);
			RMSEs.push_back(RMSE);
			WriteFloat(i, RMSE, RMSE_path);
			if (RMSE < RMSE_minmum)
			{
				rmse_count = 0;
				RMSE_minmum = RMSE;
				compressedData = tmpCompressedData;
			}
			else if (!_TwoStage || pass >= maxpass)
			{
				rmse_count++;
				if (rmse_patience >= 0 && rmse_count >= rmse_patience)
				{
					std::cout << "reach rmse patience in final pass" << endl;
					break;
				}
			}
			std::cout << "minmum : " << minmum << " "
				<< "maxmum : " << maxmum <<	" "
				<< "RMSE_minmum: " << RMSE_minmum << std::endl;
		}

		if (IsFileExist("close.txt"))
			break;

	}

	auto time = std::chrono::system_clock::now() - start;
	cout << "Run time: " << std::chrono::duration_cast<std::chrono::seconds>(time).count() << 's'<<endl;
	cout << "DTBC time: " << DTBC_time/1000.f << 's'<<endl;
	SaveStringToFile(ToString(RMSEs, ",\n"), "RMSEs.csv");
}

void Compressor::GetCTex(UMBD::FCompressedData& compressedData, std::string name)
{
	UMBD::FTex3D ParamTexC = compressedData.GetC();
	ParamTexC = ParamTexC.ToUint8();
	for (int i = 0; i < ParamTexC.GetGrid().Depth; ++i)
	{
		UMBD::SaveTex2DAsPNG(ParamTexC, (name +"_"+ std::to_string(i) + ".png").c_str(), i, 0, ParamTexC.GetNumChannel());
	}
}

Tensor Compressor::QuantizeMask(const Tensor& mask /*[m,n,b*b]*/, int qmax, float roundc)
{
	Tensor min_mask = get<0>(torch::min(mask, 2, true)); //[m,n,1]
	Tensor max_mask = get<0>(torch::max(mask, 2, true)); //[m,n,1]
	Tensor mask_range = max_mask - min_mask + _eps6; //[m,n,1]
	Tensor normed_mask = (mask - min_mask) / mask_range; //[m,n,b*b]
	Tensor normed_qmask = AutoRound(normed_mask * qmax, roundc) / qmax;
	return mask_range * normed_qmask + min_mask;
}

Tensor Compressor::QuantizeAlphaMask(const Tensor& Alphamask/*[m,n,b*b]*/, int Alphaqmax, int Alphamaskqmax, float roundc)
{
	if (_QuantizeColor || _leaky != 1.f || _QuantizeMask)
	{
		Tensor minmum = get<0>(torch::min(Alphamask, 2, true)); //[m,n,1]
		Tensor maxmum = get<0>(torch::max(Alphamask, 2, true)); //[m,n,1]
		if (_QuantizeColor)
		{
			minmum = torch::floor((minmum + 1.f) / 2.f * Alphaqmax) / Alphaqmax * 2.f - 1.f;
			maxmum = torch::ceil ((maxmum + 1.f) / 2.f * Alphaqmax) / Alphaqmax * 2.f - 1.f;
		}
		minmum = LeakyClamp(minmum, -1.f, 1.f, _leaky);
		maxmum = LeakyClamp(maxmum, -1.f, 1.f, _leaky);
		Tensor scale = maxmum - minmum + _eps6; //[m,n,1]
		Tensor normed_mask = (Alphamask - minmum) / scale; //[m,n,b*b]
		if (_QuantizeMask)
			normed_mask = AutoRound(normed_mask * Alphamaskqmax, roundc) / Alphamaskqmax;
		normed_mask = LeakyClamp(normed_mask, 0.f, 1.f, _leaky);
		return normed_mask * scale + minmum; //[m,n,b*b]
	}
	else
	{
		return Alphamask;
	}
}