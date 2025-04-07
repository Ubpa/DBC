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
			//Bc7e(ParamTexC);
			nvtt_bc7(ParamTexC);
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
	encode();
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
			//Bc7e(ParamTexC);
			nvtt_bc7(ParamTexC);
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
	auto optimizer = torch::optim::Adam({ B,C }, torch::optim::AdamOptions(_lr));
	auto start = std::chrono::system_clock::now();
	float DTBC_time = 0;
	double histcost = 1e20;
	int lr_patience = 40;
	int cooldown = 0;
	int lr_interval = 0;
	double noisy = 100;
	std::vector<float> RMSEs;
	float RMSE_minmum = 1e15f;
	string cost_path = "cost.txt";
	string RMSE_path = "RMSE.txt";
	TxTClear(cost_path);
	TxTClear(RMSE_path);
	_init_MoP_weight = true;
	int rmse_patience = 5;
	int rmse_count = 0;

	for (int i = 0; i < _epoch; ++i)
	{
		noisy *= std::pow(1e-4 / 100, 1.0 / 2000.0);
		auto tmp_start = std::chrono::system_clock::now();

		optimizer.zero_grad();

		// encode C
		_src = C;
		encode();
		_init_MoP_weight = false;
		// C: torch -> MBD
		Tensor edC = decode(getcode(), noisy); // e: encode, d: decode, [-1,1]

		//Tensor lossq = 100 * torch::sum(torch::pow(torch::clamp(C, -1, 1) - C, 2));
		//if (i % 50 == 0)
		//{
		//	cout << "max:" << torch::max(C).item().toFloat()
		//		<< " min:" << torch::min(C).item().toFloat() << endl;
		//}
		//float _lossqsum = (lossq.item().toFloat());
		//lossq.backward(torch::ones_like(lossq), true);
		
		//Tensor lossq1 = torch::mean(torch::sum(torch::pow(torch::clamp(edC, -1, 1) - edC, 2), { -2,-1 })) * 0.1;
		//float lossq1sum = (lossq1.item().toFloat());
		//lossq1.backward(torch::ones_like(lossq1), true);

		auto MBD_start = std::chrono::system_clock::now();
		BlockToTex(newParamTexC, edC, 4, L);

		double cost = 0.;
		bool bBaseEvaluateSuccess = mbdf->Evaluate(newParameters, &cost, gradient);
		//cost += _lossqsum;

		Tensor tensorGradC = TexToBlock(GradTexC, 4, L, _device).to(tensorType).to(_device);
		auto MBD_time = std::chrono::system_clock::now() - MBD_start;
		B.backward(tensorGradB);
		edC.backward(tensorGradC);
		optimizer.step();

		auto time = std::chrono::system_clock::now() - tmp_start;
		auto DTBC_curr_time = std::chrono::duration_cast<std::chrono::milliseconds>(time-MBD_time).count();
		cout
			<< " e: " << i
			<< " rmse: " << std::sqrt(2. * cost / GridC.GetOuterVolume())
			<< " lr: " << optimizer.param_groups()[0].options().get_lr()
			<< " t: " << DTBC_curr_time << " ms"
			<< " n: " << noisy
			<< endl;
		DTBC_time += DTBC_curr_time;
		WriteFloat(i, (float)cost, cost_path);
		
		bool LRstep = DTBCLRScheduler(cost, histcost, lr_interval, lr_patience);
		if (LRstep)
		{
			double lr_now = optimizer.param_groups()[0].options().get_lr();
			optimizer.param_groups()[0].options().set_lr(lr_now * 0.9);
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
			//Bc7e(tmpCompressedData.GetC());
			nvtt_bc7(tmpCompressedData.GetC());
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
			else
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
