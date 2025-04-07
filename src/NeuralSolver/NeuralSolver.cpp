#include <NeuralAidedMBD/NeuralSolver.h>
#include <NeuralAidedMBD/Compressor.h>
#include <NeuralAidedMBD/Utils.h>

#include <iostream>
#include <mutex>
#include <assert.h>
#include <UMBDIO.h>

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;
using namespace torch::autograd;

FNerualFirstOrderFunction::FNerualFirstOrderFunction(
	UMBD::FThreadPool* pThreadPool,
	double Theta,
	const double* ThetaWeights,
	const double* CustomWeights,
	const double* BRanges,
	bool bCGradientWall,
	const UMBD::FTex3D* TexF,
	UMBD::FGrid InGridB,
	UMBD::FGrid InGridC,
	uint64_t InL,
	uint64_t InD,
	bool bDebugInfo,
	const FData* InData)
	: FFirstOrderFunction(
		pThreadPool,
		Theta,
		ThetaWeights,
		CustomWeights,
		BRanges,
		bCGradientWall,
		TexF,
		InGridB,
		InGridC,
		InL,
		InD,
		bDebugInfo)
	, GridB(InGridB)
	, GridC(InGridC)
	, L(InL)
	, D(InD)
	, Data(*InData)
{
	assert(InData);
}


void GetGradCTex(const UMBD::FTex3D& GradTexC, string name, uint64_t iterations)
{
	if (iterations < 10 || iterations % 50 == 1)
	{
		UMBD::FTex3D newTexC(GradTexC, UMBD::EOwnership::TakeOwnership, nullptr);
		for (size_t i = 0; i < newTexC.GetGrid().GetOuterVolume() * newTexC.GetNumChannel(); i++) {
			newTexC.At<double>(i) = UCommon::Clamp<double>(std::abs(newTexC.At<double>(i)) * 100000, 0, 1);
		}
		newTexC = newTexC.ToUint8();
		UMBD::SaveTex2DAsPNG(newTexC, (name + std::to_string(iterations) + ".png").c_str(), 0, 0, newTexC.GetNumChannel());
	}
}

void GetEvaluateCTex(const UMBD::FTex3D& TexC, std::string name, uint64_t iterations)
{
	if (iterations < 10 || iterations % 50 == 1)
	{
		UMBD::FTex3D newTexC(TexC, UMBD::EOwnership::TakeOwnership, nullptr);
		for (size_t i = 0; i < newTexC.GetGrid().GetOuterVolume() * newTexC.GetNumChannel(); i++) {
			newTexC.At<double>(i) = UCommon::Clamp<double>(newTexC.At<double>(i) * 0.5 + 0.5, 0, 1);
		}
		newTexC = newTexC.ToUint8();
		UMBD::SaveTex2DAsPNG(newTexC, (name + std::to_string(iterations) + ".png").c_str(), 0, 0, newTexC.GetNumChannel());
	}
}

void GetScaleGradCTex(const Tensor& tensorGradC2, UMBD::FGrid GridC, int64_t BlockSize, int64_t BlockChannels, at::Device device, std::string name, uint64_t iterations, bool exception = false)
{
	if (iterations < 10 || iterations % 50 == 1 || exception)
	{
		UMBD::FTex3D ScaleGradC(GridC, BlockChannels, UMBD::EElementType::Float);
		BlockToTex(ScaleGradC, tensorGradC2, BlockSize, BlockChannels);
		UMBD::FTex3D N0ScaleGradC(GridC, BlockChannels, UMBD::EElementType::Float);
		UMBD::FTex3D P0ScaleGradC(GridC, BlockChannels, UMBD::EElementType::Float);
		float maxmum = (float)1e-10, minmum = (float)1e10;
		int maxi = 0;
		for (const auto& point : GridC)
		{
			for (int channel = 0; channel < ScaleGradC.GetNumChannel(); ++channel)
			{
				uint64_t i = GridC.GetIndex(point) * ScaleGradC.GetNumChannel() + channel;
				P0ScaleGradC.At<float>({ point.X,point.Y,point.Z }, channel) = UCommon::Clamp<float>(log10(ScaleGradC.At<float>(i)) * 50.f / 255.f, 0, 1);
				N0ScaleGradC.At<float>({ point.X,point.Y,point.Z }, channel) = UCommon::Clamp<float>(-log10(ScaleGradC.At<float>(i)) * 50.f / 255.f, 0, 1);
			}
		}
		P0ScaleGradC = P0ScaleGradC.ToUint8();
		N0ScaleGradC = N0ScaleGradC.ToUint8();
		UMBD::SaveTex2DAsPNG(P0ScaleGradC, (name + "P0_"+std::to_string(iterations) + ".png").c_str(), 0, 0, P0ScaleGradC.GetNumChannel());
		UMBD::SaveTex2DAsPNG(N0ScaleGradC, (name + "N0_"+std::to_string(iterations) + ".png").c_str(), 0, 0, N0ScaleGradC.GetNumChannel());
		UMBD::SaveTex2DAsPNG(P0ScaleGradC, (name + "P1_"+std::to_string(iterations) + ".png").c_str(), 1, 0, P0ScaleGradC.GetNumChannel());
		UMBD::SaveTex2DAsPNG(N0ScaleGradC, (name + "N1_"+std::to_string(iterations) + ".png").c_str(), 1, 0, N0ScaleGradC.GetNumChannel());
	}
}

bool FNerualFirstOrderFunction::Evaluate(const double* const parameters, double* cost, double* gradient) const
{
	if (Data.Iterations % 100 == 0)
		cout << Data.Iterations << endl;
	UMBD::FTex3D GradTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, gradient + GridB.GetOuterVolume() * L * D);
	const UMBD::FTex3D ParamTexB(GridB, L * D, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, const_cast<double*>(parameters));
	const UMBD::FTex3D ParamTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, const_cast<double*>(parameters) + ParamTexB.GetNumElements());

	double* newparameters = new double[ParamTexB.GetNumElements() + ParamTexC.GetNumElements()];
	memcpy_s(newparameters, sizeof(double) * (ParamTexB.GetNumElements() + ParamTexC.GetNumElements()), parameters, sizeof(double) * (ParamTexB.GetNumElements() + ParamTexC.GetNumElements()));
	UMBD::FTex3D newParamTexC(GridC, L, UMBD::EOwnership::DoNotTakeOwnership, UMBD::EElementType::Double, const_cast<double*>(newparameters) + ParamTexB.GetNumElements());
	//GetEvaluateCTex(newParamTexC, "C/C_MBD_", Data.Iterations);
	torch::Tensor tensorC = TexToBlock(newParamTexC, Data.BlockSize, L, Data.device);
	tensorC = Data.Compressor->forward(tensorC);
	BlockToTex(newParamTexC, tensorC, Data.BlockSize, L);
	//GetEvaluateCTex(newParamTexC, "C/C_BC1_", (int)Data.Iterations);

	UMBD::FFirstOrderFunction::EvaluateF(newparameters, cost, gradient);
	delete[] newparameters;

	//GetGradCTex(GradTexC, "GradC/GradC_MBD_", Data.Iterations);
	torch::Tensor tensorGradC = TexToBlock(GradTexC, Data.BlockSize, L, Data.device);
	//Tensor tensorGradC1 = tensorGradC;
	tensorGradC = Data.Compressor->backward(tensorGradC);
	//Tensor tensorGradC2 = (tensorGradC.abs() / (tensorGradC1.abs() + 1e-8));
	//GetScaleGradCTex(tensorGradC2, GridC, Data.BlockSize, L, Data.device, "ScaleGradC/ScaleGradC_", Data.Iterations, false);
	BlockToTex(GradTexC, tensorGradC, Data.BlockSize, L);
	//GetGradCTex(GradTexC, "GradC/GradC_BC1_", Data.Iterations);
	UMBD::FFirstOrderFunction::EvaluateNorm(parameters, cost, gradient);
	return true;
}

UMBD::FFirstOrderFunction* FNeuralSolver::CreateFirstOrderFunction(
	double Theta,
	const double* ThetaWeights,
	const double* CustomWeights,
	const double* BRanges,
	bool bCGradientWall,
	const UMBD::FTex3D* TexF,
	UMBD::FGrid GridB,
	UMBD::FGrid GridC,
	uint64_t L,
	uint64_t D,
	bool bDebugInfo)
{
	return new FNerualFirstOrderFunction(
		GetThreadPool(),
		Theta,
		ThetaWeights,
		CustomWeights,
		BRanges,
		bCGradientWall,
		TexF,
		GridB,
		GridC,
		L,
		D,
		bDebugInfo,
		&FunctionData);
}
