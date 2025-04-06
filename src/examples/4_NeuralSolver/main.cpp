#include "NeuralAidedMBD/stb_image.h"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <cmath>
#include <tuple>
//#include <stdio.h>
//#include <cassert>
//#include <typeinfo>
//#include "tqdm.h"
#include <NeuralAidedMBD/BC1.h>
#include <NeuralAidedMBD/BC7.h>
#include <NeuralAidedMBD/NeuralSolver.h>
#include <UMBDIO.h>
#include "rdo_bc_encoder.h"
#include <NeuralAidedMBD/Utils.h>
#include <NeuralAidedMBD/NeuralMaterial.h>
//#include <cublas_v2.h>
using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;

const float max_pixel = 255.0;
at::DeviceType device;
void GetCTex(UMBD::FCompressedData& compressedData, std::string name)
{
	UMBD::FTex3D ParamTexC = compressedData.GetC();
	ParamTexC = ParamTexC.ToUint8();
	for (int i = 0; i < ParamTexC.GetGrid().Depth; ++i)
	{
		UMBD::SaveTex2DAsPNG(ParamTexC, (name +"_"+ std::to_string(i) + ".png").c_str(), i, 0, ParamTexC.GetNumChannel());
	}
}

UMBD::FCompressedData MBDSolve(const UMBD::FSolverConfig& config, const UMBD::FTex3D& f, FNeuralSolver& solver, uint64_t L, const UMBD::FGrid& GridB, bool NeuralSolver, bool compress)
{
	const uint64_t D = f.GetNumChannel();
	const UMBD::FGrid GridC = f.GetGrid();
	UMBD::FCompressedData compressedData(D, L, GridB, GridC, UMBD::EElementType::Float);

	if (NeuralSolver)
	{
		UMBD::FSolverConfig NonOptimizeConfig = config;
		NonOptimizeConfig.bDebugInfo = false;
		NonOptimizeConfig.bRunOptimization = false;
		std::vector<double> ThetaWeights(L, 0.0);
		NonOptimizeConfig.ThetaWeights = ThetaWeights.data();
		UMBD::Solve(NonOptimizeConfig, f, compressedData);
		compressedData = UMBD::LoadCompressedData("41130 MBD_2048.bin");
		//UMBD::SaveCompressedData(compressedData, ("MBD_"+std::to_string(NonOptimizeConfig.MaxNumIterations) + ".bin").c_str());

		UMBD::FCeresSolver BasicSolver;
		UMBD::FFirstOrderFunction* mbdf = BasicSolver.CreateFirstOrderFunction(0.0, ThetaWeights.data(), nullptr, nullptr, false, &f, GridB, GridC, L, D, false);
		
		//solver.FunctionData.Compressor->MBDOptim(compressedData, mbdf, f);
		//solver.FunctionData.Compressor->BC1DOptim(compressedData, mbdf, f);
		solver.FunctionData.Compressor->BCOptim(compressedData, mbdf, f);
		//solver.FunctionData.Compressor->BC7DOptim(compressedData, mbdf, f);

		BasicSolver.DestroyFirstOrderFunction(mbdf);
	}
	else
		UMBD::Solve(config, f, compressedData);

	std::string CompressedDataPathName = "MBD_BC_" + std::to_string(solver.FunctionData.Iterations) + ".bin";
	std::cout << "Save MBD CompressedData to " << CompressedDataPathName << std::endl;
	UMBD::SaveCompressedData(compressedData, CompressedDataPathName.c_str());
	//compressedData = UMBD::LoadCompressedData("MBD3_5329.bin");
	//compressedData = UMBD::LoadCompressedData("MBD3_adam_591.bin");
	//compressedData = UMBD::LoadCompressedData("MBD3_DBC1_980.bin");
	//compressedData = UMBD::LoadCompressedData("MBD4_adam_2000.bin");
	//compressedData = UMBD::LoadCompressedData("MBD_DBC_1.bin");
	UMBD::FTex3D& ParamTexC = compressedData.GetC();
	double minimum = std::numeric_limits<double>::max(), maximum = -std::numeric_limits<double>::max();
	for (size_t i = 0; i < ParamTexC.GetNumElements(); i++) {
		double val = ParamTexC.GetDouble(i);
		minimum = std::min(minimum, val);
		maximum = std::max(maximum, val);
		ParamTexC.SetDouble(i, std::clamp(val, 0., 1.));
	}
	cout << "minimum: " << minimum << ", maximum: " << maximum << endl;
	GetCTex(compressedData,"CTexSrc");

	if (compress)
	{
#if 0
		UMBD::FTex3D& ParamTexC = compressedData.GetC();
		Tensor tensorC = TexToBlock(ParamTexC, solver.FunctionData.BlockSize, L);
		//tensorC = tensorC * 0.5 + 0.5;
		//cout << tensorC.sizes() << endl;
		//cout << tensorC << endl;
		//cout << tensorC.min() << endl << tensorC.max() << endl;
		//cout << tensorC[1000] << endl;
		tensorC = solver.FunctionData.Compressor->forward(tensorC);
		//cout << tensorC.min() << endl << tensorC.max() << endl;
		//cout << tensorC[1000] << endl;
		//tensorC = (tensorC - 0.5) * 2;
		BlockToTex(ParamTexC, tensorC, solver.FunctionData.BlockSize, L);
		GetCTex(compressedData,"CTexCom");
#else
		UMBD::FTex3D& ParamTexC = compressedData.GetC();
		Bc7e(ParamTexC);
		GetCTex(compressedData,"CTexCom");
#endif
	}
	else
	{
		UMBD::FTex3D& ParamTexC = compressedData.GetC();
		for (uint64_t i = 0; i < ParamTexC.GetNumElements(); i++)
		{
			float& val = ParamTexC.At<float>(i);
			val = UMBD::ElementUint8ToFloat(UMBD::ElementFloatClampToUint8(val));
		}
	}
	return compressedData;
}

void TestNeuralSolver(int refinecount, int epoch, float lr, Compressor::QuantizeMode quantizeMode, Compressor::OptimizeMode optimizeMode, BC7::Mode7Type mode7Type)
{
	UCommon::FThreadPool ThreadPool;
	UCommon::FThreadPoolRegistry::GetInstance().Register(&ThreadPool);

	UMBD::FCeresSolver CeresSolver;
	UMBD::FSolverRegistry::GetInstance().Register(&CeresSolver);

	bool use_mode4[8] = { 0,0,0,0,1,0,0,0 };
	bool use_mode5[8] = { 0,0,0,0,0,1,0,0 };
	bool use_mode6[8] = { 0,0,0,0,0,0,1,0 };
	bool use_mode7[8] = { 0,0,0,0,0,0,0,1 };
	bool use_mode45[8] = { 0,0,0,0,1,1,0,0 };
	bool use_mode456[8] = { 0,0,0,0,1,1,1,0 };
	bool use_mode4567[8] = { 0,0,0,0,1,1,1,1 };

	FNeuralSolver NeuralSolver;
	NeuralSolver.FunctionData.device = device;
	NeuralSolver.FunctionData.BlockSize = 4;

	//BC1
	//NeuralSolver.FunctionData.Compressor = new BC1(device, refinecount,epoch,lr);
	//const uint64_t L = 3;

	//BC7
	NeuralSolver.FunctionData.Compressor = new BC7(device, refinecount, epoch, lr, use_mode4567, quantizeMode, optimizeMode, mode7Type);
	const uint64_t L = 4;

	//UMBD::FTex3D f = UMBD::LoadTex3D("TexFOutFile.bin");
	//UMBD::FTex3D f = UMBD::LoadTex3D("TexF_30343.bin");
	UMBD::FTex3D f = UMBD::LoadTex3D("TexF_41130.bin");
	UMBD::FGrid GridF((f.GetGrid().Width + 3) / 4 * 4, (f.GetGrid().Height + 3) / 4 * 4, f.GetGrid().Depth);
	cout << "f original grid: " << f.GetGrid().Width << ' ' << f.GetGrid().Height << ' ' << f.GetGrid().Depth << endl;
	cout << "GridF: " << GridF.Width << ' ' << GridF.Height << ' ' << GridF.Depth << endl;
	if (f.GetGrid() != GridF)
	{
		f = f.Resize(GridF);
	}
	UMBD::FGrid GridB(1 + (GridF.Width + 7) / 8, 1 + (GridF.Height + 7) / 8, (1 + (GridF.Depth + 7) / 8));

	UMBD::FSolverConfig config;
	//config.bRunOptimization = false;
	config.bDebugInfo = true;
	config.MaxNumIterations = 2048;//512
	//BC1Solver.FunctionData.MaxNumIterations = (int)config.MaxNumIterations;
	//config.IterationCallback = new INerualIterationCallback(&BC1Solver);

	float ErrorValueScale = 2.f;

	UMBD::FTex3D VisTex = GetSHVisTex(f);
	UMBD::SaveTex2DAsPNG(VisTex, "0_Reference_VisTex.png");

	bool bLoadMode = false;
	UMBD::FCompressedData MBD_compressedData;
	UMBD::FCompressedData MBD_D_compressedData;
	UMBD::FCompressedData MBD_BC1_compressedData;
	UMBD::FCompressedData MBD_DBC1_compressedData;
	UMBD::FCompressedData MBD_BC7_compressedData;
	UMBD::FCompressedData MBD_DBC7_compressedData;
	if (bLoadMode)
	{
		MBD_compressedData = UMBD::LoadCompressedData("MBD_CompressedData.bin");
		//MBD_D_compressedData = UMBD::LoadCompressedData("MBD_D_CompressedData.bin");
		//MBD_BC1_compressedData = UMBD::LoadCompressedData("MBD_BC1_CompressedData.bin");
		//MBD_DBC1_compressedData = UMBD::LoadCompressedData("MBD_DBC1_CompressedData.bin");
		MBD_BC7_compressedData = UMBD::LoadCompressedData("MBD_BC7_CompressedData.bin");
		MBD_DBC7_compressedData = UMBD::LoadCompressedData("MBD_DBC7_CompressedData.bin");
	}
	else
	{
		//MBD_compressedData = MBDSolve(config, f, NeuralSolver, L, GridB, false, false);
		//MBD_D_compressedData = MBDSolve(config, f, NeuralSolver, L, GridB, true, false);
		//MBD_BC1_compressedData = MBDSolve(config, f, NeuralSolver, L, GridB, false, true);
		//MBD_DBC1_compressedData = MBDSolve(config, f, NeuralSolver, L, GridB, true, true);
		//MBD_BC7_compressedData = MBDSolve(config, f, NeuralSolver, L, GridB, false, true);
		MBD_DBC7_compressedData = MBDSolve(config, f, NeuralSolver, L, GridB, true, true);
	}
	
	//GetCTex(MBD_compressedData,"CTex");
	//UMBD::FTex3D MBDerrorTex = GetErrorTex(f, MBD_compressedData.GetApproxF(f.GetGrid()), ErrorValueScale);
	//UMBD::SaveTex2DAsPNG(MBDerrorTex, "MBDL2Tex.png");
	//UMBD::FTex3D MBDVisTex = GetSHVisTex(MBD_compressedData.GetApproxF(f.GetGrid()));
	//UMBD::SaveTex2DAsPNG(MBDVisTex, "1_MBD_VisTex.png");

	//GetCTex(MBD_D_compressedData,"CTex");
	//UMBD::FTex3D MBDDerrorTex = GetErrorTex(f, MBD_D_compressedData.GetApproxF(f.GetGrid()), ErrorValueScale);
	//UMBD::SaveTex2DAsPNG(MBDDerrorTex, "MBDDL2Tex.png");
	//UMBD::FTex3D MBDDVisTex = GetSHVisTex(MBD_D_compressedData.GetApproxF(f.GetGrid()));
	//UMBD::SaveTex2DAsPNG(MBDDVisTex, "MBDDVisTex.png");

	//GetCTex(MBD_BC7_compressedData, "BC7CTex");
	//UMBD::FTex3D BC7errorTex = GetErrorTex(f, MBD_BC7_compressedData.GetApproxF(f.GetGrid()), ErrorValueScale);
	//UMBD::SaveTex2DAsPNG(BC7errorTex, "BC7L2Tex.png");
	//UMBD::FTex3D BC7VisTex = GetSHVisTex(MBD_BC7_compressedData.GetApproxF(f.GetGrid()));
	//UMBD::SaveTex2DAsPNG(BC7VisTex, "3_MBD+BC_VisTex.png");

	//GetCTex(MBD_DBC7_compressedData,"DBC7CTex");
	UMBD::FTex3D DBC7errorTex = GetErrorTex(f, MBD_DBC7_compressedData.GetApproxF(f.GetGrid()), ErrorValueScale);
	UMBD::SaveTex2DAsPNG(DBC7errorTex, "DBC7L2Tex.png");
	UMBD::FTex3D DBC7VisTex = GetSHVisTex(MBD_DBC7_compressedData.GetApproxF(f.GetGrid()));
	UMBD::SaveTex2DAsPNG(DBC7VisTex, "2_MBD+DTBC_VisTex.png");

	UMBD::SaveCompressedData(MBD_compressedData, "MBD_CompressedData.bin");
	//UMBD::SaveCompressedData(MBD_D_compressedData, "MBD_D_CompressedData.bin");
	UMBD::SaveCompressedData(MBD_BC7_compressedData, "MBD_BC7_CompressedData.bin");
	UMBD::SaveCompressedData(MBD_DBC7_compressedData, "MBD_DBC7_CompressedData.bin");

	UMBD::FSolverRegistry::GetInstance().Deregister();
	UCommon::FThreadPoolRegistry::GetInstance().Deregister();
}

void Test(int refinecount, int epoch, float lr, Compressor::QuantizeMode quantizeMode, Compressor::OptimizeMode optimizeMode, BC7::Mode7Type mode7Type)
{
	Tensor data = torch::zeros(at::IntArrayRef({ 16,4 }), torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			//data[i][j] = 1.0 / 15.0 * i;
			data[i][j] = rand() / (float)RAND_MAX;
			//data[i][j] = 1.0;
		}
		//data[i][3] = 1;
		//data[0][3] = 1.01;
		//data[i][3] = rand() / (float)RAND_MAX;;
		//data[i][3] = 1.0 / 15.0 * i;
	}
	Tensor data1 = torch::zeros(at::IntArrayRef({ 16,4 }), torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));
	for (int i = 0; i < 16; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			//data1[i][j] = 1.0 / 15.0 * i;
			data1[i][j] = rand() / (float)RAND_MAX;
			//data1[i][j] = 0.0;
		}
		//data1[i][3] = 1.0 / 15.0 * i;
		//data1[i][3] = 0.5;
	}
	//Bc7e(data);
	data = data.unsqueeze(0);
	//data = data7.unsqueeze(0);
	//data = torch::stack({ data,data1 });
	cout << data << endl;
	Tensor src = data.clone().detach().requires_grad_(false);
	torch::nn::L1Loss loss_fn(torch::nn::L1LossOptions().reduction(torch::kNone));
	torch::nn::MSELoss l2loss_fn(torch::nn::MSELossOptions().reduction(torch::kNone));
	//Tensor target = torch::rand_like(data);
	//Tensor target = torch::tensor({ 1.0, 0.0, 0.0 }, torch::TensorOptions().dtype(torch::kFloat64)).repeat({ 16, 1 });
	Tensor target = data.clone().detach();
	//cout << target << endl;
	//exit(0);
	bool use_mode456[8] = { 0,0,0,0,1,1,1,0 };
	bool use_mode7[8] = { 0,0,0,0,0,0,0,1 };
	bool use_mode6[8] = { 0,0,0,0,0,0,1,0 };
	BC7 enc = BC7(device, refinecount, epoch, lr, use_mode7, quantizeMode, optimizeMode, mode7Type);
	//BC1 enc = BC1(device, refinecount,epoch, lr);
	//target = enc.forward(target).clone().detach();
	//cout << target * 255 << endl;
	for (int i = 0; i < epoch; ++i)
	{
		Tensor dest = enc.forward(src);
		cout << dest << endl;
		//cout << loss_fn(dest.to(at::kCPU), src.index({ Slice(),Slice(),Slice(0,dest.size(2)) }).repeat({ dest.size(0) / data.size(0),1,1 })).mean({ 1,2 }) << endl;
		//cout << l2loss_fn(dest.to(at::kCPU), src.index({ Slice(),Slice(),Slice(0,dest.size(2)) }).repeat({ dest.size(0) / data.size(0),1,1 })).mean({ 1,2 }) << endl;
		//cout << grad3.sizes() << endl << dest.sizes() << endl;
		//Tensor grad = enc.backward(torch::ones_like(dest));
		//cout << grad << endl;
		break;
		//cout << 1 << endl;
		Tensor loss = loss_fn(dest, target).mean();
		cout << loss.item() << endl;
		if (loss.item().toDouble() < 0.01)
		{
			cout << dest * 255 << endl;
			break;
		}
		torch::optim::Adam optimizer({ enc._src }, 0.01);
		optimizer.zero_grad();
		//cout << 3 << endl;
		loss.backward();
		//if (i % 50 == 0)
		//	cout << dest << endl << enc._src.grad() << endl << enc._error << endl;
		//cout << 4 << endl;
		//cout << enc._src.grad() << endl;
		//break;
		optimizer.step();
		src = enc._src.clone().detach().requires_grad_(false);
	}
}

void RGBMOptim(int refinecount, int epoch, float lr, Compressor::QuantizeMode quantizeMode, Compressor::OptimizeMode optimizeMode, BC7::Mode7Type mode7Type)
{

	bool use_mode4[8] = { 0,0,0,0,1,0,0,0 };
	bool use_mode5[8] = { 0,0,0,0,0,1,0,0 };
	bool use_mode6[8] = { 0,0,0,0,0,0,1,0 };
	bool use_mode7[8] = { 0,0,0,0,0,0,0,1 };
	bool use_mode45[8] = { 0,0,0,0,1,1,0,0 };
	bool use_mode456[8] = { 0,0,0,0,1,1,1,0 };
	bool use_mode4567[8] = { 0,0,0,0,1,1,1,1 };

	FNeuralSolver NeuralSolver;
	NeuralSolver.FunctionData.device = device;
	NeuralSolver.FunctionData.BlockSize = 4;

	//BC1
	//NeuralSolver.FunctionData.Compressor = new BC1(device, refinecount,epoch,lr);

	//BC7
	Compressor* compressor = new BC7(device, refinecount, epoch, lr, use_mode4567, quantizeMode, optimizeMode, mode7Type);
	stbi_set_flip_vertically_on_load(true);
	int width, height, nrComponents;
	std::string strHDRName = "dry_orchard_meadow_1k.hdr";
	float* data = stbi_loadf(strHDRName.c_str(), &width, &height, &nrComponents, 0);
	//float* tensor_data = new float[width * height * nrComponents];
	//for (int i = 0; i < height; ++i)
	//	for (int j = 0; j < width; ++j)
	//	{
	//		for (int k = 0; k < nrComponents; ++k)
	//		{
	//			int pixel_index = i * (width * nrComponents) + j * nrComponents + k;
	//			tensor_data[pixel_index] = (float)(data[pixel_index]) / max_pixel;
	//			//cout << data[pixel_index] <<' ';
	//		}
	//		//cout << endl;
	//	}
	//stbi_image_free(data);
	Tensor output_tensor = torch::from_blob(data, { height, width, nrComponents },c10::TensorOptions().dtype(c10::kFloat));
	cout << output_tensor.sizes() << endl;//512,1024,3
	cout << output_tensor[0][0] << endl;
	cout << output_tensor[0][1] << endl;
	cout << output_tensor[0][2] << endl;
}

int main(int argc, char* argv[])
{
	torch::manual_seed(1);
	if (torch::cuda::is_available())
	{
		cout << "use cuda" << endl;
		device = at::kCUDA;
		torch::cuda::manual_seed_all(1);
	}
	else
	{
		cout << "use cpu" << endl;
		device = at::kCPU;
	}

	int run_mode = 1;
	int refinecount = 0;
	int epoch = 6000;
	float lr = 0.01f;
	Compressor::QuantizeMode quantizeMode = Compressor::QuantizeMode::Default;
	Compressor::OptimizeMode optimizeMode = Compressor::OptimizeMode::DTBC;
	BC7::Mode7Type mode7Type = BC7::Mode7Type::MoP;
	int pretain = 1;
	string objectname = "";
	int nm_vaild = 0;
	string Fix_DTBC_best_epoch = "";
	string DTBC_best_epoch = "";
	int argindex = 1;
	if (argc > argindex)
		run_mode = std::atoi(argv[argindex++]);
	if (argc > argindex)
		refinecount = std::atoi(argv[argindex++]);
	if (argc > argindex)
		epoch = std::atoi(argv[argindex++]);
	if (argc > argindex)
		lr = (float)std::atof(argv[argindex++]);
	if (argc > argindex)
		quantizeMode = (Compressor::QuantizeMode)std::atoi(argv[argindex++]);
	if (argc > argindex)
		optimizeMode = (Compressor::OptimizeMode)std::atoi(argv[argindex++]);
	if (argc > argindex)
		mode7Type = (BC7::Mode7Type)std::atoi(argv[argindex++]);
	if (argc > argindex)
		pretain = std::atoi(argv[argindex++]);
	if (argc > argindex)
		nm_vaild = std::atoi(argv[argindex++]);
	if (argc > argindex)
		objectname = argv[argindex++];
	if (argc > argindex)
		Fix_DTBC_best_epoch = argv[argindex++];
	if (argc > argindex)
		DTBC_best_epoch = argv[argindex++];
	cout
		<< "[args]" << endl
		<< "run_mode: " << run_mode << endl
		<< "refinecount : " << refinecount << endl
		<< "epoch : " << epoch << endl
		<< "lr : " << lr << endl
		<< "quantizeMode : " << (uint32_t)quantizeMode << endl
		<< "optimizeMode : " << (uint32_t)optimizeMode << endl
		<< "mode7Type : " << (int)mode7Type << endl
		<< "pretain : " << pretain << endl
		<< "nm_vaild : " << nm_vaild << endl
		<< "objectname : " << objectname << endl
		<< "Fix_DTBC_best_epoch : " << Fix_DTBC_best_epoch << endl
		<< "DTBC_best_epoch : " << DTBC_best_epoch << endl
		;
		
	if (run_mode == 0)
	{
		Test(refinecount, epoch, lr, quantizeMode, optimizeMode, mode7Type);
	}
	else if (run_mode == 1)
	{
		TestNeuralSolver(refinecount, epoch, lr, quantizeMode, optimizeMode, mode7Type);
	}
	else if (run_mode == 2)
	{
		bool use_mode4[8] = { 0,0,0,0,1,0,0,0 };
		bool use_mode5[8] = { 0,0,0,0,0,1,0,0 };
		bool use_mode6[8] = { 0,0,0,0,0,0,1,0 };
		bool use_mode7[8] = { 0,0,0,0,0,0,0,1 };
		bool use_mode45[8] = { 0,0,0,0,1,1,0,0 };
		bool use_mode456[8] = { 0,0,0,0,1,1,1,0 };
		bool use_mode4567[8] = { 0,0,0,0,1,1,1,1 };
		DTBC_config config(run_mode, refinecount, epoch, lr, quantizeMode, optimizeMode, mode7Type, use_mode4567);
		NeuralMaterial nm(device, lr, config, pretain, objectname, nm_vaild, Fix_DTBC_best_epoch, DTBC_best_epoch);
		nm.start();
	}
	else
		cout << "run_mode: " << run_mode << " is unsupported" << endl;
	return 0;
}
