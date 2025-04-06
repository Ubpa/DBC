#pragma once
#include <UMBD/UMBDSolver.h>
#include <torch/torch.h>
#include <NeuralAidedMBD/BC7.h>
struct DTBC_config
{
	DTBC_config(int run_mode = 1, int refinecount = 0, int epoch = 6000, float lr = 0.1f, Compressor::QuantizeMode quantizeMode = Compressor::QuantizeMode::None, Compressor::OptimizeMode optimizeMode = Compressor::OptimizeMode::DTBC, BC7::Mode7Type mode7Type = BC7::Mode7Type::MoP, bool* use_mode = nullptr)
	{
		_run_mode = run_mode;
		_refinecount = refinecount;
		_epoch = epoch;
		_lr = lr;
		_quantizeMode = quantizeMode;
		_optimizeMode = optimizeMode;
		_mode7Type = mode7Type;
		if (use_mode == nullptr)
		{
			for (int i = 0; i < 8; ++i)
				_use_mode[i] = 0;
		}
		else
		{
			for (int i = 0; i < 8; ++i)
				_use_mode[i] = use_mode[i];
		}
	}
	int _run_mode;
	int _refinecount;
	int _epoch;
	float _lr;
	Compressor::QuantizeMode _quantizeMode;
	Compressor::OptimizeMode _optimizeMode;
	BC7::Mode7Type _mode7Type;
	bool _use_mode[8];
};

void Bc7e(UMBD::FTex3D& Tex);
/*[h, w, c]*/
torch::Tensor Bc7e(torch::Tensor src);
float EvaluateError(const UMBD::FTex3D& f, const UMBD::FTex3D& approx_f);
UMBD::FTex3D GetErrorTex(const UMBD::FTex3D& f, const UMBD::FTex3D& approx_f, float MaxError);
UMBD::FTex3D GetSHVisTex(const UMBD::FTex3D& f);
void GetCompTex(UMBD::FTex3D& tex1, UMBD::FTex3D& tex2, std::string name);

//[n,b*b,c]
torch::Tensor TexToBlock(const UMBD::FTex3D& Tex, int64_t BlockSize, int64_t BlockChannels, at::Device device);
void BlockToTex(UMBD::FTex3D& Tex, const torch::Tensor& blockin, int64_t BlockSize, int64_t BlockChannels);

bool IsFileExist(const char* filepath);

std::string ToString(const std::vector<float> Values, const std::string& Split);

void SaveStringToFile(const std::string& String, const char* FilePath);

inline void ReadIndices(std::vector< uint32_t>& vec, std::string path)
{
	FILE* fp = NULL;
	fopen_s(&fp, path.c_str(), "r");
	if (fp == NULL)
	{
		std::cout << "Read " << path << " error" << std::endl;
		return;
	}
	int num = 0;
	fscanf_s(fp, "%d\n", &num);
	vec.resize(num);
	for (int i = 0; i < num; i++)
	{
		fscanf_s(fp, "%I32u\n", &vec[i]);
	}
	fclose(fp);
}

inline void WriteFloat(int epoch, float number, std::string path)
{
	FILE* fp = NULL;
	fopen_s(&fp, path.c_str(), "a");
	if (fp == NULL)
	{
		std::cout << "Write " << path << " error" << std::endl;
		return;
	}
	fprintf_s(fp, "%d %f\n", epoch, number);
	fclose(fp);
}

inline void TxTClear(std::string path)
{
	FILE* fp = NULL;
	fopen_s(&fp, path.c_str(), "w");
	if (fp == NULL)
	{
		std::cout << "Clear " << path << " error" << std::endl;
		return;
	}
	fclose(fp);
}
