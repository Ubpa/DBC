#pragma once
#include <torch/torch.h>
#include <DBC/BC7.h>
#include <DBC/BC6.h>
#include <DBC/nvtt.h>
#include <io.h>

using std::cout;
using std::endl;
using torch::Tensor;

inline Tensor TensorToBlock(Tensor tensor/*[n, c, h, w]*/, int block_size)
{
	torch::nn::Unfold unfold(torch::nn::UnfoldOptions({ block_size, block_size }).stride(block_size));
	tensor = unfold(tensor)//[n,c*b*b,L]
		.reshape({ tensor.size(0),tensor.size(1),block_size * block_size,-1 })//[n,c,b*b,L]
		.permute({ 0,3,2,1 })
		.reshape({ -1,block_size * block_size,tensor.size(1) });//[N,b*b,c]
	return tensor;//[N,b*b,c]
}
inline Tensor BlockToTensor(Tensor tensor/*[N, b*b, c]*/, int block_size, torch::IntArrayRef tensor_size/*[n, c, h, w]*/)
{
	torch::nn::Fold fold(torch::nn::FoldOptions({ tensor_size[2],tensor_size[3] }, { block_size, block_size }).stride(block_size));
	tensor = tensor.permute({ 2,1,0 }).reshape({ -1, tensor.size(0) });//[c*b*b, N]
	tensor = fold(tensor).unsqueeze(0);
	return tensor;//[n, c, h, w]
}

struct DBC_config
{
	DBC_config(at::DeviceType device = at::kCPU, float lr = 0.1f, Compressor::QuantizeMode quantizeMode = Compressor::QuantizeMode::None, Compressor::OptimizeMode optimizeMode = Compressor::OptimizeMode::DBC, int encode_config_selection_Type = 1/*MoP*/, bool* use_mode = nullptr, std::string codec_name = "BC6", int Ns = 2, int Nr = 2)
	{
		_lr = lr;
		_quantizeMode = quantizeMode;
		_optimizeMode = optimizeMode;
		_codec_name = codec_name;
		if (_codec_name == "BC6")
			_bc6_mode1To10Type = (BC6::Mode1To10Type)encode_config_selection_Type;
		else //BC7
			_bc7_mode7Type = (BC7::Mode7Type)encode_config_selection_Type;
		_device = device;
		_Ns = Ns;
		_Nr = Nr;
		if (use_mode == nullptr)
		{
			for (int i = 0; i < 8; ++i)
				_bc7_use_mode[i] = 0;
			for (int i = 0; i < 2; ++i)
				_bc6_use_mode[i] = 0;
		}
		else
		{
			if (_codec_name == "BC6")
			{
				for (int i = 0; i < 2; ++i)
					_bc6_use_mode[i] = use_mode[i];
			}
			else //BC7
			{
				for (int i = 0; i < 8; ++i)
					_bc7_use_mode[i] = use_mode[i];
			}
		}
	}
	float _lr;
	Compressor::QuantizeMode _quantizeMode;
	Compressor::OptimizeMode _optimizeMode;
	BC7::Mode7Type _bc7_mode7Type;
	BC6::Mode1To10Type _bc6_mode1To10Type;
	bool _bc7_use_mode[8];
	bool _bc6_use_mode[2];
	std::string _codec_name;
	at::DeviceType _device;
	int _Ns;
	int _Nr;
};

/*[c,d,h,w]*/
torch::Tensor nvtt_bc6(torch::Tensor src/*[c,d,h,w]*/);
/*[c,d,h,w]*/
torch::Tensor nvtt_bc7(torch::Tensor src/*[c,d,h,w]*/);

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

inline void printlog(const char* output)
{
	printf(output);
	int fd = _fileno(stderr);
	if (!_isatty(fd))
	{
		fprintf(stderr, output);
		fflush(stderr);
	}
}