#pragma once
#include <torch/torch.h>
#include <NeuralAidedMBD/Compressor.h>
#include <NeuralAidedMBD/BC7.h>
#include <NeuralAidedMBD/BC6.h>
#include <NeuralAidedMBD/Utils.h>

using std::cout;
using std::endl;
using std::string;
using torch::Tensor;
using namespace torch::indexing;
struct NetImpl : torch::nn::Module {
	NetImpl(int in_channels, int hidden_channels, int num_layers, int out_channels) {
		_in_channels = in_channels;
		_hidden_channels = hidden_channels;
		_num_layers = num_layers;
		_out_channels = _out_channels;

		layers->push_back(torch::nn::Linear(in_channels, hidden_channels));
		layers->push_back(torch::nn::ReLU());
		for (int i = 0; i < num_layers - 2; ++i)
		{
			layers->push_back(torch::nn::Linear(hidden_channels, hidden_channels));
            layers->push_back(torch::nn::ReLU());
		}
		layers->push_back(torch::nn::Linear(hidden_channels, out_channels));

		layers = register_module("layers", layers);
		for (auto m : this->modules(false))
		{
			if (m->name() == "torch::nn::BatchNorm2dImpl")
			{
				printf("init the batchnorm2d parameters.\n");
				auto spBatchNorm2d = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m);
				torch::nn::init::constant_(spBatchNorm2d->weight, 1);
				torch::nn::init::constant_(spBatchNorm2d->bias, 0);
			}
			else if (m->name() == "torch::nn::LinearImpl")
			{
			  printf("init the Linear parameters.\n");
			  auto spLinear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(m);
			}
		}
	} 
	torch::Tensor forward(torch::Tensor x/*[n,c,h,w]*/) {
		x = x.permute({ 0,2,3,1 });//[n,h,w,c]
		x = layers->forward(x);
		x = x.permute({ 0,3,1,2 });//[n,c,h,w]
		return x;
	}
	torch::nn::Linear fc0{ nullptr }, fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
	torch::nn::BatchNorm2d bn0{ nullptr }, bn1{ nullptr }, bn2{ nullptr };
	int _in_channels, _hidden_channels, _num_layers, _out_channels;
	torch::nn::Sequential layers;
};
TORCH_MODULE(Net);

struct FeatureImpl : torch::nn::Module {
public:
	FeatureImpl(at::DeviceType device, std::vector<torch::IntArrayRef> feature_size, Compressor* compressor = nullptr)
	{
		_device = device;
		_compressor = compressor;
		char targetString[256];
		for (int i = 0; i < feature_size.size(); ++i)
		{
			_features.push_back((torch::rand(feature_size[i], torch::TensorOptions().dtype(torch::kFloat32).device(_device)) * 2 - 1).clone().detach().set_requires_grad(true));
			snprintf(targetString, sizeof(targetString), "feature %d", i);
			register_parameter(targetString, _features[i], true);
		}
		bn0 = register_module("bn0", torch::nn::BatchNorm2d(4));
	}
	torch::Tensor forward(torch::Tensor batch_grid/*[n,h,w,2]*/, EncodeMode encodeMode, double noisy)
	{
		std::vector<torch::Tensor> tmp_features = _features; //[1,c,h,w]
		if (encodeMode != EncodeMode::None)
		{
			if ((uint32_t)encodeMode & (uint32_t)EncodeMode::DTBC)
			{
				for (auto& feature : tmp_features)
					feature = TensorToBlock(feature, _BlockSize);//[N,b*b,c]
				Tensor blockfeature = torch::cat(tmp_features, 0);
				Tensor DTBC_blockfeature = _compressor->DTBCcodec(blockfeature, noisy);//[N,b*b,c]:[-1,1]
				int prefixsum_blockcount = 0;
				for (int i = 0; i < tmp_features.size(); ++i)
				{
					int blockcount = (int)tmp_features[i].size(0);
					tmp_features[i] = BlockToTensor(DTBC_blockfeature.index({ Slice(prefixsum_blockcount,prefixsum_blockcount + blockcount) }), _BlockSize, _features[i].sizes());//[n, c, h, w]
					prefixsum_blockcount += blockcount;
				}
			}

			if ((uint32_t)encodeMode & (uint32_t)EncodeMode::BC)
			{
				if (typeid(*_compressor) == typeid(BC6))
				{
					for (auto& feature : tmp_features)
					{
						auto dtype = feature.dtype();
						feature = feature.squeeze().unsqueeze(1);//[c,1,h,w]
						feature = nvtt_bc6(feature).to(_device).to(dtype).squeeze().unsqueeze(0);//[1,c,h,w]
					}
				}
				else //BC7
				{
					for (auto& feature : tmp_features)
					{
						auto dtype = feature.dtype();
						feature = feature.squeeze().permute({ 1,2,0 });//[h,w,c]:[-1,1]
						feature = torch::round(torch::clamp((feature + 1) / 2.f, 0.f, 1.f) * 255.f).to(torch::kUInt8);//[h,w,c]:[0,255]
						//feature = Bc7e(feature).to(_device).permute({ 2,0,1 }).unsqueeze(0).to(dtype);//[1,c,h,w]:[0,255]
						feature = nvtt_bc7(feature.permute({ 2,0,1 }).unsqueeze(1)).to(_device).permute({ 1,0,2,3 }).to(dtype);//[1,c,h,w]:[0,255]
						feature = (feature / 255.f) * 2.f - 1.f;//[1,c,h,w]:[-1,1]
					}
				}
			}
		}
		for (int i = 0; i < tmp_features.size(); ++i)
		{
			tmp_features[i] = torch::nn::functional::grid_sample(
				tmp_features[i].broadcast_to({ batch_grid.size(0),tmp_features[i].size(1),tmp_features[i].size(2),tmp_features[i].size(3) }),
				batch_grid,
				torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true));//[n,c,h,w]
		}
		Tensor batch_feature = torch::cat(tmp_features, 1);//[n,sum_c,h,w]
		return batch_feature;
	}
	std::vector<torch::Tensor> _features;//[1,c,h,w]
	at::DeviceType _device;
	torch::nn::BatchNorm2d bn0{ nullptr };
	Compressor* _compressor;
	int _BlockSize = 4;
	int _BlockChannels = 4;
};
TORCH_MODULE(Feature);

class NeuralMaterial
{
public:
	enum class BatchMode
	{
		Rand,
		MeshGrid,
	};
	NeuralMaterial(DTBC_config config, int pretain, string objectname,int nm_vaild, string Fix_DTBC_best_epoch,string DTBC_best_epoch, int featuresize);
	~NeuralMaterial() { delete _compressor; }
	std::tuple<Tensor, Tensor> getBatch(int batch_size, int tile_size, int patch_size, BatchMode batchmode = BatchMode::Rand);
	void start();
	void train(Net& model, Feature& feature, torch::optim::Adam* optimizer, torch::nn::MSELoss& loss_fn, int epoch, int batch_size, int print_interval, int eval_interval, EncodeMode encodeMode);
	void valid(Net& model, Feature& feature, torch::nn::MSELoss& loss_fn, int batch_size, EncodeMode encodeMode);
	Tensor _train_tex;//[1,c,h,w]
	DTBC_config _config;
	Compressor* _compressor;
	std::vector<string> _data_name;
	std::vector<int> _data_channel;
	int _pretain;
	string _objectname;
	int _vaild;
	string _Fix_DTBC_best_epoch;
	string _DTBC_best_epoch;
	int _FeatureSize;
};
