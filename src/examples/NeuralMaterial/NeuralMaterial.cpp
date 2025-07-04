#include "NeuralMaterial.h"
#include <DBC/stb_image.h>
#include <DBC/stb_image_write.h>
#include <iostream>
#include <string>
#include <cmath>
#include <tuple>
#include <filesystem>
#include <cassert>
#include <typeinfo>
using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;

NeuralMaterial::NeuralMaterial(DBC_config config,string objectname,int featuresize)
{
	_objectname = objectname;
	_config = config;
	_FeatureSize = featuresize;
	if (config._codec_name == "BC6")
		_compressor = new BC6(config._device, config._bc6_use_mode, config._quantizeMode, config._optimizeMode, config._bc6_mode1To10Type, config._Ns, config._Nr);
	else //BC7
		_compressor = new BC7(config._device, config._bc7_use_mode, config._quantizeMode, config._optimizeMode, config._bc7_mode7Type, config._Ns, config._Nr);

}

std::tuple<Tensor,Tensor> NeuralMaterial::getBatch(int batch_size, int tile_size, int patch_size, BatchMode batchmode)
{
	Tensor batch_grid, batch_tex;
	if (batchmode == BatchMode::Rand)
	{
		uint64_t seed = torch::randint(16, { 1 }).squeeze().item().toUInt64();
		torch::Generator gen = at::detail::createCPUGenerator();
		gen.set_current_seed(seed);
		Tensor x = torch::rand({ batch_size, tile_size,tile_size }, gen, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(_config._device);
		seed = torch::randint(16, { 1 }).squeeze().item().toLong();
		gen.set_current_seed(seed);
		Tensor y = torch::rand({ batch_size, tile_size,tile_size }, gen, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(_config._device);
		batch_grid = torch::stack({ x, y }, 3);//[batch_size, tile_size, tile_size, 2]
	}
	else if (batchmode == BatchMode::MeshGrid)
	{
		Tensor x = torch::arange(0, tile_size, 1, torch::TensorOptions().device(_config._device).dtype(torch::kFloat32).requires_grad(false)) / (tile_size - 1);
		Tensor y = torch::arange(0, tile_size, 1, torch::TensorOptions().device(_config._device).dtype(torch::kFloat32).requires_grad(false)) / (tile_size - 1);
		std::vector<Tensor>  meshgrid = torch::meshgrid({ x,y }, "xy");
		batch_grid = torch::stack({ meshgrid[0], meshgrid[1] }, 2).unsqueeze(0);//[1, tile_size, tile_size, 2]
	}
	batch_grid = 2 * batch_grid - 1;
	batch_tex = torch::nn::functional::grid_sample(
		_train_tex.broadcast_to({ batch_grid.size(0),_train_tex.size(1),_train_tex.size(2),_train_tex.size(3) }),
		batch_grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kBorder).align_corners(true));
	if (patch_size)
	{
		batch_tex = batch_tex.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).reshape({ batch_tex.size(0),batch_tex.size(1),-1,patch_size, patch_size }).permute({ 0,2,1,3,4 }).reshape({ -1,batch_tex.size(1),patch_size,patch_size });//[n,c,p,p]
		batch_grid = batch_grid.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, batch_grid.size(3), batch_grid.size(3)).reshape({ -1,patch_size,patch_size,batch_grid.size(3) });//[n,p,p,2]
	}
	return { batch_tex,batch_grid };
}

const float max_pixel = 255.0;

tuple<int, int, int, Tensor> ImageToTensor(const char* image_path) {
	int iw, ih, n;
	unsigned char* idata = stbi_load(image_path, &iw, &ih, &n, 0);
	float* data = new float[iw * ih * n];
	for (int i = 0; i < ih; ++i)
		for (int j = 0; j < iw; ++j)
		{
			for (int k = 0; k < n; ++k)
			{
				int pixel_index = i * (iw * n) + j * n + k;
				data[pixel_index] = (float)(idata[pixel_index]) / max_pixel;
			}
		}
	stbi_image_free(idata);
	Tensor output_tensor = torch::from_blob(data, { ih, iw, n }, torch::TensorOptions().dtype(torch::kFloat32));
	return { iw ,ih ,n , output_tensor };
}
void TensorToImage(Tensor image_tensor/*[1,c,h,w]*/, const char* image_path)
{
	image_tensor = image_tensor.squeeze(0).permute({ 1, 2, 0 });//[h,w,c]
	image_tensor = torch::round(image_tensor.mul(255).clamp(0, 255)).to(torch::kUInt8);
	image_tensor = image_tensor.to(torch::kCPU).contiguous();
	unsigned char* data = image_tensor.data_ptr<unsigned char>();
	int height = (int)image_tensor.size(0);
	int width = (int)image_tensor.size(1);
	int channels = (int)image_tensor.size(2);
	printf("write image: %s\n", image_path);
	stbi_write_png(image_path, width, height, channels, data, width * channels);
}
void data_generate(const char *path, const char *save_path) 
{
	tuple<int, int, int, Tensor> res = ImageToTensor(path);
	Tensor x_img_tensor = get<3>(res);
	torch::save(x_img_tensor, save_path);
}

void NeuralMaterial::start()
{
	_data_name = {
		_objectname + "_diff_2k.png",
		_objectname + "_nor_gl_2k.png",
		_objectname + "_arm_2k.png"
	};
	for (int i = 0; i < _data_name.size(); ++i)
	{
		if (!std::filesystem::exists("data/" + _data_name[i] + ".pth"))
		{
			printlog(("get " + _data_name[i] + ".pth\n").c_str());
			data_generate(("image/" + _data_name[i]).c_str(), ("data/" + _data_name[i] + ".pth").c_str());
		}
	}
	std::vector<Tensor> data_tensor;
	data_tensor.resize(_data_name.size());
	_data_channel.resize(_data_name.size());
	for (int i = 0; i < data_tensor.size(); ++i)
	{
		torch::load(data_tensor[i], "data/" + _data_name[i] + ".pth");
		if (i == 1)
		{
			data_tensor[i] = data_tensor[i].index({ Slice(),Slice(),Slice(0,2) });
		}
		_data_channel[i] = (int)data_tensor[i].size(-1);
	}
	_train_tex = torch::cat(data_tensor, -1).to(_config._device).unsqueeze(0);//[1,h,w,c]
	_train_tex = _train_tex.permute({ 0,3,1,2 });//[1,c,h,w]
	_train_tex.set_requires_grad(false);

	int FeatureChannel = 4;//BC7
	if (_config._codec_name == "BC6")
		FeatureChannel = 3;
	int in_channels = FeatureChannel * 4;
	int hidden_channels = 16;
	int num_layers = 2;
	int out_channels = (int)_train_tex.size(1);
	auto model = Net(in_channels, hidden_channels, num_layers, out_channels);
	model->to(_config._device);
	torch::nn::MSELoss loss_fn(torch::nn::MSELossOptions().reduction(torch::kMean));
	auto feature = Feature(_config._device, std::initializer_list<torch::IntArrayRef>({
		{1,FeatureChannel,_FeatureSize,_FeatureSize},
		{1,FeatureChannel,_FeatureSize / 2,_FeatureSize / 2},
		{1,FeatureChannel,_FeatureSize / 4,_FeatureSize / 4},
		{1,FeatureChannel,_FeatureSize / 8,_FeatureSize / 8} }), _compressor);
	feature->to(_config._device);
	torch::optim::Adam *optimizer=new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(_config._lr));
	optimizer->add_param_group(torch::optim::OptimizerParamGroup(feature->_features));
	optimizer->param_groups()[0].options().set_lr(0.01);//pretrain lr
	optimizer->param_groups()[1].options().set_lr(0.01);//pretrain lr

	int epoch1 = 10000;
	int epoch2 = 10000;

	int batch_size = 1;
	string model_file = "pth\\" + std::to_string(_FeatureSize) + " " + _config._codec_name + " " + _objectname + "_10000_model.pth";
	string feature_file = "pth\\" + std::to_string(_FeatureSize) + " " + _config._codec_name + " " + _objectname + "_10000_feature.pth";
	bool exist = std::filesystem::exists(model_file) && std::filesystem::exists(feature_file);
	if (!exist)
		train(model, feature, optimizer, loss_fn, epoch1, batch_size, 100, 5000, EncodeMode::None);
	else
	{
		torch::load(model, model_file);
		torch::load(feature, feature_file);
	}
	delete optimizer;
	optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(_config._lr));
	optimizer->add_param_group(torch::optim::OptimizerParamGroup(feature->_features));
	optimizer->param_groups()[0].options().set_lr(_config._lr);
	optimizer->param_groups()[1].options().set_lr(_config._lr);
	train(model, feature, optimizer, loss_fn, epoch2, batch_size, 1, 50, EncodeMode::DBC);
}

void NeuralMaterial::train(Net& model, Feature& feature, torch::optim::Adam* optimizer, torch::nn::MSELoss& loss_fn, int epoch, int batch_size, int print_interval, int eval_interval, EncodeMode encodeMode)
{
	try {
	string prefix = encodeMode == EncodeMode::DBC ? "DBC_" : (encodeMode == EncodeMode::BC ? "BC_" : "");
	if (encodeMode == EncodeMode::DBC && feature->_compressor->_optimizeMode == Compressor::OptimizeMode::FixConfig)
		prefix = "Fix_" + prefix;
	char targetString[1024];
	feature->_compressor->_init_MoP_weight = true;
	double noisy = 100.f;
	double histloss = 1e20;
	int lr_patience = encodeMode == EncodeMode::DBC ? 40 : 200;
	int lr_interval = 0;
	int cooldown = encodeMode == EncodeMode::DBC ? 0 : 100;
	int error_patience = 5;
	int error_count = 0;
	float error_minmum = 1e15f;
	int error_minmum_index = -1;
	string cost_path = "cost.txt";
	string RMSE_path = "RMSE.txt";
	TxTClear(cost_path);
	TxTClear(RMSE_path);
	torch::optim::ReduceLROnPlateauScheduler reduceLR(*optimizer, torch::optim::ReduceLROnPlateauScheduler::min, 0.9f, lr_patience, 1e-6, torch::optim::ReduceLROnPlateauScheduler::abs, cooldown, {}, 1e-8, true);
	auto batch = getBatch(batch_size, (int)_train_tex.size(2), 0, BatchMode::MeshGrid);
	Tensor batch_tex = get<0>(batch), batch_grid = get<1>(batch);
	for (int i = 0; i <= epoch; ++i)
	{
		auto tmp_start = std::chrono::system_clock::now();
		if(i > 0)
		{
			model->train();
			feature->train();
			optimizer->zero_grad();
		}
		noisy *= std::pow(1e-4 / 100, 1.0 / 4000.0);
		Tensor batch_feature = feature->forward(batch_grid, encodeMode, noisy);
		torch::Tensor pred = model->forward(batch_feature);//[1,c,h,w]
		torch::Tensor loss = loss_fn(pred, batch_tex);
		float loss_item = loss.item().toFloat();
		for (Tensor& feature : feature->_features)
		{
			//if (i % print_interval == 0)
			//{
			//	cout << "max:" << torch::max(feature).item().toFloat()
			//		<< " min:" << torch::min(feature).item().toFloat() << endl;
			//}
			Tensor diff = feature - torch::clamp(feature, -1, 1);
			loss += 0.1 * torch::mean(diff * diff);
		}
		float loss2_item = loss.item().toFloat();
		//if (i % print_interval == 0)
		//{
		//	printf("mse: %f mse2: %f\n",
		//		loss_item, loss2_item);
		//}
		//WriteFloat(i, (float)loss_item, cost_path);
		if (i > 0)
		{
			loss.backward();
			optimizer->step();
			model->eval();
			feature->eval();
		}
		auto time = std::chrono::system_clock::now() - tmp_start;
		float qloss_item = loss_item;
		if (encodeMode == EncodeMode::DBC)
		{
			double lr_model = optimizer->param_groups()[0].options().get_lr();
			double lr_feature = optimizer->param_groups()[1].options().get_lr();
			bool LRstep = feature->_compressor->DBCLRScheduler(qloss_item, histloss, lr_interval, lr_patience);
			if (LRstep)
			{
				optimizer->param_groups()[0].options().set_lr(lr_model * 0.9);
				optimizer->param_groups()[1].options().set_lr(lr_feature * 0.9);
			}
		}
		else
			reduceLR.step(qloss_item);
		feature->_compressor->_init_MoP_weight = false;
		if (i % print_interval == 0 || i == 0)
		{
			double mse = loss_item / batch_tex.size(0);
			double rmse = std::sqrt(mse);
			double psnr = 20. * std::log10(1. / rmse);
			double lr_model = optimizer->param_groups()[0].options().get_lr();
			double lr_feature = optimizer->param_groups()[1].options().get_lr();
			int mstime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
			WriteFloat(i, (float)psnr, cost_path);
			snprintf(targetString, sizeof(targetString), "e: %d mse: %f rmse: %f psnr: %f lr_m: %lf lr_f: %lf t: %d ms noisy: %f ec: %d\n", 
				i, mse, rmse, psnr, lr_model, lr_feature, mstime, noisy, error_count);
			printlog(targetString);
		}
		if (i % eval_interval == 0 || i == epoch)
		{
			if (i > 0 && encodeMode == EncodeMode::None)
			{
				snprintf(targetString, sizeof(targetString), ("pth\\%d " + _config._codec_name + " " + _objectname + "_" + prefix + "%d_model.pth").c_str(), _FeatureSize, i);
				torch::save(model, targetString);
				snprintf(targetString, sizeof(targetString), ("pth\\%d " + _config._codec_name + " " + _objectname + "_" + prefix + "%d_feature.pth").c_str(), _FeatureSize, i);
				torch::save(feature, targetString);
			}
			c10::InferenceMode guard(true);
			feature->_compressor->_updateMoPweight = false;
			Tensor batch_feature = feature->forward(batch_grid, (EncodeMode)((uint32_t)encodeMode | (uint32_t)EncodeMode::BC), 0.);
			Tensor pred = model->forward(batch_feature);
			torch::Tensor loss = loss_fn(pred, batch_tex);
			feature->_compressor->_updateMoPweight = true;
			float error = loss.item().toFloat();
			//WriteFloat(i, (float)error, RMSE_path);
			if (error < error_minmum)
			{
				if (i > 0)
				{
					int now_channel = 0;
					for (int j = 0; j < _data_name.size(); ++j)
					{
						snprintf(targetString, sizeof(targetString), ("pred\\%d " + _config._codec_name + " %d %d " + prefix + "%d_pred_" + _data_name[j]).c_str(), _FeatureSize, _config._Ns, _config._Nr, i);
						Tensor imageTensor = pred.index({ Slice(),Slice(now_channel,now_channel + _data_channel[j]) });
						if (j == 1)
							imageTensor = torch::cat({ imageTensor,
								torch::sqrt(1 - torch::pow((imageTensor * 2 - 1).index({ Slice(),Slice(0,1) }), 2) - torch::pow((imageTensor * 2 - 1).index({ Slice(),Slice(1,2) }), 2)) * 0.5 + 0.5 }, 1);
						TensorToImage(imageTensor, targetString);
						now_channel += _data_channel[j];
					}
					snprintf(targetString, sizeof(targetString), ("pth\\%d " + _config._codec_name + " %d %d " + _objectname + "_"+prefix + "%d_model.pth").c_str(), _FeatureSize, _config._Ns, _config._Nr, i);
					torch::save(model, targetString);
					snprintf(targetString, sizeof(targetString), ("pth\\%d " + _config._codec_name + " %d %d " + _objectname + "_"+prefix + "%d_feature.pth").c_str(), _FeatureSize, _config._Ns, _config._Nr, i);
					torch::save(feature, targetString);
					error_count = 0;
					error_minmum = loss.item().toFloat();
					error_minmum_index = i;
				}
			}
			else
			{
				error_count++;
				if (error_patience >= 0 && error_count >= error_patience)
				{
					printlog("reach error patience in final pass\n");
					break;
				}
			}
			double mse = error;
			double rmse = std::sqrt(mse);
			double psnr = 20. * std::log10(1. / rmse);
			double mse_minmum = error_minmum;
			double rmse_minmum = std::sqrt(mse_minmum);
			double psnr_minmum = 20. * std::log10(1. / rmse_minmum);
			WriteFloat(i, (float)psnr, RMSE_path);
			snprintf(targetString, sizeof(targetString), "Test: mse(%f) rmse(%f) psnr(%f) psnr_minmum(%f) @%d\n",
				mse, rmse, psnr, psnr_minmum, error_minmum_index);
			printlog(targetString);
		}
	}
	}
	catch (const std::exception& e) {
		printlog("Error during train:\n");
		printlog(e.what());
		return;
	}
}