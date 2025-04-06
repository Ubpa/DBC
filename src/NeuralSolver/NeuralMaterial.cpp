#include "NeuralAidedMBD/NeuralMaterial.h"
#include "NeuralAidedMBD/stb_image.h"
#include "NeuralAidedMBD/stb_image_write.h"
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

ExponentialLR::ExponentialLR(torch::optim::Optimizer& optimizer, float gamma) 
	: torch::optim::LRScheduler(optimizer), _gamma(gamma) {}

std::vector<double> ExponentialLR::get_lrs()
{
	std::vector<double> lrs = get_current_lrs();
	for (int i = 0; i < lrs.size(); ++i)
		lrs[i] *= _gamma;
	return lrs;
}

NeuralMaterial::NeuralMaterial(at::DeviceType device, float lr, DTBC_config config, int pretain,string objectname,int nm_vaild,string Fix_DTBC_best_epoch,string DTBC_best_epoch)
{
	_Fix_DTBC_best_epoch = Fix_DTBC_best_epoch;
	_DTBC_best_epoch = DTBC_best_epoch;
	_vaild = nm_vaild;
	_objectname = objectname;
	_pretain = pretain;
	_device = device;
	_lr = lr;
	_config = config;
	_compressor = new BC7(device, config._refinecount, config._epoch, lr, config._use_mode, config._quantizeMode, config._optimizeMode, config._mode7Type);
}

std::tuple<Tensor,Tensor> NeuralMaterial::getBatch(int batch_size, int tile_size, int patch_size, BatchMode batchmode)
{
	Tensor batch_grid, batch_tex;
	if (batchmode == BatchMode::Rand)
	{
		uint64_t seed = torch::randint(16, { 1 }).squeeze().item().toUInt64();
		torch::Generator gen = at::detail::createCPUGenerator();
		gen.set_current_seed(seed);
		Tensor x = torch::rand({ batch_size, tile_size,tile_size }, gen, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(_device);
		seed = torch::randint(16, { 1 }).squeeze().item().toLong();
		gen.set_current_seed(seed);
		Tensor y = torch::rand({ batch_size, tile_size,tile_size }, gen, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)).to(_device);
		batch_grid = torch::stack({ x, y }, 3);//[batch_size, tile_size, tile_size, 2]
	}
	else if (batchmode == BatchMode::MeshGrid)
	{
		Tensor x = torch::arange(0, tile_size, 1, torch::TensorOptions().device(_device).dtype(torch::kFloat32).requires_grad(false)) / (tile_size - 1);
		Tensor y = torch::arange(0, tile_size, 1, torch::TensorOptions().device(_device).dtype(torch::kFloat32).requires_grad(false)) / (tile_size - 1);
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
		//cout << batch_tex.sizes() << endl;
		//cout << batch_grid.sizes() << endl;
	}
	return { batch_tex,batch_grid };
}

const float max_pixel = 255.0;

tuple<int, int, int, Tensor> ImageToTensor(const char* image_path) {
	int iw, ih, n;
	unsigned char* idata = stbi_load(image_path, &iw, &ih, &n, 0);
	//cout << targetString << endl;
	float* data = new float[iw * ih * n];
	for (int i = 0; i < ih; ++i)
		for (int j = 0; j < iw; ++j)
		{
			for (int k = 0; k < n; ++k)
			{
				int pixel_index = i * (iw * n) + j * n + k;
				data[pixel_index] = (float)(idata[pixel_index]) / max_pixel;
				//cout << data[pixel_index] <<' ';
			}
			//cout << endl;
		}
	stbi_image_free(idata);
	Tensor output_tensor = torch::from_blob(data, { ih, iw, n }, torch::TensorOptions().dtype(torch::kFloat32));
	//cout << output_tensor << endl;
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
	/*_data_name = {"Ground068_1K-JPG_Color.png","Ground068_1K-JPG_NormalGL_normed.png","Ground068_1K-JPG_Displacement.png","Ground068_1K-JPG_Roughness.png","Ground068_1K-JPG_AmbientOcclusion.png" };*/
	//string objectname = "Ukulele_01";//"concrete_cat_statue";
	_data_name = {
		_objectname + "_diff_2k.png",
		_objectname + "_nor_gl_2k.png",
		_objectname + "_arm_2k.png"
	};
	for (int i = 0; i < _data_name.size(); ++i)
	{
		if (!std::filesystem::exists("data/" + _data_name[i] + ".pth"))
		{
			cout << "get " + _data_name[i] + ".pth" << endl;
			data_generate(("image/" + _data_name[i]).c_str(), ("data/" + _data_name[i] + ".pth").c_str());
		}
	}
	std::vector<Tensor> data_tensor;
	data_tensor.resize(_data_name.size());
	_data_channel.resize(_data_name.size());
	for (int i = 0; i < data_tensor.size(); ++i)
	{
		torch::load(data_tensor[i], "data/" + _data_name[i] + ".pth");
		//Tensor f = torch::round(data_tensor[i] * 255.f).to(torch::kUInt8);//[h,w,c]:[0,255]
		//f = Bc7e(f).to(data_tensor[i].dtype()) / 255.f;//[h,w,c]:[0,1]
		//cout << _data_name[i]<<": "<<torch::pow(f - data_tensor[i], 2).mean() << endl;
		//TensorToImage(f.unsqueeze(0).permute({ 0,3,1,2 }), ("pred/bc7e_"+ _data_name[i]).c_str());
		if (i == 1)
		{
			data_tensor[i] = data_tensor[i].index({ Slice(),Slice(),Slice(0,2) });
			//data_tensor[i] = torch::sum(torch::pow(data_tensor[i] * 2 - 1, 2), -1);
			//cout << data_tensor[i].index({ Slice(0,4),Slice(0,4) }) << endl;
		}
		_data_channel[i] = (int)data_tensor[i].size(-1);
	}
	_train_tex = torch::cat(data_tensor, -1).to(_device).unsqueeze(0);//[1,h,w,c]
	_train_tex = _train_tex.permute({ 0,3,1,2 });//[1,c,h,w]
	_train_tex.set_requires_grad(false);
	cout << "load pth" << endl;

	int in_channels = 16;
	int hidden_channels = 16;
	int num_layers = 2;
	int out_channels = (int)_train_tex.size(1);
	auto model = Net(in_channels, hidden_channels, num_layers, out_channels);
	model->to(_device);
	torch::nn::MSELoss loss_fn(torch::nn::MSELossOptions().reduction(torch::kMean));
	int FeatureSize = 512;
	auto feature = Feature(_device, std::initializer_list<torch::IntArrayRef>({
		{1,4,FeatureSize,FeatureSize},
		{1,4,FeatureSize / 2,FeatureSize / 2},
		{1,4,FeatureSize / 4,FeatureSize / 4},
		{1,4,FeatureSize / 8,FeatureSize / 8} }), _compressor);
	feature->to(_device);
	torch::optim::Adam *optimizer=new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(_lr));
	optimizer->add_param_group(torch::optim::OptimizerParamGroup(feature->_features));
	optimizer->param_groups()[0].options().set_lr(0.01);
	optimizer->param_groups()[1].options().set_lr(0.01);

	int epoch1 = 10000;
	int epoch2 = 10000;

	int batch_size = 1;
	if (!_vaild)
	{
		if (_pretain)
			train(model, feature, optimizer, loss_fn, epoch1, batch_size, 100, 5000, EncodeMode::None);
		else
		{
			torch::load(model, "pth\\" + _objectname + "_10000_model.pth");
			torch::load(feature, "pth\\" + _objectname + "_10000_feature.pth");
		}
		delete optimizer;
		optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(_lr));
		optimizer->add_param_group(torch::optim::OptimizerParamGroup(feature->_features));
		optimizer->param_groups()[0].options().set_lr(_lr);
		optimizer->param_groups()[1].options().set_lr(_lr);
		train(model, feature, optimizer, loss_fn, epoch2, batch_size, 1, 50, EncodeMode::DTBC);
	}
	else
	{
		torch::load(model, "pth\\" + _objectname + "_10000_model.pth");
		torch::load(feature, "pth\\" + _objectname + "_10000_feature.pth");
		valid(model, feature, loss_fn, batch_size, EncodeMode::BC);

		feature->_compressor->_optimizeMode = Compressor::OptimizeMode::FixConfig;
		feature->_compressor->_init_mode7_weight = true;
		auto batch = getBatch(batch_size, (int)_train_tex.size(2), 0, BatchMode::MeshGrid);
		Tensor batch_tex = get<0>(batch), batch_grid = get<1>(batch);
		Tensor batch_feature = feature->forward(batch_grid, EncodeMode::DTBC, 0.);
		torch::load(model, "pth\\" + _objectname + "_Fix_DTBC_" + _Fix_DTBC_best_epoch + "_model.pth");
		torch::load(feature, "pth\\" + _objectname + "_Fix_DTBC_" + _Fix_DTBC_best_epoch + "_feature.pth");
		feature->_compressor->_init_mode7_weight = false;
		valid(model, feature, loss_fn, batch_size, EncodeMode::DTBC);

		torch::load(model, "pth\\" + _objectname + "_DTBC_" + _DTBC_best_epoch + "_model.pth");
		torch::load(feature, "pth\\" + _objectname + "_DTBC_" + _DTBC_best_epoch + "_feature.pth");
		feature->_compressor->_optimizeMode = Compressor::OptimizeMode::DTBC;
		feature->_compressor->_init_mode7_weight = true;
		valid(model, feature, loss_fn, batch_size, EncodeMode::DTBC);
	}
}

void NeuralMaterial::valid(Net& model, Feature& feature, torch::nn::MSELoss& loss_fn, int batch_size, EncodeMode encodeMode)
{
	string prefix = encodeMode == EncodeMode::DTBC ? "DTBC_" : (encodeMode == EncodeMode::BC ? "BC_" : "");
	if (encodeMode == EncodeMode::DTBC && feature->_compressor->_optimizeMode == Compressor::OptimizeMode::FixConfig)
		prefix = "Fix_" + prefix;
	char targetString[256];
	feature->_compressor->_updateMoPweight = false;
	feature->_roundc = -1.f;
	c10::InferenceMode guard(true);
	auto batch = getBatch(batch_size, (int)_train_tex.size(2), 0, BatchMode::MeshGrid);
	Tensor batch_tex = get<0>(batch), batch_grid = get<1>(batch);
	Tensor batch_feature = feature->forward(batch_grid, (EncodeMode)((uint32_t)encodeMode | (uint32_t)EncodeMode::BC), 0.);
	Tensor pred = model->forward(batch_feature);
	torch::Tensor loss = loss_fn(pred, batch_tex);
	float error = loss.item().toFloat();
	double mse = error / batch_tex.size(0);
	double rmse = std::sqrt(mse);
	double psnr = 20. * std::log10(1 / rmse);
	printf("valid: mse: %f rmse: %f psnr: %f \n", mse, rmse, psnr);
	int now_channel = 0;
	for (int j = 0; j < _data_name.size(); ++j)
	{
		snprintf(targetString, sizeof(targetString), ("pred\\" + prefix + "pred_" + _data_name[j]).c_str());
		Tensor imageTensor = pred.index({ Slice(),Slice(now_channel,now_channel + _data_channel[j]) });
		if (j == 1)
			imageTensor = torch::cat({ imageTensor,
				torch::sqrt(1 - torch::pow((imageTensor * 2 - 1).index({ Slice(),Slice(0,1) }), 2) - torch::pow((imageTensor * 2 - 1).index({ Slice(),Slice(1,2) }), 2)) * 0.5 + 0.5 }, 1);
		TensorToImage(imageTensor, targetString);
		now_channel += _data_channel[j];
	}
}
void NeuralMaterial::train(Net& model, Feature& feature, torch::optim::Adam* optimizer, torch::nn::MSELoss& loss_fn, int epoch, int batch_size, int print_interval, int eval_interval, EncodeMode encodeMode)
{
	try {
	string prefix = encodeMode == EncodeMode::DTBC ? "DTBC_" : (encodeMode == EncodeMode::BC ? "BC_" : "");
	if (encodeMode == EncodeMode::DTBC && feature->_compressor->_optimizeMode == Compressor::OptimizeMode::FixConfig)
		prefix = "Fix_" + prefix;
	char targetString[256];
	feature->_compressor->_init_mode7_weight = true;
	feature->_roundc = 0.f;
	double noisy = 100.f;
	double histloss = 1e20;
	int lr_patience = encodeMode == EncodeMode::DTBC ? 40 : 200;
	int lr_interval = 0;
	int cooldown = encodeMode == EncodeMode::DTBC ? 0 : 100;
	int enter_nextpass = 0;
	int enter_nextpass_patience = 3;
	int pass = 1;
	int maxpass = 2;
	int error_patience = 5;
	int error_count = 0;
	float error_minmum = 1e15f;
	int error_minmum_index = -1;
	string cost_path = prefix + "cost.txt";
	string qcost_path = prefix + "qcost.txt";
	string RMSE_path = prefix + "RMSE.txt";
	TxTClear(cost_path);
	TxTClear(qcost_path);
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
		noisy *= std::pow(1e-4 / 100, 1.0 / 2000.0);
		Tensor batch_feature = feature->forward(batch_grid, encodeMode, noisy);
		torch::Tensor pred = model->forward(batch_feature);//[1,c,h,w]
		torch::Tensor loss = loss_fn(pred, batch_tex);
		float loss_item = loss.item().toFloat();
		WriteFloat(i, (float)loss_item, cost_path);
		if (i > 0)
		{
			loss.backward();
			optimizer->step();
			model->eval();
			feature->eval();
		}
		auto time = std::chrono::system_clock::now() - tmp_start;
		float qloss_item = loss_item;
		if (encodeMode == EncodeMode::DTBC)
		{
			if (pass == 2)
			{
				feature->_compressor->_updateMoPweight = false;
				float originalRoundC = feature->_roundc;
				feature->_roundc = -1.f;
				Tensor batch_feature = feature->forward(batch_grid, (EncodeMode)((uint32_t)encodeMode | (uint32_t)EncodeMode::BC), 0.);
				Tensor pred = model->forward(batch_feature);
				torch::Tensor loss = loss_fn(pred, batch_tex);
				qloss_item = loss.item().toFloat();
				//feature->_compressor->_updateMoPweight = true;
			}
			double lr_model = optimizer->param_groups()[0].options().get_lr();
			double lr_feature = optimizer->param_groups()[1].options().get_lr();
			int LRstep = feature->_compressor->DTBCLRScheduler(feature->_roundc, qloss_item, histloss, cooldown, lr_interval, lr_patience, enter_nextpass, enter_nextpass_patience, pass, maxpass);
			if (LRstep == 1)
			{
				std::cout << "reach patience in first pass" << endl;
				cooldown = 20;
				torch::optim::Adam* tmp = new torch::optim::Adam(optimizer->param_groups(), torch::optim::AdamOptions(std::min(0.008, lr_model)));
				tmp->param_groups()[0].options().set_lr(std::min(0.008, lr_model));
				tmp->param_groups()[1].options().set_lr(std::min(0.008, lr_feature));
				delete optimizer;
				optimizer = tmp;
			}
			else if (LRstep == 2)
			{
				optimizer->param_groups()[0].options().set_lr(lr_model * 0.9);
				optimizer->param_groups()[1].options().set_lr(lr_feature * 0.9);
			}
		}
		else
			reduceLR.step(qloss_item);
		feature->_compressor->_init_mode7_weight = false;
		if (i % print_interval == 0 || i == 0)
		{
			double mse = loss_item / batch_tex.size(0);
			double rmse = std::sqrt(mse);
			double psnr = 20. * std::log10(1. / rmse);
			double lr_model = optimizer->param_groups()[0].options().get_lr();
			double lr_feature = optimizer->param_groups()[1].options().get_lr();
			int mstime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
			printf("e: %d mse: %f rmse: %f psnr: %f lr_m: %lf lr_f: %lf t: %d ms noisy: %f pass: %d enp:%d ec: %d\n",
				i, mse, rmse, psnr, lr_model, lr_feature, mstime, noisy, pass, enter_nextpass, error_count);
		}
		if (i % eval_interval == 0 || i == epoch)
		{
			if (i > 0 && encodeMode == EncodeMode::None)
			{
				snprintf(targetString, sizeof(targetString), ("pth\\" + _objectname + "_"+prefix + "%d_model.pth").c_str(), i);
				torch::save(model, targetString);
				snprintf(targetString, sizeof(targetString), ("pth\\" + _objectname + "_"+prefix + "%d_feature.pth").c_str(), i);
				torch::save(feature, targetString);
			}
			c10::InferenceMode guard(true);
			feature->_compressor->_updateMoPweight = false;
			float originalRoundC = feature->_roundc;
			feature->_roundc = -1.f;
			Tensor batch_feature = feature->forward(batch_grid, (EncodeMode)((uint32_t)encodeMode | (uint32_t)EncodeMode::BC), 0.);
			Tensor pred = model->forward(batch_feature);
			torch::Tensor loss = loss_fn(pred, batch_tex);
			feature->_roundc = originalRoundC;
			int now_channel = 0;
			for (int j = 0; j < _data_name.size(); ++j)
			{
				snprintf(targetString, sizeof(targetString), ("pred\\" + prefix + "%d_pred_" + _data_name[j]).c_str(), i);
				Tensor imageTensor = pred.index({ Slice(),Slice(now_channel,now_channel + _data_channel[j]) });
				if (j == 1)
					imageTensor = torch::cat({ imageTensor,
						torch::sqrt(1 - torch::pow((imageTensor * 2 - 1).index({ Slice(),Slice(0,1) }), 2) - torch::pow((imageTensor * 2 - 1).index({ Slice(),Slice(1,2) }), 2)) * 0.5 + 0.5 }, 1);
				TensorToImage(imageTensor, targetString);
				now_channel += _data_channel[j];
			}
			feature->_compressor->_updateMoPweight = true;
			float error = loss.item().toFloat();
			WriteFloat(i, (float)error, RMSE_path);
			if (error < error_minmum)
			{
				if (i > 0)
				{
					snprintf(targetString, sizeof(targetString), ("pth\\" + _objectname + "_"+prefix + "%d_model.pth").c_str(), i);
					torch::save(model, targetString);
					snprintf(targetString, sizeof(targetString), ("pth\\" + _objectname + "_"+prefix + "%d_feature.pth").c_str(), i);
					torch::save(feature, targetString);
					error_count = 0;
					error_minmum = loss.item().toFloat();
					error_minmum_index = i;
				}
			}
			else if (!feature->_compressor->_TwoStage || pass >= maxpass)
			{
				error_count++;
				if (error_patience >= 0 && error_count >= error_patience)
				{
					std::cout << "reach error patience in final pass" << endl;
					break;
				}
			}
			double mse = error_minmum;
			double rmse = std::sqrt(mse);
			double psnr = 20. * std::log10(1. / rmse);
			printf("Test error: %f Test error minmum : mse(%f) rmse(%f) psnr(%f) @%d\n", loss.item().toFloat(), mse, rmse, psnr, error_minmum_index);
		}
	}
	}
	catch (const std::exception& e) {
		std::cerr << "Error during backward: " << e.what() << std::endl;
		return;
	}
}