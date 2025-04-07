#include "NeuralAidedMBD/RGBM.h"
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

RGBMcodec::RGBMcodec(DTBC_config config, int pretain, string objectname, int nm_vaild, string Fix_DTBC_best_epoch, string DTBC_best_epoch)
{
	_Fix_DTBC_best_epoch = Fix_DTBC_best_epoch;
	_DTBC_best_epoch = DTBC_best_epoch;
	_vaild = nm_vaild;
	_objectname = objectname;
	_pretain = pretain;
	_config = config;
	_compressor = new BC7(config._device, config._epoch, config._lr, config._bc7_use_mode, config._quantizeMode, config._optimizeMode, config._bc7_mode7Type, config._Ns, config._Nr);
	//_compressor = new BC3(config._device, config._epoch, config._lr, config._quantizeMode);
}

tuple<int, int, int, Tensor> HDRImageToTensor(const char* image_path) {
	int iw, ih, n;
	float* data = stbi_loadf(image_path, &iw, &ih, &n, 0);
	Tensor output_tensor = torch::from_blob(data, { ih, iw, n }, c10::TensorOptions().dtype(torch::kFloat32));
	output_tensor = output_tensor.clamp(0.f, 32.f);
	return { iw ,ih ,n , output_tensor };
}
void TensorToPngImage(Tensor image_tensor/*[h,w,c]*/, const char* image_path)
{
	image_tensor = torch::clamp(image_tensor * 255.f, 0.f, 255.f).to(torch::kUInt8);//[h,w,c]:[0,255]
	image_tensor = image_tensor.to(torch::kCPU).contiguous();
	uint8_t* data = image_tensor.data_ptr<uint8_t>();
	int height = (int)image_tensor.size(0);
	int width = (int)image_tensor.size(1);
	int channels = (int)image_tensor.size(2);
	stbi_write_png(image_path, width, height, channels, data, 0);
}
void TensorToHDRImage(Tensor image_tensor/*[1,c,h,w]*/, const char* image_path)
{
	image_tensor = image_tensor.squeeze(0).permute({ 1, 2, 0 }).to(torch::kFloat32);//[h,w,c]
	image_tensor = image_tensor.to(torch::kCPU).contiguous();
	float* data = image_tensor.data_ptr<float>();
	int height = (int)image_tensor.size(0);
	int width = (int)image_tensor.size(1);
	int channels = (int)image_tensor.size(2);
	stbi_write_hdr(image_path, width, height, channels, data);
}
void hdr_data_generate(const char* path, const char* save_path)
{
	tuple<int, int, int, Tensor> res = HDRImageToTensor(path);
	Tensor x_img_tensor = get<3>(res);
	torch::save(x_img_tensor, save_path);
}

void RGBMcodec::BC6_Test(torch::nn::MSELoss& loss_fn)
{
	Tensor tmp = _test_tex.squeeze().unsqueeze(1);//[c,1,h,w]
	tmp = nvtt_bc6(tmp).to(_config._device).to(torch::kFloat32).squeeze().unsqueeze(0);//[1,c,h,w]
	torch::Tensor loss = loss_fn(tmp, _test_tex);
	float error = loss.item().toFloat();
	double mse = error;
	double rmse = std::sqrt(mse);
	double psnr = 20. * std::log10(1 / rmse);
	printf("BC6: mse: %f rmse: %f psnr: %f \n", mse, rmse, psnr);
}

void RGBMcodec::start()
{
	//string objectname = "dry_orchard_meadow";
	_data_name = _objectname + "_1k.hdr";
	if (!std::filesystem::exists("data/" + _data_name + ".pth"))
	{
		cout << "get " + _data_name + ".pth" << endl;
		hdr_data_generate(("hdr_image/" + _data_name).c_str(), ("data/" + _data_name + ".pth").c_str());
	}
	torch::load(_test_tex, "data/" + _data_name + ".pth");
	_test_tex = _test_tex.to(_config._device).unsqueeze(0);//[1,h,w,c]
	_test_tex = _test_tex.permute({ 0,3,1,2 });//[1,c,h,w]
	_test_tex.set_requires_grad(false);
	cout << "load pth" << endl;

	auto model = RGBM(_config._device, _test_tex.sizes(), _compressor);

	torch::nn::MSELoss loss_fn(torch::nn::MSELossOptions().reduction(torch::kMean));
	BC6_Test(loss_fn);
	torch::optim::Adam* optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(_config._lr));
	optimizer->param_groups()[0].options().set_lr(_config._lr);

	int epoch1 = 10000;
	int epoch2 = 10000;

	if (!_vaild)
	{
		model->init(_test_tex);
		train(model, optimizer, loss_fn, epoch2, 1, 50, EncodeMode::DTBC);
	}
	else
	{
		torch::load(model, "pth\\RGBM_" + _config._codec_name + "_" + _objectname + "_10000.pth");
		valid(model, loss_fn, EncodeMode::BC);

		model->_compressor->_optimizeMode = Compressor::OptimizeMode::FixConfig;
		model->_compressor->_init_MoP_weight = true;
		model->forward(_test_tex,EncodeMode::DTBC, 0.);
		torch::load(model, "pth\\RGBM_" + _config._codec_name + "_" + _objectname + "_Fix_DTBC_" + _Fix_DTBC_best_epoch + ".pth");
		model->_compressor->_init_MoP_weight = false;
		valid(model, loss_fn, EncodeMode::DTBC);

		torch::load(model, "pth\\RGBM_" + _config._codec_name + "_" + _objectname + "_DTBC_" + _DTBC_best_epoch + ".pth");
		model->_compressor->_optimizeMode = Compressor::OptimizeMode::DTBC;
		model->_compressor->_init_MoP_weight = true;
		valid(model, loss_fn, EncodeMode::DTBC);
	}
}

void RGBMcodec::valid(RGBM& model, torch::nn::MSELoss& loss_fn, EncodeMode encodeMode)
{
	string prefix = encodeMode == EncodeMode::DTBC ? "DTBC_" : (encodeMode == EncodeMode::BC ? "BC_" : "");
	if (encodeMode == EncodeMode::DTBC && model->_compressor->_optimizeMode == Compressor::OptimizeMode::FixConfig)
		prefix = "Fix_" + prefix;
	model->_compressor->_updateMoPweight = false;
	c10::InferenceMode guard(true);
	Tensor pred = model->forward(_test_tex, (EncodeMode)((uint32_t)encodeMode | (uint32_t)EncodeMode::BC), 0.);
	torch::Tensor loss = loss_fn(pred, _test_tex);
	float error = loss.item().toFloat();
	double mse = error;
	double rmse = std::sqrt(mse);
	double psnr = 20. * std::log10(1 / rmse);
	printf("valid: mse: %f rmse: %f psnr: %f \n", mse, rmse, psnr);
	char targetString[256];
	snprintf(targetString, sizeof(targetString), ("pred\\RGBM_" + _config._codec_name + "_" + prefix + "pred_" + _data_name).c_str());
	Tensor imageTensor = pred;
	TensorToHDRImage(imageTensor, targetString);
}
void RGBMcodec::train(RGBM& model, torch::optim::Adam* optimizer, torch::nn::MSELoss& loss_fn, int epoch, int print_interval, int eval_interval, EncodeMode encodeMode)
{
	try {
		string prefix = encodeMode == EncodeMode::DTBC ? "DTBC_" : (encodeMode == EncodeMode::BC ? "BC_" : "");
		if (encodeMode == EncodeMode::DTBC && model->_compressor->_optimizeMode == Compressor::OptimizeMode::FixConfig)
			prefix = "Fix_" + prefix;
		char targetString[256];
		model->_compressor->_init_MoP_weight = true;
		double noisy = 100.f;
		double histloss = 1e20;
		int lr_patience = encodeMode == EncodeMode::DTBC ? 40 : 200;
		int lr_interval = 0;
		int cooldown = encodeMode == EncodeMode::DTBC ? 0 : 100;
		int error_patience = 5;
		int error_count = 0;
		float error_minmum = 1e15f;
		int error_minmum_index = -1;
		string cost_path = prefix + "cost.txt";
		string RMSE_path = prefix + "RMSE.txt";
		TxTClear(cost_path);
		TxTClear(RMSE_path);
		torch::optim::ReduceLROnPlateauScheduler reduceLR(*optimizer, torch::optim::ReduceLROnPlateauScheduler::min, 0.9f, lr_patience, 1e-6, torch::optim::ReduceLROnPlateauScheduler::abs, cooldown, {}, 1e-8, true);
		for (int i = 0; i <= epoch; ++i)
		{
			auto tmp_start = std::chrono::system_clock::now();
			if (i > 0)
			{
				model->train();
				optimizer->zero_grad();
			}
			noisy *= std::pow(1e-4 / 100, 1.0 / 500.0);
			torch::Tensor pred = model->forward(_test_tex, encodeMode, noisy);//[1,c,h,w]
			torch::Tensor loss = loss_fn(pred, _test_tex);
			float loss_item = loss.item().toFloat();
			Tensor absM = torch::abs(model->_m);
			if (i % print_interval == 0)
			{
				cout << "max:" << torch::max(absM).item().toFloat()
					<< " min:" << torch::min(absM).item().toFloat() << endl;
			}
			Tensor diff = absM - torch::clamp(absM, model->_InLowClamp, 1);
			loss += 100 * torch::mean(diff * diff);
			float loss2_item = loss.item().toFloat();
			if (i % print_interval == 0)
			{
				printf("mse: %f mse2: %f\n",
					loss_item, loss2_item);
			}
			WriteFloat(i, (float)loss_item, cost_path);
			if (i > 0)
			{
				loss.backward();
				optimizer->step();
				model->eval();
			}
			auto time = std::chrono::system_clock::now() - tmp_start;
			float qloss_item = loss_item;
			if (encodeMode == EncodeMode::DTBC)
			{
				double lr = optimizer->param_groups()[0].options().get_lr();
				bool LRstep = model->_compressor->DTBCLRScheduler(qloss_item, histloss, lr_interval, lr_patience);
				if (LRstep)
				{
					optimizer->param_groups()[0].options().set_lr(lr * 0.9);
				}
			}
			else
				reduceLR.step(qloss_item);
			model->_compressor->_init_MoP_weight = false;
			if (i % print_interval == 0 || i == 0)
			{
				double mse = loss_item;
				double rmse = std::sqrt(mse);
				double psnr = 20. * std::log10(1. / rmse);
				double lr = optimizer->param_groups()[0].options().get_lr();
				int mstime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
				printf("e: %d mse: %f rmse: %f psnr: %f lr: %lf t: %d ms noisy: %f ec: %d\n",
					i, mse, rmse, psnr, lr, mstime, noisy, error_count);
			}
			if (i % eval_interval == 0 || i == epoch)
			{
				c10::InferenceMode guard(true);
				{
					model->_compressor->_updateMoPweight = false;
					Tensor rgbm = model->m_To_rgbm(_test_tex);
					auto dtype = rgbm.dtype();
					rgbm = rgbm.squeeze().permute({ 1,2,0 });//[h,w,c]:[0,1]
					Tensor RGB = rgbm.index({ Slice(),Slice(),Slice(0,3) });//[h,w,3]
					Tensor A = rgbm.index({ Slice(),Slice(),Slice(3,4) });//[h,w,1]
					Tensor hdr = _test_tex.squeeze().permute({ 1,2,0 });//[h,w,3]
					A = torch::clamp(torch::ceil(A * 255.f) / 255.f, 0.f, 1.f);
					RGB = hdr / model->_Multiplier / (A * A);
					RGB = torch::round(RGB * 255.f) / 255.f;
					rgbm = torch::cat({ RGB,A }, 2);//[h,w,4]:[0,1]
					snprintf(targetString, sizeof(targetString), ("pred\\RGBM_" + prefix + "%d_" + _data_name).c_str(), i);
					TensorToPngImage(rgbm, targetString);
					model->_compressor->_updateMoPweight = true;
				}
				{
					model->_compressor->_updateMoPweight = false;
					Tensor pred = model->forward(_test_tex, encodeMode, 0.);
					torch::Tensor loss = loss_fn(pred, _test_tex);
					int now_channel = 0;
					snprintf(targetString, sizeof(targetString), ("pred\\RGBM_" + prefix + "%d_pred_" + _data_name).c_str(), i);
					Tensor imageTensor = pred;
					TensorToHDRImage(imageTensor, targetString);
					model->_compressor->_updateMoPweight = true;
					double mse = loss.item().toFloat();
					double rmse = std::sqrt(mse);
					double psnr = 20. * std::log10(1. / rmse);
					printf("RGBM error: mse(%f) rmse(%f) psnr(%f)\n", mse, rmse, psnr);
				}
				if (i > 0 && encodeMode == EncodeMode::None)
				{
					snprintf(targetString, sizeof(targetString), ("pth\\RGBM_" + _config._codec_name + "_" + _objectname + "_" + prefix + "%d.pth").c_str(), i);
					torch::save(model, targetString);
				}
				model->_compressor->_updateMoPweight = false;
				Tensor pred = model->forward(_test_tex, (EncodeMode)((uint32_t)encodeMode | (uint32_t)EncodeMode::BC), 0.);
				torch::Tensor loss = loss_fn(pred, _test_tex);
				int now_channel = 0;
				snprintf(targetString, sizeof(targetString), ("pred\\RGBM_" + _config._codec_name + "_" + prefix + "%d_pred_" + _data_name).c_str(), i);
				Tensor imageTensor = pred;
				TensorToHDRImage(imageTensor, targetString);
				model->_compressor->_updateMoPweight = true;
				float error = loss.item().toFloat();
				WriteFloat(i, (float)error, RMSE_path);
				if (error < error_minmum)
				{
					if (i > 0)
					{
						snprintf(targetString, sizeof(targetString), ("pth\\RGBM_" + _config._codec_name + "_" + _objectname + "_" + prefix + "%d.pth").c_str(), i);
						torch::save(model, targetString);
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
						std::cout << "reach error patience in final pass" << endl;
						break;
					}
				}
				double mse = loss.item().toFloat();
				double rmse = std::sqrt(mse);
				double psnr = 20. * std::log10(1. / rmse);
				double min_mse = error_minmum;
				double min_rmse = std::sqrt(min_mse);
				double min_psnr = 20. * std::log10(1. / min_rmse);
				printf("Test error: mse(%f) rmse(%f) psnr(%f), Test error minmum : mse(%f) rmse(%f) psnr(%f) @%d\n",
					mse, rmse, psnr,
					min_mse, min_rmse, min_psnr, error_minmum_index);
			}
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error during train: " << e.what() << std::endl;
		return;
	}
}