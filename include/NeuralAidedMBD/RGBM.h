#pragma once
#include <torch/torch.h>
#include <NeuralAidedMBD/Compressor.h>
#include <NeuralAidedMBD/BC3.h>
#include <NeuralAidedMBD/BC7.h>
#include <NeuralAidedMBD/Utils.h>

using std::cout;
using std::endl;
using std::string;
using torch::Tensor;
using namespace torch::indexing;

struct RGBMImpl : torch::nn::Module {
public:
	RGBMImpl(at::DeviceType device, torch::IntArrayRef hdr_size, Compressor* compressor = nullptr)
	{
		_device = device;
		_compressor = compressor;
		_one = torch::tensor(1.0f, c10::TensorOptions().device(_device).dtype(torch::kFloat32));
		_zero = torch::tensor(0.f, c10::TensorOptions().device(_device).dtype(torch::kFloat32));
		_InLowClamp = 16.f / 255.f;
		_m = torch::rand({ hdr_size[0],1 ,hdr_size[2],hdr_size[3] }, torch::TensorOptions().dtype(torch::kFloat32).device(_device).requires_grad(true));
		register_parameter("m", _m, true);
	}
	void init(torch::Tensor src/*[n,c,h,w]*/)
	{
		_m.set_requires_grad(false);
		_m.copy_(RGBMEncode(src).index({ Slice(),Slice(3,4) }).detach());
		_m.set_requires_grad(true);
	}
	torch::Tensor m_To_rgbm(torch::Tensor src/*[n,c,h,w]*/)
	{
		//leaky clamp
		Tensor m = torch::abs(_m);
		m = torch::clamp(m, _InLowClamp, 1)
			+ torch::clamp(_m - 1, 0, {}) * 0.01
			+ torch::clamp(_m - _InLowClamp, {}, 0) * 0.01;
		Tensor color = src / _Multiplier / (m * m);
		color = torch::clamp(color, 0, 1)
			+ torch::clamp(_m - 1, 0, {}) * 0.01
			+ torch::clamp(_m - 0, {}, 0) * 0.01;
		Tensor dest = torch::cat({ color ,m }, 1);//[n,4,h,w]
		return dest;
	}
	torch::Tensor forward(torch::Tensor src/*[n,c,h,w]*/, EncodeMode encodeMode, double noisy)
	{
		Tensor tmp_rgbm = m_To_rgbm(src);
		if (encodeMode != EncodeMode::None)
		{
			if ((uint32_t)encodeMode & (uint32_t)EncodeMode::DTBC)
			{
				tmp_rgbm = tmp_rgbm * 2.f - 1.f;//[-1,1]
				Tensor block_rgbm = TensorToBlock(tmp_rgbm, _BlockSize);//[N,b*b,c]
				Tensor DTBC_block_rgbm = _compressor->DTBCcodec(block_rgbm, noisy);//[N,b*b,c]:[-1,1]
				tmp_rgbm = BlockToTensor(DTBC_block_rgbm, _BlockSize, src.sizes());//[n, c, h, w]
				tmp_rgbm = (tmp_rgbm + 1.f) / 2.f;//[0,1]
			}
			if ((uint32_t)encodeMode & (uint32_t)EncodeMode::BC)
			{
				auto dtype = tmp_rgbm.dtype();
				tmp_rgbm = tmp_rgbm.squeeze().permute({ 1,2,0 });//[h,w,c]:[0,1]
				//Tensor RGB = tmp_rgbm.index({ Slice(),Slice(),Slice(0,3) });//[h,w,3]
				//Tensor A = tmp_rgbm.index({ Slice(),Slice(),Slice(3,4) });//[h,w,1]
				//Tensor hdr = src.squeeze().permute({ 1,2,0 });//[h,w,3]
				//A = torch::clamp(torch::ceil(A * 255.f) / 255.f, 0.f, 1.f);
				//RGB = hdr / _Multiplier / (A * A);
				//RGB = torch::round(RGB * 255.f) / 255.f;
				//tmp_rgbm = torch::cat({ RGB,A }, 2);//[h,w,4]:[0,1]
				tmp_rgbm = torch::clamp(torch::round(tmp_rgbm * 255.f), 0.f, 255.f).to(torch::kUInt8);//[h,w,c]:[0,255]
				Tensor orig = tmp_rgbm;
				//tmp_rgbm = nvtt_bc3(tmp_rgbm.permute({ 2,0,1 }).unsqueeze(1)).squeeze(1).to(_device).unsqueeze(0).to(dtype);//[1,c,h,w]:[0,255]
				tmp_rgbm = nvtt_bc7(tmp_rgbm.permute({ 2,0,1 }).unsqueeze(1)).squeeze(1).to(_device).unsqueeze(0).to(dtype);//[1,c,h,w]:[0,255]
				//tmp_rgbm = Bc7e(tmp_rgbm).to(_device).permute({ 2,0,1 }).unsqueeze(0).to(dtype);//[1,c,h,w]:[0,255]
				{
					Tensor diff = tmp_rgbm.squeeze(0).permute({ 1,2,0 }).to(torch::kFloat32) / 255.f - orig.to(torch::kFloat32) / 255.f;
					std::cout << "RGBM MSE:" << torch::sum(diff * diff).item() << endl;
				}
				tmp_rgbm = tmp_rgbm / 255.f;//[1,c,h,w]:[0,1]
			}
		}
		Tensor dest_rgbm = RGBMDecode(tmp_rgbm);//[n,c,h,w]
		return dest_rgbm;
	}
	torch::Tensor RGBMEncode(const torch::Tensor src/*[n,c,h,w]*/)
	{
		Tensor RGB = torch::maximum(_zero, src) / _Multiplier;
		Tensor MaxRGB = std::get<0>(RGB.max(1, true));//[n,1,h,w]
		Tensor SqrtMaxRGB = torch::sqrt(MaxRGB);
		SqrtMaxRGB = torch::clamp(SqrtMaxRGB, _InLowClamp, 1.f);
		//Tensor SqrtMScale = torch::minimum(_one, torch::ceil(SqrtMaxRGB * 255.0f) / 255.0f);
		Tensor SqrtMScale = torch::minimum(_one, SqrtMaxRGB);
		Tensor MScale = SqrtMScale * SqrtMScale;//[n,1,h,w]
		Tensor Ratio = torch::minimum(_one, MScale / MaxRGB);//[n,1,h,w]
		Tensor RGBScale = RGB * Ratio / MScale;
		RGBScale = torch::round(RGBScale * 255.f) / 255.f;
		Tensor dest = torch::cat({ RGBScale ,SqrtMScale },1);//[n,4,h,w]
		return dest;
	}
	torch::Tensor RGBMDecode(const torch::Tensor src/*[n,c,h,w]*/)
	{
		Tensor RGB = src.index({ Slice(),Slice(0,3) });//[n,3,h,w]
		Tensor A = src.index({ Slice(),Slice(3,4) });//[n,1,h,w]
		Tensor dest = RGB * A * A * _Multiplier;
		return dest;
	}
	at::DeviceType _device;
	Compressor* _compressor;
	int _BlockSize = 4;
	int _BlockChannels = 4;
	float _Multiplier = 32;
	float _InLowClamp = 0.f;
	Tensor _one;
	Tensor _zero;
	Tensor _m;
};
TORCH_MODULE(RGBM);

class RGBMcodec
{
public:
	RGBMcodec(DTBC_config config, int pretain, string objectname,int nm_vaild, string Fix_DTBC_best_epoch,string DTBC_best_epoch);
	~RGBMcodec() { delete _compressor; }
	void start();
	void train(RGBM& model, torch::optim::Adam* optimizer, torch::nn::MSELoss& loss_fn, int epoch, int print_interval, int eval_interval, EncodeMode encodeMode);
	void valid(RGBM& model, torch::nn::MSELoss& loss_fn, EncodeMode encodeMode);
	void BC6_Test(torch::nn::MSELoss& loss_fn);
	Tensor _test_tex;//[1,c,h,w]
	DTBC_config _config;
	Compressor* _compressor;
	string _data_name;
	int _pretain;
	string _objectname;
	int _vaild;
	string _Fix_DTBC_best_epoch;
	string _DTBC_best_epoch;
};
