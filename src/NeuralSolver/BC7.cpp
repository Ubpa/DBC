#include <NeuralAidedMBD/BC7.h>

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;
using namespace torch::autograd;

BC7::BC7(at::DeviceType device, int epoch, float lr, bool* use_mode, QuantizeMode quantizeMode, OptimizeMode optimizeMode, Mode7Type mode7Type,int Ns,int Nr) : Compressor(device, epoch, lr, quantizeMode, optimizeMode) {
	_code456.resize(8);
	_rotation = _rotation.to(device);
	_rotationRGB = _rotationRGB.to(device);
	_index_selector = _index_selector.to(device);
	_mode7Type = mode7Type;
	for (int i = 0; i < 8; ++i)
		_use_mode[i] = use_mode[i];
	_Ns = Ns;
	_Nr = Nr;
	if (_use_mode[7] == true)
	{
		_bc7_partition2 = _bc7_partition2.to(device);
		for (int i = 0; i < 64; ++i)
		{
			_origin_bc7_partition2_subset[i] = torch::argwhere(_bc7_partition2[i] == 0).squeeze();
			_origin_bc7_partition2_subset[i + 64] = torch::argwhere(_bc7_partition2[i] == 1).squeeze();
			_origin_bc7_partition2_subset_repermute.index_put_({ Slice(i,i + 1),torch::cat({ _origin_bc7_partition2_subset[i],_origin_bc7_partition2_subset[i + 64] }) },torch::arange(0, _bc7_partition2.size(1), 1,torch::TensorOptions().dtype(torch::kInt64)));
			_origin_bc7_partition2_subset[i] = _origin_bc7_partition2_subset[i].to(device);
			_origin_bc7_partition2_subset[i + 64] = _origin_bc7_partition2_subset[i + 64].to(device);
		}
		_origin_bc7_partition2_subset_repermute = _origin_bc7_partition2_subset_repermute.to(device);
	}
}

void BC7::encode()
{
	if (_optimizeMode != OptimizeMode::FixConfig)
	{
		delete _modeweight;
		_modeweight = nullptr;
	}
	if (_use_mode[4] == true || _use_mode[5] == true)
	{
		Tensor srcRGB = torch::stack({
			_src.index({ Slice(), Slice(), _rotationRGB[0] }),
			_src.index({ Slice(), Slice(), _rotationRGB[1] }),
			_src.index({ Slice(), Slice(), _rotationRGB[2] }),
			_src.index({ Slice(), Slice(), _rotationRGB[3] }) }); //[rot,n,b*b,3]
		Tensor srcA = torch::stack({
			_src.select(-1, 3),
			_src.select(-1, 0),
			_src.select(-1, 1),
			_src.select(-1, 2)}).unsqueeze(-1); //[rot,n,b*b,1]
		Tensor c0, c1, mask;
		subset_encode(srcRGB, c0, c1, mask);
		Tensor Alphamax16, Alphamin16; //[rot,n,1]
		Tensor Alphamask; //[rot,n,b*b]
		OptimizeAlphaBlock(srcA, Alphamax16, Alphamin16, Alphamask);
		_code456[0] = c0;
		_code456[1] = c1;
		_code456[2] = mask;
		_code456[3] = Alphamask;
		_code456[4] = srcRGB;

	}
	if (_use_mode[6] == true)
	{
		Tensor c0, c1, mask;
		subset_encode(_src.unsqueeze(0), c0, c1, mask);
		_code456[5] = c0;
		_code456[6] = c1;
		_code456[7] = mask;
	}
	if (_use_mode[7] == true)
	{
		if (_init_MoP_weight)
		{
			std::vector<Tensor>dests;
			dests.resize(64);
			for (int partition = 0; partition < 64; ++partition)
			{
				Tensor c0[2], c1[2]; //[1,n,4]
				Tensor mask[2]; //[1,n,sum=b*b]
				Tensor src_subset0 = _src.index({ Slice(),_origin_bc7_partition2_subset[0 * 64 + partition] });
				Tensor src_subset1 = _src.index({ Slice(),_origin_bc7_partition2_subset[1 * 64 + partition] });
				subset_encode(src_subset0.unsqueeze(0), c0[0], c1[0], mask[0]);
				subset_encode(src_subset1.unsqueeze(0), c0[1], c1[1], mask[1]);
				Tensor dec0 = subset_decode(src_subset0.unsqueeze(0), c0[0], c1[0], mask[0], 3, 31);
				Tensor dec1 = subset_decode(src_subset1.unsqueeze(0), c0[1], c1[1], mask[1], 3, 31);
				Tensor dest7 = torch::cat({ dec0,dec1 }, 2); //[1,n,b*b,4]
				dest7 = dest7.index({ Slice(),Slice(),_origin_bc7_partition2_subset_repermute[partition] }); //[1,n,b*b,4]
				dests[partition] = dest7;
			}
			Tensor dest7 = torch::cat(dests); //[m,n,b*b,4]
			Tensor error = torch::sum(torch::pow(dest7 - this->_src.unsqueeze(0), 2), { -2,-1 }); //[m,n]
			_mode7_learned_weight = 1 / error.detach();
			_mode7_learned_weight.set_requires_grad(false);
		}

		if (_mode7Type == Mode7Type::BruteForce)
		{
			_Ns = 64, _Nr = 0;
		}
		if (_optimizeMode == OptimizeMode::FixConfig)
		{
			_Ns = 1, _Nr = 0;
		}
		_MoPIndices = MoPSelect(_mode7_learned_weight, _Ns, _Nr);
		_mode7_subset0_weight = _bc7_partition2.index({ _MoPIndices });//[Ns+Nr,n,b*b]
		_mode7_subset1_weight = 1 - _mode7_subset0_weight;//[Ns+Nr,n,b*b]

		Tensor* c0 = _code7[0];
		Tensor* c1 = _code7[1];
		Tensor* mask = _code7[2];
		subset_encode(_src.unsqueeze(0), c0[0], c1[0], mask[0], &_mode7_subset0_weight);
		subset_encode(_src.unsqueeze(0), c0[1], c1[1], mask[1], &_mode7_subset1_weight);
	}
}
std::vector<Tensor> BC7::getcode()
{
	return _code456;
}

Tensor BC7::decode(const std::vector<Tensor>& code, double noisy)
{
	std::vector<Tensor> dests;
	if (_use_mode[4] == true || _use_mode[5] == true)
	{
		Tensor c0 = code[0], c1 = code[1], mask = code[2], Alphamask = code[3], srcRGB = code[4];
		//Tensor qAlphamask = QuantizeAlphaMask(Alphamask); //[rot,n,b*b]
		//Tensor qRGB = subset_decode(srcRGB, c0, c1, mask);
		//Tensor dec = torch::cat({ qRGB, qAlphamask.unsqueeze(3) }, -1); //[rot,n,b*b,4]

		//dests.push_back(dec[0].unsqueeze(0)); //[n,b*b,4]
		//dests.push_back(dec[1].index({ Slice(),Slice(),_rotation[1] }).unsqueeze(0)); //[n,b*b,4]
		//dests.push_back(dec[2].index({ Slice(),Slice(),_rotation[2] }).unsqueeze(0)); //[n,b*b,4]
		//dests.push_back(dec[3].index({ Slice(),Slice(),_rotation[3] }).unsqueeze(0)); //[n,b*b,4]

		int maskqmax[4] = { 0,3,7,3 }, Alphamaskqmax[4] = { 0,7,3,3 }, colorqmax[4] = { 0,31,31,127 }, Alphaqmax[4] = { 0,63,63,255 };
		std::vector<int> qidxs;
		if (_QuantizeColor || _QuantizeMask)
		{
			if (_use_mode[4])
			{
				qidxs.push_back(1);
				qidxs.push_back(2);
			}
			if (_use_mode[5])
			{
				qidxs.push_back(3);
			}
		}
		else
		{
			qidxs.push_back(0);
		}
		for (int i : qidxs)
		{
			Tensor qAlphamask = QuantizeAlphaMask(Alphamask, Alphaqmax[i], Alphamaskqmax[i]); //[rot,n,b*b]
			Tensor qRGB = subset_decode(srcRGB, c0, c1, mask, maskqmax[i], colorqmax[i]);
			Tensor dec = torch::cat({ qRGB, qAlphamask.unsqueeze(3) }, -1); //[rot,n,b*b,4]

			dests.push_back(dec[0].unsqueeze(0)); //[n,b*b,4]
			dests.push_back(dec[1].index({ Slice(),Slice(),_rotation[1] }).unsqueeze(0)); //[n,b*b,4]
			dests.push_back(dec[2].index({ Slice(),Slice(),_rotation[2] }).unsqueeze(0)); //[n,b*b,4]
			dests.push_back(dec[3].index({ Slice(),Slice(),_rotation[3] }).unsqueeze(0)); //[n,b*b,4]
		}
	}

	if (_use_mode[6] == true)
	{
		Tensor c0 = code[5], c1 = code[6], mask = code[7];
		dests.push_back(subset_decode(_src, c0, c1, mask, 15, 127));
	}

	int mode7beginindex = dests.size();
	if (_use_mode[7] == true)
	{
		Tensor* c0 = _code7[0];
		Tensor* c1 = _code7[1];
		Tensor* mask = _code7[2];
		Tensor dec0 = subset_decode(_src, c0[0], c1[0], mask[0], 3, 31);//[n,b*b,c]
		Tensor dec1 = subset_decode(_src, c0[1], c1[1], mask[1], 3, 31);//[n,b*b,c]
		dests.push_back(_mode7_subset0_weight.unsqueeze(3) * dec0 + _mode7_subset1_weight.unsqueeze(3) * dec1); //[Ns+Nr,n,b*b,c]
	}
	Tensor dest = torch::cat(dests); //[m,n,b*b,4]
	Tensor error = torch::sum(torch::pow(dest - this->_src, 2), { -2,-1 }); //[m,n]
	if (_use_mode[7] == true && _mode7Type == Mode7Type::MoP && _updateMoPweight && _optimizeMode != OptimizeMode::FixConfig)
	{
		const float alpha = 0.2;
		Tensor new_weight = _mode7_learned_weight;
		new_weight.index_put_({
				_MoPIndices,
				torch::arange(0,_MoPIndices.size(1),torch::TensorOptions().dtype(torch::kInt64).device(_device)).unsqueeze(0).broadcast_to(_MoPIndices.sizes())
			},
			//torch::log10(1.f / torch::sqrt(error.index({ Slice(mode7beginindex, mode7beginindex + _Ns + _Nr) }).detach())));
			1 / error.index({ Slice(mode7beginindex, mode7beginindex + _Ns + _Nr) }).detach());
		_mode7_learned_weight = _mode7_learned_weight * (1.0 - alpha) + new_weight * alpha;
	}
	if (_modeweight == nullptr)
	{
		_modeweight = new Tensor();
		if (_mode7Type == Mode7Type::BruteForce
			|| _optimizeMode == OptimizeMode::FixConfig)
			noisy = 0;
		*_modeweight = AutoMax(
			//torch::rand(error.sizes(),error.options()),
			1/error,
			//torch::log10(1.f / torch::sqrt(error)),
			/*tau*/ 0, noisy, true, 0);
	}
	Tensor w = *_modeweight;
	w = w.unsqueeze(2).unsqueeze(3);
	dest = dest * w;
	return torch::sum(dest, 0); //[n,b*b,4]
}
