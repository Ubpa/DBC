#include <DBC/BC6.h>

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;
using namespace torch::autograd;

BC6::BC6(at::DeviceType device, bool* use_mode, QuantizeMode quantizeMode, OptimizeMode optimizeMode, Mode1To10Type mode1To10Type,int Ns,int Nr) : Compressor(device, quantizeMode, optimizeMode) {
	_code11To14.resize(3);
	_mode1To10Type = mode1To10Type;
	for (int i = 0; i < 2; ++i)
		_use_mode[i] = use_mode[i];
	_Ns = Ns;
	_Nr = Nr;
	if (_use_mode[0] == true)
	{
		_bc6_partition2 = _bc6_partition2.to(device);
		for (int i = 0; i < 32; ++i)
		{
			_origin_bc6_partition2_subset[i] = torch::argwhere(_bc6_partition2[i] == 0).squeeze();
			_origin_bc6_partition2_subset[i + 32] = torch::argwhere(_bc6_partition2[i] == 1).squeeze();
			_origin_bc6_partition2_subset_repermute.index_put_({ Slice(i,i + 1),torch::cat({ _origin_bc6_partition2_subset[i],_origin_bc6_partition2_subset[i + 32] }) }, torch::arange(0, _bc6_partition2.size(1), 1, torch::TensorOptions().dtype(torch::kInt64)));
			_origin_bc6_partition2_subset[i] = _origin_bc6_partition2_subset[i].to(device);
			_origin_bc6_partition2_subset[i + 32] = _origin_bc6_partition2_subset[i + 32].to(device);
		}
		_origin_bc6_partition2_subset_repermute = _origin_bc6_partition2_subset_repermute.to(device);
	}
	_QuantizeColor = false;
}

void BC6::encode()
{
	if (_optimizeMode != OptimizeMode::FixConfig)
	{
		delete _modeweight;
		_modeweight = nullptr;
	}
	if (_use_mode[1] == true)
	{
		Tensor c0, c1, mask;
		subset_encode(_src.unsqueeze(0), c0, c1, mask);
		_code11To14[0] = c0;
		_code11To14[1] = c1;
		_code11To14[2] = mask;
	}
	if (_use_mode[0] == true)
	{
		if (_init_MoP_weight)
		{
			std::vector<Tensor>dests;
			dests.resize(32);
			for (int partition = 0; partition < 32; ++partition)
			{
				Tensor c0[2], c1[2]; //[1,n,4]
				Tensor mask[2]; //[1,n,sum=b*b]
				Tensor src_subset0 = _src.index({ Slice(),_origin_bc6_partition2_subset[0 * 32 + partition] });
				Tensor src_subset1 = _src.index({ Slice(),_origin_bc6_partition2_subset[1 * 32 + partition] });
				subset_encode(src_subset0.unsqueeze(0), c0[0], c1[0], mask[0]);
				subset_encode(src_subset1.unsqueeze(0), c0[1], c1[1], mask[1]);
				Tensor dec0 = subset_decode(src_subset0.unsqueeze(0), c0[0], c1[0], mask[0], 7, 0);
				Tensor dec1 = subset_decode(src_subset1.unsqueeze(0), c0[1], c1[1], mask[1], 7, 0);
				Tensor dest1To10 = torch::cat({ dec0,dec1 }, 2); //[1,n,b*b,4]
				dest1To10 = dest1To10.index({ Slice(),Slice(),_origin_bc6_partition2_subset_repermute[partition] }); //[1,n,b*b,4]
				dests[partition] = dest1To10;
			}
			Tensor dest1To10 = torch::cat(dests); //[m,n,b*b,4]
			Tensor error = torch::sum(torch::pow(dest1To10 - this->_src.unsqueeze(0), 2), { -2,-1 }); //[m,n]
			_bc6_mode1To10_learned_weight = 1 / error.detach();
			_bc6_mode1To10_learned_weight.set_requires_grad(false);
		}

		if (_mode1To10Type == Mode1To10Type::BruteForce)
		{
			_Ns = 32, _Nr = 0;
		}
		if (_optimizeMode == OptimizeMode::FixConfig)
		{
			_Ns = 1, _Nr = 0;
		}
		_MoPIndices = MoPSelect(_bc6_mode1To10_learned_weight, _Ns, _Nr);
		_bc6_mode1To10_subset0_weight = _bc6_partition2.index({ _MoPIndices });//[Ns+Nr,n,b*b]
		_bc6_mode1To10_subset1_weight = 1 - _bc6_mode1To10_subset0_weight;//[Ns+Nr,n,b*b]

		Tensor* c0 = _code1To10[0];
		Tensor* c1 = _code1To10[1];
		Tensor* mask = _code1To10[2];
		subset_encode(_src.unsqueeze(0), c0[0], c1[0], mask[0], &_bc6_mode1To10_subset0_weight);
		subset_encode(_src.unsqueeze(0), c0[1], c1[1], mask[1], &_bc6_mode1To10_subset1_weight);
	}
}
std::vector<Tensor> BC6::getcode()
{
	return _code11To14;
}

Tensor BC6::decode(const std::vector<Tensor>& code, double noisy)
{
	std::vector<Tensor> dests;
	if (_use_mode[1] == true)
	{
		Tensor c0 = code[0], c1 = code[1], mask = code[2];
		dests.push_back(subset_decode(_src, c0, c1, mask, 15, 0));
	}
	int mode1To10beginindex = (int)dests.size();
	if (_use_mode[0] == true)
	{
		Tensor* c0 = _code1To10[0];
		Tensor* c1 = _code1To10[1];
		Tensor* mask = _code1To10[2];
		Tensor dec0 = subset_decode(_src, c0[0], c1[0], mask[0], 7, 0);//[n,b*b,c]
		Tensor dec1 = subset_decode(_src, c0[1], c1[1], mask[1], 7, 0);//[n,b*b,c]
		dests.push_back(_bc6_mode1To10_subset0_weight.unsqueeze(3) * dec0 + _bc6_mode1To10_subset1_weight.unsqueeze(3) * dec1); //[Ns+Nr,n,b*b,c]
	}
	Tensor dest = torch::cat(dests); //[m,n,b*b,3]
	Tensor error = torch::sum(torch::pow(dest - this->_src, 2), { -2,-1 }); //[m,n]
	if (_use_mode[0] == true && _mode1To10Type == Mode1To10Type::MoP && _updateMoPweight && _optimizeMode != OptimizeMode::FixConfig)
	{
		const float alpha = 0.2f;
		Tensor new_weight = _bc6_mode1To10_learned_weight;
		new_weight.index_put_({
				_MoPIndices,
				torch::arange(0,_MoPIndices.size(1),torch::TensorOptions().dtype(torch::kInt64).device(_device)).unsqueeze(0).broadcast_to(_MoPIndices.sizes())
			},
			1 / error.index({ Slice(mode1To10beginindex, mode1To10beginindex + _Ns + _Nr) }).detach());
		_bc6_mode1To10_learned_weight = _bc6_mode1To10_learned_weight * (1.0f - alpha) + new_weight * alpha;
	}
	if (_modeweight == nullptr)
	{
		_modeweight = new Tensor();
		if (_mode1To10Type == Mode1To10Type::BruteForce
			|| _optimizeMode == OptimizeMode::FixConfig)
			noisy = 0;
		*_modeweight = GumbelMax(-error, noisy, 0);
	}
	Tensor w = *_modeweight;
	w = w.unsqueeze(2).unsqueeze(3);
	dest = dest * w;
	return torch::sum(dest, 0); //[n,b*b,3]
}
