#include <NeuralAidedMBD/BC3.h>

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;
using namespace torch::autograd;

BC3::BC3(at::DeviceType device, int epoch, float lr, QuantizeMode quantizeMode) : Compressor(device, epoch, lr, quantizeMode) {	}

void BC3::encode()
{
	_srcRGB = _src.index({ Slice(), Slice(), Slice(0,3) }); //[n,b*b,3]
	Tensor srcA = _src.select(-1, 3).unsqueeze(-1); //[n,b*b,1]
	subset_encode(_srcRGB, _v, _mu, _mask);
	Tensor Alphamax16, Alphamin16; //[rot,n,1]
	OptimizeAlphaBlock(srcA, Alphamax16, Alphamin16, _alphamask);
}
std::vector<Tensor> BC3::getcode()
{
	return std::vector<Tensor>({ _v,_mu,_mask,_alphamask,_srcRGB });
}

Tensor BC3::decode(const std::vector<Tensor>& code, double noisy)
{
	if (code.size() != 5)
		cout << "decode Error" << endl;
	Tensor v = code[0], mu = code[1], mask = code[2], alphamask = code[3], srcRGB = code[4];

	Tensor color = subset_decode(srcRGB, v, mu, mask, 3, 31);
	alphamask = QuantizeAlphaMask(alphamask, 255, 7);
	return torch::cat({ color, alphamask.unsqueeze(-1) }, -1);
}
