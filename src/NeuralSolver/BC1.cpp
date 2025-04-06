#include <NeuralAidedMBD/BC1.h>

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;
using namespace torch::autograd;

BC1::BC1(at::DeviceType device, int refinecount, int epoch, float lr, QuantizeMode quantizeMode) : Compressor(device, refinecount, epoch, lr, quantizeMode) {	}

void BC1::encode(float roundc, double tau, double noisy)
{
	//Tensor srcRGB = _src.index({Slice(), Slice(), Slice(0, 3) });
	Tensor srcRGB = _src;
	//torch::Tensor minmum = srcRGB.min(), maxmum = srcRGB.max();
	//srcRGB = (srcRGB - minmum) / (maxmum - minmum);
	/*srcRGB = (srcRGB - minmum) / torch::max((maxmum - minmum), _eps);*/
	OptimizeColorsBlock(srcRGB, _c0, _c1, _mask);
	//cout << torch::autograd::grad({ torch::sum(min16) }, { src }) << endl;
	//color = Quantize565(color);//del
	//mask = MatchColorsBlock(srcRGB, color);//del quant
	//cout << _refinecount << endl;
	for (int j = 0; j < _refinecount; ++j)
	{
		RefineBlock(srcRGB, _mask, _c0, _c1);
		//color = Quantize565(color);
		_mask = MatchColorsBlock(srcRGB, _c0, _c1);
	}
}
std::vector<Tensor> BC1::getcode()
{
	return std::vector<Tensor>({ _c0,_c1,_mask });
}

Tensor BC1::decode(const std::vector<Tensor>& code)
{
	if (code.size() != 3)
		cout << "decode Error" << endl;

	Tensor c0 = code[0], c1 = code[1], mask = code[2];
	Tensor maskreshape = mask.unsqueeze(2); //[n,b*b,1]
	c0 = c0.unsqueeze(1); //[n,1,c]
	c1 = c1.unsqueeze(1); //[n,1,c]
	//return maskreshape * (c0 - c1) + c1;
	return maskreshape * c0 + c1; //[n,b*b,c]
}

Tensor BC1::qdecode(const std::vector<Tensor>& code, float roundc, double tau, double noisy)
{
	if (code.size() != 3)
		cout << "qdecode Error" << endl;
	Tensor c0 = code[0], c1 = code[1], mask = code[2];

	torch::Tensor minmum = get<0>(mask.min(1)).unsqueeze(1), maxmum = get<0>(mask.max(1)).unsqueeze(1); //[n,1]
	Tensor scale = maxmum - minmum + _eps8; //[n,1]
	Tensor qmask = (mask - minmum) / scale; //[n,b*b]
	qmask = CustomRound::apply(qmask * 3) / 3; //[n,b*b]
	//qmask = torch::round(qmask * 3) / 3;
	qmask = qmask * scale + minmum; //[n,b*b]
	Tensor maskreshape = qmask.unsqueeze(2); //[n,b*b,1]
	c0 = c0.unsqueeze(1); //[n,1,c]
	c1 = c1.unsqueeze(1); //[n,1,c]
	//return maskreshape * (c0 - c1) + c1;
	return maskreshape * c0 + c1;
}

Tensor BC1::forward(const Tensor& src)
{
	_src = src.clone().detach().to(_device).requires_grad_(true);
	encode(-1.f);
	_dest = decode(getcode());
	return _dest;
}
Tensor BC1::backward(const Tensor& gradinput)
{
	Tensor grad = gradinput.to(_device);
	_dest.backward(grad);
	return _src.grad();
}
