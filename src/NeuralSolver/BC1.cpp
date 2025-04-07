//#include <NeuralAidedMBD/BC1.h>
//
//using std::cout;
//using std::endl;
//using std::min;
//using std::string;
//using std::tuple;
//using std::get;
//using torch::Tensor;
//using namespace torch::indexing;
//using namespace torch::autograd;
//
//BC1::BC1(at::DeviceType device, int epoch, float lr) : Compressor(device, epoch, lr) {	}
//
//void BC1::encode()
//{
//	Tensor srcRGB = _src;
//	OptimizeColorsBlock(srcRGB, _c0, _c1, _mask);
//}
//std::vector<Tensor> BC1::getcode()
//{
//	return std::vector<Tensor>({ _c0,_c1,_mask });
//}
//
//Tensor BC1::decode(const std::vector<Tensor>& code, double noisy)
//{
//	if (code.size() != 3)
//		cout << "decode Error" << endl;
//	Tensor c0 = code[0], c1 = code[1], mask = code[2];
//
//	mask = mask.unsqueeze(2); //[n,b*b,1]
//	c0 = c0.unsqueeze(1); //[n,1,c]
//	c1 = c1.unsqueeze(1); //[n,1,c]
//	//return mask * (c0 - c1) + c1;
//	return mask * c0 + c1;
//}
