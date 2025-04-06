#include <torch/torch.h>
#include <NeuralAidedMBD/Utils.h>
#include <iostream>
#include <string>
#include <cmath>
#include <tuple>
using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;


int main(int argc, char* argv[])
{
	at::DeviceType device;

	if (torch::cuda::is_available())
	{
		cout << "use cuda" << endl;
		device = at::kCUDA;
	}
	else
	{
		cout << "use cpu" << endl;
		device = at::kCPU;
	}

	UMBD::FTex3D Tex(UMBD::FGrid(16, 12, 2), 4, UMBD::EElementType::Float);
	for (uint64_t i = 0; i < Tex.GetNumElements(); i++)
	{
		Tex.At<float>(i) = (float)i;
	}

	Tensor Blocks = TexToBlock(Tex, 4, 4, device);

	auto sizes = Blocks.sizes();
	assert(sizes.size() == 3);
	assert(sizes[0] == Tex.GetGrid().GetOuterVolume() / 16);
	assert(sizes[1] == 16);
	assert(sizes[2] == 4);

	UMBD::FTex3D OutTex(UMBD::FGrid(16, 12, 2), 4, UMBD::EElementType::Float);
	BlockToTex(OutTex, Blocks, 4, 4);
	for (uint64_t i = 0; i < OutTex.GetNumElements(); i++)
	{
		assert(Tex.At<float>(i) == OutTex.At<float>(i));
	}

	return 0;
}
