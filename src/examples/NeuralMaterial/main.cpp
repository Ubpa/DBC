#include "DBC/stb_image.h"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <cmath>
#include <tuple>
#include <DBC/BC7.h>
#include <DBC/Utils.h>
#include "NeuralMaterial.h"

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;
using namespace torch::indexing;

const float max_pixel = 255.0;
at::DeviceType device;

int main(int argc, char* argv[])
{
	torch::manual_seed(1);
	if (torch::cuda::is_available())
	{
		cout << "use cuda" << endl;
		device = at::kCUDA;
		torch::cuda::manual_seed_all(1);
	}
	else
	{
		cout << "use cpu" << endl;
		device = at::kCPU;
	}

	int epoch = 6000;
	float lr = 0.01f;
	Compressor::QuantizeMode quantizeMode = Compressor::QuantizeMode::Default;
	Compressor::OptimizeMode optimizeMode = Compressor::OptimizeMode::DBC;
	int encode_config_selection_Type = 1/*MoP*/;
	string objectname = "lubricant_spray";
	int pretain = 0;
	string nm_codec_name = "BC7";
	int Ns = 2;
	int Nr = 2;
	int featuresize = 512;
	int log = 1;
	int argindex = 1;
	if (argc > argindex)
		epoch = std::atoi(argv[argindex++]);
	if (argc > argindex)
		lr = (float)std::atof(argv[argindex++]);
	if (argc > argindex)
		quantizeMode = (Compressor::QuantizeMode)std::atoi(argv[argindex++]);
	if (argc > argindex)
		optimizeMode = (Compressor::OptimizeMode)std::atoi(argv[argindex++]);
	if (argc > argindex)
		encode_config_selection_Type = std::atoi(argv[argindex++]);
	if (argc > argindex)
		pretain = std::atoi(argv[argindex++]);
	if (argc > argindex)
		objectname = argv[argindex++];
	if (argc > argindex)
		nm_codec_name = argv[argindex++];
	if (argc > argindex)
		Ns = std::atoi(argv[argindex++]);
	if (argc > argindex)
		Nr = std::atoi(argv[argindex++]);
	if (argc > argindex)
		featuresize = std::atoi(argv[argindex++]);
	if (argc > argindex)
		log = std::atoi(argv[argindex++]);

	if (log)
	{
		char tmp[1024];
		snprintf(tmp, sizeof(tmp), "epoch=%d lr=%.3f qMode=%d optimMode=%d MoP=%d pretain=%d object=%s nm_codec=%s Ns=%d Nr=%d fsize=%d",
			epoch, lr, quantizeMode, optimizeMode, encode_config_selection_Type, pretain, objectname.c_str(), nm_codec_name.c_str(), Ns, Nr, featuresize);
		string filename(tmp);
		filename = "log\\" + filename + ".txt";
		FILE* stream1;
		freopen_s(&stream1, filename.c_str(), "w", stderr);
	}

	char targetString[1024];
	snprintf(targetString, sizeof(targetString),
		"[args]\n\
epoch: %d\n\
lr:  %f\n\
quantizeMode: %d\n\
optimizeMode: %d\n\
encode_config_selection_Type: %d\n\
pretain: %d\n\
objectname: %s\n\
nm_codec_name: %s\n\
Ns: %d\n\
Nr: %d\n\
featuresize: %d\n\
log: %d\n",
		epoch,
		lr,
		quantizeMode,	
		optimizeMode,
		encode_config_selection_Type,
		pretain,
		objectname.c_str(),
		nm_codec_name.c_str(),
		Ns,
		Nr,
		featuresize,
		log);
	printlog(targetString);

	bool bc7_use_mode4[8] = { 0,0,0,0,1,0,0,0 };
	bool bc7_use_mode5[8] = { 0,0,0,0,0,1,0,0 };
	bool bc7_use_mode6[8] = { 0,0,0,0,0,0,1,0 };
	bool bc7_use_mode7[8] = { 0,0,0,0,0,0,0,1 };
	bool bc7_use_mode45[8] = { 0,0,0,0,1,1,0,0 };
	bool bc7_use_mode456[8] = { 0,0,0,0,1,1,1,0 };
	bool bc7_use_mode4567[8] = { 0,0,0,0,1,1,1,1 };

	bool bc6_use_mode1To10[2] = { 1,0 };
	bool bc6_use_mode11To14[2] = { 0,1 };
	bool bc6_use_mode1To14[2] = { 1,1 };
	bool* use_mode = nullptr;
	if (nm_codec_name == "BC6")
		use_mode = bc6_use_mode1To14;
	else //BC7
		use_mode = bc7_use_mode4567;

	DBC_config config(device, epoch, lr, quantizeMode, optimizeMode, encode_config_selection_Type, use_mode, nm_codec_name, Ns, Nr);
	NeuralMaterial nm(config, pretain, objectname, featuresize);
	nm.start();

	if(log)
		fclose(stderr);

	return 0;
}
