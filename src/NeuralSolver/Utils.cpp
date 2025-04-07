#include <NeuralAidedMBD/Utils.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <NeuralAidedMBD/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <NeuralAidedMBD/stb_image_write.h>

using std::cout;
using std::endl;
using torch::indexing::Slice;

torch::Tensor nvtt_bc6(torch::Tensor src/*[c,d,h,w]*/)
{
	src = src.to(torch::kFloat32).to(at::kCPU).contiguous();

	nvtt::Format compform = nvtt::Format::Format_BC6S;
	nvtt::RefImage inputImage;
	inputImage.width = (int)src.size(3);
	inputImage.height = (int)src.size(2);
	inputImage.depth = (int)src.size(1);
	inputImage.num_channels = (int)src.size(0);
	if (inputImage.num_channels != 3)
		fprintf(stderr, "Error: nvtt_bc6 inputImage.num_channels != 3\n");
	inputImage.channel_interleave = false;
	inputImage.data = src.data_ptr();
	std::streamsize outputSizeBytes{};
	{
		nvtt::Context            context;
		nvtt::CompressionOptions options;
		options.setFormat(compform);
		outputSizeBytes =
			context.estimateSize(inputImage.width, inputImage.height, inputImage.depth, 1 /* number of mipmaps */, options);
	}
	const nvtt::CPUInputBuffer cpuInput(&inputImage,             // Array of RefImages
		nvtt::ValueType::FLOAT32,  // The type of the elements of the image
		1,                       // Number of RefImages
		4, 4                    // Tile dimensions of BC
	);
	nvtt::EncodeSettings encodeSettings =
		nvtt::EncodeSettings().SetFormat(compform).SetQuality(nvtt::Quality_Highest).SetUseGPU(false);
	std::vector<unsigned char> outputData(outputSizeBytes);
	if (!nvtt::nvtt_encode(cpuInput, outputData.data(), encodeSettings))
		std::cerr << "Encoding failed!" << std::endl;

	//decode
	nvtt::Surface decimage;
	if (!decimage.setImage3D(compform, inputImage.width, inputImage.height, inputImage.depth, outputData.data()))
		std::cerr << "load the DDS file failed!" << std::endl;
	//if (!decimage.save("image/test.png"))
	//	std::cerr << "save failed" << std::endl;
	torch::Tensor dst = torch::zeros(src.sizes(), c10::TensorOptions().device(at::kCPU).dtype(torch::kFloat32).requires_grad(false)).contiguous();
	memcpy_s(dst.data_ptr(), sizeof(float) * dst.numel(), decimage.data(), sizeof(float) * dst.numel());
	return dst;//[c,d,h,w]
}

torch::Tensor nvtt_bc7(torch::Tensor src/*[c,d,h,w]*/)
{
	src = src.to(at::kCPU).contiguous();

	nvtt::Format compform = nvtt::Format::Format_BC7;
	nvtt::RefImage inputImage;
	inputImage.width = (int)src.size(3);
	inputImage.height = (int)src.size(2);
	inputImage.depth = (int)src.size(1);
	inputImage.num_channels = (int)src.size(0);
	if (inputImage.num_channels != 4)
		fprintf(stderr, "Error: nvtt_bc7 inputImage.num_channels != 4\n");
	inputImage.channel_interleave = false;
	inputImage.data = src.data_ptr();
	std::streamsize outputSizeBytes{};
	{
		nvtt::Context            context;
		nvtt::CompressionOptions options;
		options.setFormat(compform);
		outputSizeBytes =
			context.estimateSize(inputImage.width, inputImage.height, inputImage.depth, 1 /* number of mipmaps */, options);
	}
	const nvtt::CPUInputBuffer cpuInput(&inputImage,             // Array of RefImages
		nvtt::ValueType::UINT8,  // The type of the elements of the image
		1,                       // Number of RefImages
		4, 4                    // Tile dimensions of BC
	);
	nvtt::EncodeSettings encodeSettings =
		nvtt::EncodeSettings().SetFormat(compform).SetQuality(nvtt::Quality_Highest).SetUseGPU(false);
	std::vector<unsigned char> outputData(outputSizeBytes);
	if (!nvtt::nvtt_encode(cpuInput, outputData.data(), encodeSettings))
		std::cerr << "Encoding failed!" << std::endl;

	//decode
	nvtt::Surface decimage;
	if (!decimage.setImage3D(compform, inputImage.width, inputImage.height, inputImage.depth, outputData.data()))
		std::cerr << "load the DDS file failed!" << std::endl;
	//if (!decimage.save("image/nvtt_bc7_test.tga", true))
	//	std::cerr << "save failed" << std::endl;
	torch::Tensor dst = torch::zeros(src.sizes(), c10::TensorOptions().device(at::kCPU).dtype(torch::kFloat).requires_grad(false)).contiguous();
	memcpy_s(dst.data_ptr(), sizeof(float) * dst.numel(), decimage.data(), sizeof(float) * dst.numel());
	return dst * 255.f;//[c,d,h,w]
}

bool IsFileExist(const char* filepath)
{
	return std::filesystem::exists(filepath);
}

std::string ToString(const std::vector<float> Values, const std::string& Split)
{
	std::ostringstream oss;

	for (size_t i = 0; i < Values.size(); ++i) {
		oss << Values[i];
		if (i < Values.size() - 1) {
			oss << Split;
		}
	}

	return oss.str();
}

void SaveStringToFile(const std::string& String, const char* FilePath)
{
	std::ofstream OutFile(FilePath);

	if (!OutFile) {
		std::cerr << "Error opening file: " << FilePath << std::endl;
		return;
	}

	OutFile << String;

	OutFile.close();
}
