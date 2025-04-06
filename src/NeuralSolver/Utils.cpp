#include <NeuralAidedMBD/Utils.h>
#include "rdo_bc_encoder.h"
#include <UMBD_ext/UMBDIO.h>
#include <filesystem>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using torch::indexing::Slice;

void Bc7e(UMBD::FTex3D& Tex)
{
	rdo_bc::rdo_bc_params rp;
	rp.m_dxgi_format = DXGI_FORMAT_BC7_UNORM;
	rp.m_rdo_max_threads = 16;
	rp.m_status_output = false;
	rdo_bc::rdo_bc_encoder encoder;
	unsigned int w = (unsigned int)Tex.GetBlockGrid().Width, h = (unsigned int)Tex.GetBlockGrid().Height, d = (unsigned int)Tex.GetBlockGrid().Depth;
	unsigned int c = (unsigned int)Tex.GetNumChannel();
	for (int k = 0; k < (int)d; ++k)
	{
		bc7e_utils::image_u8 source_image;
		source_image.clear();
		std::vector<unsigned char> pixels;
		pixels.resize(w * h * 4);
		source_image.init(w, h);
		for (int i = 0; i < (int)(w * h); ++i)
		{
			for (int j = 0; j < (int)c; ++j)
			{
				pixels[i * 4 + j] = UMBD::ElementFloatClampToUint8(Tex.GetFloat(i * c + j + k * (w * h * c)));
			}
			for (int j = c; j < 4; ++j)
			{
				pixels[i * 4 + j] = (unsigned char)255;
			}
		}
		memcpy(&source_image.get_pixels()[0], &pixels[0], w * h * sizeof(uint32_t));

		if (!encoder.init(source_image, rp))
		{
			fprintf(stderr, "rdo_bc_encoder::init() failed!\n");
		}
		if (!encoder.encode())
		{
			fprintf(stderr, "rdo_bc_encoder::encode() failed!\n");
		}
		bc7e_utils::image_u8 unpacked_image;
		encoder.unpack_blocks(unpacked_image);
		for (int i = 0; i < (int)h; ++i)
		{
			for (int j = 0; j < (int)w; ++j)
			{
				for (int z = 0; z < (int)c; ++z)
				{
					Tex.SetFloat((i * w + j) * c + z + k * (w * h * c), UMBD::ElementUint8ToFloat(unpacked_image(j, i)[z]));
				}
			}
		}
	}
}

torch::Tensor Bc7e(torch::Tensor src/*[h,w,c]*/)
{
	rdo_bc::rdo_bc_params rp;
	rp.m_dxgi_format = DXGI_FORMAT_BC7_UNORM;
	rp.m_rdo_max_threads = 16;
	rp.m_status_output = false;
	rdo_bc::rdo_bc_encoder encoder;
	bc7e_utils::image_u8 source_image;
	source_image.clear();
	std::vector<unsigned char> pixels;
	unsigned int w = (unsigned int)src.size(1), h = (unsigned int)src.size(0), channel = (unsigned int)src.size(2);
	pixels.resize(w * h * 4);
	source_image.init(w, h);
	if (channel != 4)
		src = torch::cat({ src,torch::tensor({255},torch::TensorOptions().dtype(src.dtype()).device(src.device())).unsqueeze(0).unsqueeze(0).broadcast_to({h,w,1}) }, -1);
	src = src.to(at::kCPU).contiguous();
	memcpy_s(&source_image.get_pixels()[0], sizeof(unsigned char) * src.numel(), src.data_ptr(), sizeof(unsigned char) * src.numel());

	//for (int i = 0; i < 16; ++i)
	//{
	//	for (int j = 0; j < src.size(1); ++j)
	//	{
	//		pixels[i * 4 + j] = (unsigned char)(src[i][j].item().toFloat() * 255);
	//	}
	//	for (int j = (int)src.size(1); j < 4; ++j)
	//	{
	//		pixels[i * 4 + j] = (unsigned char)(1 * 255);
	//	}
	//}
	//memcpy(&source_image.get_pixels()[0], &pixels[0], w * h * sizeof(uint32_t));

	if (!encoder.init(source_image, rp))
		fprintf(stderr, "rdo_bc_encoder::init() failed!\n");
	if (!encoder.encode())
		fprintf(stderr, "rdo_bc_encoder::encode() failed!\n");
	bc7e_utils::image_u8 unpacked_image;
	if (!encoder.unpack_blocks(unpacked_image))
		fprintf(stderr, "unpacked_image error!\n");

	torch::Tensor dst = torch::zeros_like(src);
	dst = dst.to(at::kCPU).contiguous();
	memcpy_s(dst.data_ptr(), sizeof(unsigned char) * dst.numel(), &unpacked_image.get_pixels()[0], sizeof(unsigned char) * dst.numel());
	if (channel != 4)
		dst = dst.index({ Slice(),Slice(),Slice(0,3) });
	//for (uint32_t y = 0; y < 4; y++)
	//	for (uint32_t x = 0; x < 4; x++)
	//	{
	//		for (uint32_t z = 0; z < src.size(1); z++)
	//			dst[x + y * 4][z] = ((int)(unpacked_image(x, y)[z])) / 255.f;
	//	}
	//std::cout << src << std::endl << dst << std::endl;
	//torch::nn::L1Loss loss_fn(torch::nn::L1LossOptions().reduction(torch::kNone));
	//torch::nn::MSELoss l2loss_fn(torch::nn::MSELossOptions().reduction(torch::kNone));
	//std::cout << loss_fn(dst, src).mean({ -2, -1 }) << std::endl;
	//std::cout << l2loss_fn(dst, src).mean({ -2, -1 }) << std::endl;

	return dst;
}

float EvaluateError(const UMBD::FTex3D& f, const UMBD::FTex3D& approx_f)
{
	double SumSquaredSize = 0.;
	double SE = 0.;
	double MaxValue = 0.;
	double ToneSE = 0.;
	double ToneMaxValue = 0.;
	for (const UMBD::FUint64Vector& Point : f.GetGrid())
	{
		double SquaredSize = 0.;
		double SquaredSizeApprox = 0.;
		for (size_t c = 0; c < f.GetNumChannel(); c++)
		{
			double ref = f.At<float>(Point, c);
			double approx = approx_f.At<float>(Point, c);
			SquaredSize += UCommon::Pow2(ref);
			SquaredSizeApprox += UCommon::Pow2(approx);
			SE += UCommon::Pow2(ref - approx);

			double Toneref = ref / (1 + std::abs(ref));
			double Toneapprox = approx / (1 + std::abs(approx));
			ToneSE += UCommon::Pow2(Toneref - Toneapprox);
		}
		double Size = std::sqrt(SquaredSize);
		double SizeApprox = std::sqrt(SquaredSizeApprox);
		MaxValue = std::max(MaxValue, Size);
		SumSquaredSize += UCommon::Pow2(Size);

		double ToneSize = Size / (1 + Size);
		ToneMaxValue = std::max(ToneMaxValue, ToneSize);
	}
	double Mean = std::sqrt(SumSquaredSize / f.GetGrid().GetOuterVolume());
	double MSE = SE / f.GetGrid().GetOuterVolume();
	double RMSE = std::sqrt(MSE);
	double ToneMSE = ToneSE / f.GetGrid().GetOuterVolume();
	double ToneRMSE = std::sqrt(ToneMSE);
	double PSNR = 20.f * std::log10(ToneMaxValue / ToneRMSE);
	double RRMSE = RMSE / Mean;

	std::cout
		<< "Max: " << MaxValue
		<< " Mean: " << Mean
		<< " RMSE: " << RMSE
		<< " RRMSE: " << RRMSE * 100. << "%"
		<< " ToneMax: " << ToneMaxValue
		<< " ToneRMSE: " << ToneRMSE
		<< " PSNR: " << PSNR
		<< std::endl;

	return (float)RMSE;
}

UMBD::FTex3D GetErrorTex(const UMBD::FTex3D& f, const UMBD::FTex3D& approx_f, float MaxError)
{
	EvaluateError(f, approx_f);
	
	constexpr uint64_t NumColors = 5;
	UMBD::FLinearColorRGB ColorBar[NumColors] = {
		{68.f / 255.f,1.f / 255.f,84.f / 255.f},
		{58.f / 255.f,81.f / 255.f,138.f / 255.f},
		{30.f / 255.f,151.f / 255.f,138.f / 255.f},
		{100.f / 255.f,200.f / 255.f,93.f / 255.f},
		{253.f / 255.f,231.f / 255.f,36.f / 255.f},
	};
	const float ErrorDelta = MaxError / NumColors;

	UMBD::FTex3D errorTex(UMBD::FGrid(f.GetGrid().Width, f.GetGrid().Height, 1), 3, UMBD::EElementType::Uint8);
	float MaxAbsError = 0.f;
	for (uint64_t y = 0; y < f.GetGrid().Height; y++)
	{
		for (uint64_t x = 0; x < f.GetGrid().Width; x++)
		{
			float SumSE = 0.f;
			for (uint64_t z = 0; z < f.GetGrid().Depth; z++)
			{
				UMBD::FUint64Vector Point(x, y, z);
				for (uint64_t c = 0; c < f.GetNumChannel(); c++)
				{
					float diff = approx_f.GetFloat(Point, c) - f.GetFloat(Point, c);
					SumSE += UCommon::Pow2(diff);
				}
			}
			float AbsError = std::sqrt(SumSE / f.GetGrid().Depth);
			uint64_t Index0 = std::min<uint64_t>((uint64_t)(AbsError / ErrorDelta), NumColors - 1);
			uint64_t Index1 = std::min<uint64_t>(Index0 + 1, NumColors - 1);
			auto Color0 = ColorBar[Index0];
			auto Color1 = ColorBar[Index1];
			float t = (AbsError - Index0 * ErrorDelta) / ErrorDelta;
			auto Color = Color0 * (1.f - t) + Color1 * t;
			errorTex.At<uint8_t>(UMBD::FUint64Vector(x, y, 0), 0) = UMBD::ElementFloatClampToUint8(Color.X);
			errorTex.At<uint8_t>(UMBD::FUint64Vector(x, y, 0), 1) = UMBD::ElementFloatClampToUint8(Color.Y);
			errorTex.At<uint8_t>(UMBD::FUint64Vector(x, y, 0), 2) = UMBD::ElementFloatClampToUint8(Color.Z);
			MaxAbsError = std::max(MaxAbsError, AbsError);
		}
	}
	std::cout << "MaxAbsError: " << MaxAbsError << std::endl;
	return errorTex;
}

UMBD::FTex3D GetSHVisTex(const UMBD::FTex3D& f)
{
	UMBD::FTex3D SHVisTex(UMBD::FGrid(f.GetGrid().Width, f.GetGrid().Height, 1), 3, UMBD::EElementType::Uint8);
	uint64_t NumBasises = f.GetNumChannel() / 3;
	for (uint64_t y = 0; y < f.GetGrid().Height; y++)
	{
		for (uint64_t x = 0; x < f.GetGrid().Width; x++)
		{
			UMBD::FLinearColorRGB SumColor(0.f);
			for (uint64_t z = 0; z < f.GetGrid().Depth; z++)
			{
				UMBD::FUint64Vector Point(x, y, z);
				for (uint64_t c = 0; c < 3; c++)
				{
					SumColor[c] += f.GetFloat(Point, c * NumBasises);
				}
			}
			UMBD::FLinearColorRGB AvgColor = SumColor / (float)f.GetGrid().Depth;
			UMBD::FLinearColorRGB ToneAvgColor = AvgColor / (1.f + AvgColor);
			for (uint64_t c = 0; c < 3; c++)
			{
				SHVisTex.At<uint8_t>(UMBD::FUint64Vector(x, y, 0), c) = UMBD::ElementFloatClampToUint8(ToneAvgColor[c]);
			}
		}
	}
	return SHVisTex;
}

void GetCompTex(UMBD::FTex3D& tex1, UMBD::FTex3D& tex2, std::string name)
{
	assert(tex1.GetGrid() == tex2.GetGrid());
	UMBD::FTex3D errorTex(tex1.GetGrid(), 3, UMBD::EElementType::Float);
	float scale = 0;
	for (size_t i = 0; i < tex1.GetGrid().GetOuterVolume(); i++) {
		float delta = tex1.At<float>(i) - tex2.At<float>(i);
		scale = std::max(scale, std::abs(delta));
		errorTex.At<float>(i * 3) = 0;
		errorTex.At<float>(i * 3 + 1) = 0;
		errorTex.At<float>(i * 3 + 2) = 0;
		if (delta > 0)
			errorTex.At<float>(i * 3 + 1) = delta;
		else
			errorTex.At<float>(i * 3) = -delta;
	}
	for (size_t i = 0; i < errorTex.GetGrid().GetOuterVolume(); i++)
		for (size_t j = 0; j < 3; ++j)
			errorTex.At<float>(i * 3 + j) = errorTex.At<float>(i * 3 + j) / scale;
	errorTex = errorTex.ToUint8();
	UMBD::SaveTex2DAsPNG(errorTex, name.c_str(), 0, 0, 3);
}

torch::Tensor TexToBlock(const UMBD::FTex3D& Tex, int64_t BlockSize, int64_t BlockChannels, at::Device device)
{
	assert(Tex.GetElementType() == UMBD::EElementType::Float
		|| Tex.GetElementType() == UMBD::EElementType::Double);

	const UMBD::FGrid Grid = Tex.GetGrid();

	const c10::ScalarType tensorType = Tex.GetElementType() == UMBD::EElementType::Float ? c10::kFloat : c10::kDouble;

#if 1
	torch::Tensor tensorC = torch::from_blob(
		const_cast<void*>(Tex.GetStorage()),
		{ (int64_t)Grid.GetOuterVolume(), BlockChannels },
		c10::TensorOptions().dtype(tensorType))
		.to(c10::kFloat).clone().detach().to(device); //[d*h*w,c]

	tensorC = tensorC.reshape({ (int64_t)Grid.Depth,(int64_t)Grid.Height,(int64_t)Grid.Width,BlockChannels }); //[d,h,w,c]

	tensorC = tensorC
		.unfold(1, BlockSize, BlockSize)
		.unfold(2, BlockSize, BlockSize)
		.unfold(3, BlockChannels, BlockChannels)
		.reshape({ -1,BlockSize * BlockSize,BlockChannels }); //[n,b*b,c]
#else
	UMBD::FGrid GridBlockGrid = Grid.GetBlockGrid(UMBD::FUint64Vector2(BlockSize, BlockSize));
	UMBD::FTex3D BC1BlockTex(UMBD::FGrid(BlockSize, BlockSize, 1), BlockChannels, Tex.GetElementType());
	torch::Tensor tensorC = torch::empty({ (int64_t)GridBlockGrid.GetOuterVolume(),BlockSize * BlockSize,BlockChannels }, torch::TensorOptions().dtype(tensorType));
	for (uint64_t Index = 0; Index < GridBlockGrid.GetOuterVolume(); Index++)
	{
		const UMBD::FUint64Vector Point = GridBlockGrid.GetPoint(Index);
		UMBD::FUint64Vector BeginPoint(Point.X * BlockSize, Point.Y * BlockSize, Point.Z);
		for (const UMBD::FUint64Vector& PointInBlock : UMBD::FGrid(BlockSize, BlockSize, 1))
		{
			for (uint64_t C = 0; C < (uint64_t)BlockChannels; C++)
			{
				if (BC1BlockTex.GetElementType() == UMBD::EElementType::Float)
					BC1BlockTex.At<float>(PointInBlock, C) = Tex.At<float>(BeginPoint + PointInBlock, C);
				else
					BC1BlockTex.At<double>(PointInBlock, C) = Tex.At<double>(BeginPoint + PointInBlock, C);
			}
		}
		torch::Tensor x = torch::from_blob(BC1BlockTex.GetStorage(), { BlockSize * BlockSize, BlockChannels }, c10::TensorOptions().dtype(tensorType));
		tensorC[Index] = x;
	}
#endif
	return tensorC;
}

void BlockToTex(UMBD::FTex3D& Tex, const torch::Tensor& blockin, int64_t BlockSize, int64_t BlockChannels)
{
	assert(Tex.GetElementType() == UMBD::EElementType::Float
		|| Tex.GetElementType() == UMBD::EElementType::Double);

	const UMBD::FGrid Grid = Tex.GetGrid();

	const c10::ScalarType tensorType = Tex.GetElementType() == UMBD::EElementType::Float ? c10::kFloat : c10::kDouble;

#if 1
	torch::Tensor block = blockin.reshape({ (int64_t)Grid.Depth,-1, BlockSize * BlockSize,BlockChannels });
	int64_t blockcnt = block.size(1);
	block = block.permute({ 0,3,2,1 }).reshape({ (int64_t)Grid.Depth,-1, blockcnt });
	torch::nn::Fold fold(torch::nn::FoldOptions({ (int64_t)Grid.Height,(int64_t)Grid.Width }, { BlockSize, BlockSize }).stride(BlockSize));
	block = fold(block);
	block = block.permute({ 0,2,3,1 }).reshape({ -1 }).to(tensorType).to(at::kCPU).contiguous();
	assert(block.numel() == Tex.GetNumElements());
	if (Tex.GetElementType() == UMBD::EElementType::Double)
		memcpy_s(Tex.GetStorage(), sizeof(double) * block.numel(), block.data_ptr(), sizeof(double) * block.numel());
	else
		memcpy_s(Tex.GetStorage(), sizeof(float) * block.numel(), block.data_ptr(), sizeof(float) * block.numel());
#else
	Tensor block_cpu = block.to(at::kCPU);
	UMBD::FGrid GridBlockGrid = Grid.GetBlockGrid(UMBD::FUint64Vector2(BlockSize, BlockSize));
	for (uint64_t Index = 0; Index < GridBlockGrid.GetOuterVolume(); Index++)
	{
		const UMBD::FUint64Vector Point = GridBlockGrid.GetPoint(Index);
		UMBD::FUint64Vector BeginPoint(Point.X * BlockSize, Point.Y * BlockSize, Point.Z);
		for (const UMBD::FUint64Vector& PointInBlock : UMBD::FGrid(BlockSize, BlockSize, 1))
		{
			for (uint64_t C = 0; C < (uint64_t)BlockChannels; C++)
			{
				if (Tex.GetElementType() == UMBD::EElementType::Double)
					Tex.At<double>(BeginPoint + PointInBlock, C) = block_cpu[Index][PointInBlock.Y * BlockSize + PointInBlock.X][C].item().toDouble();
				else
					Tex.At<float>(BeginPoint + PointInBlock, C) = block_cpu[Index][PointInBlock.Y * BlockSize + PointInBlock.X][C].item().toFloat();
			}
		}
	}
#endif
}

bool IsFileExist(const char* filepath)
{
	return std::filesystem::exists(filepath);
}

std::string ToString(const std::vector<float> Values, const std::string& Split)
{
	std::ostringstream oss; // 创建一个字符串流对象

	for (size_t i = 0; i < Values.size(); ++i) {
		oss << Values[i]; // 将当前值写入字符串流
		if (i < Values.size() - 1) {
			oss << Split; // 如果不是最后一个元素，添加分隔符
		}
	}

	return oss.str(); // 返回构建的字符串
}

void SaveStringToFile(const std::string& String, const char* FilePath)
{
	// 创建一个输出文件流
	std::ofstream OutFile(FilePath);

	// 检查文件是否成功打开
	if (!OutFile) {
		std::cerr << "Error opening file: " << FilePath << std::endl;
		return;
	}

	// 将字符串写入文件
	OutFile << String;

	// 关闭文件
	OutFile.close();
}
