#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <cmath>
#include <tuple>
#include <stdio.h>
#include <cassert>
#include <typeinfo>
#include "tqdm.h"

using std::cout;
using std::endl;
using std::min;
using std::string;
using std::tuple;
using std::get;
using torch::Tensor;

const float max_pixel = 255.0;
at::DeviceType device;

tuple<int, int, int, Tensor> image_ToTensor(char* image_path, int num = -1) {
	char targetString[256];
	int realLen = 0;
	if (num == -1)
	{
		realLen = strlen(image_path);
		strcpy(targetString, image_path);
	}
	else
		realLen = snprintf(targetString, sizeof(targetString), image_path, num);
	assert(realLen<256);
	int iw, ih, n;
	unsigned char* idata = stbi_load(targetString, &iw, &ih, &n, 0);
	//cout << targetString << endl;
	float* data = new float[iw * ih * n];
	for (int i = 0; i < ih; ++i)
		for (int j = 0; j < iw; ++j)
		{
			for (int k = 0; k < n; ++k)
			{
				int pixel_index = i * (iw * n) + j * n + k;
				data[pixel_index] = (float)(idata[pixel_index]) / max_pixel;
				//cout << data[pixel_index] <<' ';
			}
			//cout << endl;
		}
	stbi_image_free(idata);
	Tensor output_tensor = torch::from_blob(data, { 1, ih, iw, n });
	//cout << output_tensor << endl;
	return { iw ,ih ,n , output_tensor };
}

void data_generate(int start, int end, char* name) {
	char x_path[] = { ".\\data_x\\%d.png" };
	char y_path[] = { ".\\data_y\\%d_decompressed.png" };
	
	Tensor x = torch::empty(0, torch::TensorOptions().dtype(torch::kFloat32));
	Tensor y = torch::empty(0, torch::TensorOptions().dtype(torch::kFloat32));
	for (int i : tqdm::range(start, end))
	//for (int i=start;i<end;++i)
	{
		//cout << start << ' '<< end<<endl;
		tuple<int, int, int, Tensor> res;
		res = image_ToTensor(x_path, i);
		Tensor x_img_tensor = get<3>(res);
		x = torch::cat({ x, x_img_tensor }, 0);
		//cout << x << endl;
		res = image_ToTensor(y_path, i);
		Tensor y_img_tensor = get<3>(res);
		y_img_tensor = (y_img_tensor - x_img_tensor).abs().sum().unsqueeze(0);
		y = torch::cat({ y, y_img_tensor }, 0);
	}
	char save_path[] = { ".\\%s_%s.pth" };
	char target_path[256];
	int realLen = snprintf(target_path, sizeof(target_path), save_path, name,"x");
	assert(realLen < 256);
	torch::save(x, target_path);
	realLen = snprintf(target_path, sizeof(target_path), save_path, name, "y");
	assert(realLen < 256);
	torch::save(y, target_path);
}

void data() {
	//划分训练集, 验证集, 测试集
	int sum_num = 136900;
	int train_num = (int)round(sum_num * 0.76);
	int valid_num = (int)round(sum_num * 0.12);
	int test_num = (int)round(sum_num * 0.12);
	data_generate(0, train_num, "train");
	data_generate(train_num, train_num + valid_num, "valid");
	data_generate(train_num + valid_num, min(train_num + valid_num + test_num, sum_num), "test");
}

class myDataset :public torch::data::Dataset<myDataset, torch::data::Example<>>
{
public:
	myDataset(torch::Tensor data, torch::Tensor target) {
		this->data = std::move(data);
		this->target = std::move(target);
	}
	torch::data::Example<> get(size_t index) override {
		return	{ data[index].clone(), target[index].clone() };
	}
	torch::optional<size_t> size() const override {
		return data.sizes()[0];
	}
	std::vector<ExampleType> get_batch(at::ArrayRef<size_t> indices) override {
		std::vector<ExampleType> batch;
		batch.resize(1);
		torch::Tensor tensor_indices = torch::empty(indices.size(), torch::TensorOptions().dtype(torch::kInt64));
		int now = 0;
		for (const auto i : indices)
		{
			tensor_indices[now] = (int64_t)i;
			now++;
		}
		batch[0].data = data.index({ tensor_indices });
		batch[0].target = target.index({ tensor_indices });
		return batch;
	}
	torch::Tensor data, target;
};

struct Net : torch::nn::Module {
	Net() {
		conv1 = register_module("conv1", torch::nn::Conv2d(4, 4, 1));
		bn1 = register_module("bn1", torch::nn::BatchNorm2d(4));
		conv2 = register_module("conv2", torch::nn::Conv2d(4, 4, 2));
		bn2 = register_module("bn2", torch::nn::BatchNorm2d(4));
		//conv3 = register_module("conv3", torch::nn::Conv2d(4, 4, 2));
		//bn3 = register_module("bn3", torch::nn::BatchNorm2d(4));

		fc1 = register_module("fc1", torch::nn::Linear(4 * 3 * 3, 256));
		fc2 = register_module("fc2", torch::nn::Linear(256, 64));
		//fc3 = register_module("fc3", torch::nn::Linear(256, 64));
		fc4 = register_module("fc4", torch::nn::Linear(64, 1));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(conv1->forward(x));
		x = bn1->forward(x);
		x = torch::relu(conv2->forward(x));
		x = bn2->forward(x);
		//x = torch::relu(conv3->forward(x));
		//x = bn3->forward(x);

		x = x.reshape({ -1, 4 * 3 * 3 });
		x = torch::relu(fc1->forward(x));
		//x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc2->forward(x));
		//x = torch::relu(fc3->forward(x));
		x = torch::relu(fc4->forward(x));
		x = x.squeeze(1);
		//x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
		return x;
	}

	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr };
	torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };
};

tuple<torch::Tensor, torch::Tensor> astc_error_prediction(char* input_x_path) {
	tuple<int, int, int, Tensor> res = image_ToTensor(input_x_path);
	Tensor input_x = get<3>(res);
	input_x = input_x.to(device);
	input_x = torch::autograd::Variable(input_x).set_requires_grad(true);
	auto model = std::make_shared<Net>();
	model->to(device);
	torch::load(model, ".\\pth\\36_model.pth");
	model->eval();
	torch::Tensor pred = model->forward(input_x);
	pred.backward();
	return { pred, input_x.grad() };
}

int main()
{
	//if (torch::cuda::is_available())
	//	device = at::kCUDA;
	//else
		device = at::kCPU;

	//data();

	tuple<torch::Tensor, torch::Tensor> res = astc_error_prediction("D:\\Workspace\\NeuralAidedMBD\\data_x\\100.png");
	cout << get<0>(res) << ' ' << get<1>(res) << endl;
	exit(0);

	torch::manual_seed(1);
	torch::cuda::manual_seed(1);
	torch::cuda::manual_seed_all(1);

	Tensor train_x, train_y, valid_x, valid_y, test_x, test_y;
	torch::load(train_x ,".\\train_x.pth");
	torch::load(train_y, ".\\train_y.pth");
	torch::load(valid_x, ".\\valid_x.pth");
	torch::load(valid_y, ".\\valid_y.pth");
	torch::load(test_x, ".\\test_x.pth");
	torch::load(test_y, ".\\test_y.pth");
	std::cout << "1" << std::endl;
	train_x = train_x.to(device);
	std::cout << "2" << std::endl;
	train_y = train_y.to(device);
	std::cout << "3" << std::endl;
	valid_x = valid_x.to(device);
	std::cout << "4" << std::endl;
	valid_y = valid_y.to(device);
	std::cout << "5" << std::endl;
	test_x = test_x.to(device);
	std::cout << "6" << std::endl;
	test_y = test_y.to(device);
	std::cout << "7" << std::endl;

	torch::data::DataLoaderOptions options;
	auto train_ds = myDataset(std::move(train_x), std::move(train_y));
	auto val_ds = myDataset(std::move(valid_x), std::move(valid_y));
	auto test_ds = myDataset(std::move(test_x), std::move(test_y));
	auto train_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_ds), options.batch_size(256));
	auto val_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(val_ds), options.batch_size(1024));
	auto test_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_ds), options.batch_size(1));
	//cout << typeid(*train_dl).name() << endl;
	auto model = std::make_shared<Net>();
	model->to(device);
	torch::nn::L1Loss loss_fn;
	torch::nn::L1Loss test_fn(torch::nn::L1LossOptions().reduction(torch::kSum));
	torch::optim::Adam optimizer(model->parameters(), /*lr=*/0.01);

	std::cout << "8" << std::endl;

	//测试
	torch::load(model, ".\\pth\\36_model.pth");
	model->eval();
	float test_loss = 0;
	int sum = 0;
	{
		c10::InferenceMode guard(true);
		for (auto& batch : tqdm::tqdm(*test_dl))
		{
			//cout << batch[0].data << endl;
			torch::Tensor pred = model->forward(batch[0].data);
			torch::Tensor loss = test_fn(pred, batch[0].target);
			test_loss += loss.item().toFloat();
			sum += batch[0].target.sizes()[0];
		}
		printf("Test Error: Avg acy: %8f\n", test_loss / sum);
	}
	exit(0);

	int epoch = 1000000, min_epoch = 0;
	float min_loss = 1e9;
	for (int i = 0; i < epoch; ++i)
	{
		printf("Epoch %d\n-------------------------------\n",i);
		model->train();
		float avg_loss = 0;
		int times = 0, sum = 0;
		for (auto& batch : tqdm::tqdm(*train_dl))
		{
			optimizer.zero_grad();
			torch::Tensor pred = model->forward(batch[0].data);
			torch::Tensor loss = loss_fn(pred, batch[0].target);
			loss.backward();
			optimizer.step();
			avg_loss += loss.item().toFloat() * batch[0].target.sizes()[0];
			times++;
			sum += batch[0].target.sizes()[0];
		}
		printf("Train Error %d: Avg_loss: %.8f\n",i, avg_loss / sum);
		model->eval();
		avg_loss = 0;
		sum = 0;
		float val_error = 0, valid_loss=0;
		{
			c10::InferenceMode guard(true);
			for (auto& batch : tqdm::tqdm(*val_dl))
			{
				torch::Tensor pred = model->forward(batch[0].data);
				torch::Tensor loss = test_fn(pred, batch[0].target);
				avg_loss += loss.item().toFloat();
				torch::Tensor tmp = torch::abs(pred - batch[0].target) / (batch[0].target + 1e-2);
				val_error += torch::sum(tmp).item().toFloat();
				sum += batch[0].target.sizes()[0];
			}
			valid_loss = avg_loss / sum;
			val_error /= sum;
			if (valid_loss < min_loss)
			{
				min_loss = valid_loss;
				min_epoch = i;
			}
			printf("Valid Error(%d): Avg_loss: %8f Avg_error: %8f min_loss:%8f min_epoch:%d\n",i, valid_loss, val_error,min_loss,min_epoch);
			if (i % 1 == 0)
			{
				char name[] = { ".\\pth\\%d_model.pth" };
				char targetString[256];
				int realLen = snprintf(targetString, sizeof(targetString), name, i);
				assert(realLen < 256);
				torch::save(model, targetString);
				printf("Saved PyTorch Model State to model.pth\n\n");
			}
		}
	}
	return 0;
}