/*
MIT License

Copyright (c) 2021 Ubpa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <UMBD_ext/UMBDCeresSolver.h>

#include <torch/torch.h>

#include <functional>
#include <memory>
#include "Compressor.h"

class FNerualFirstOrderFunction : public UMBD::FFirstOrderFunction
{
public:
	struct FData
	{
	public:
		at::DeviceType device = at::kCPU;
		Compressor* Compressor = NULL;
		int64_t BlockSize = 4;
		uint64_t Iterations = 0;
	};
	
	FNerualFirstOrderFunction(
		UMBD::FThreadPool* pThreadPool,
		double Theta,
		const double* ThetaWeights,
		const double* CustomWeights,
		const double* BRanges,
		bool bCGradientWall,
		const UMBD::FTex3D* TexF,
		UMBD::FGrid InGridB,
		UMBD::FGrid InGridC,
		uint64_t InL,
		uint64_t InD,
		bool bDebugInfo,
		const FData* InData);

	virtual bool Evaluate(const double* const parameters, double* cost, double* gradient) const override;

private:
	UMBD::FGrid GridB;
	UMBD::FGrid GridC;
	uint64_t L;
	uint64_t D;
	FData Data;
};

class FNeuralSolver : public UMBD::FCeresSolver
{
public:
	virtual UMBD::FFirstOrderFunction* CreateFirstOrderFunction(
		double Theta,
		const double* ThetaWeights,
		const double* CustomWeights,
		const double* BRanges,
		bool bCGradientWall,
		const UMBD::FTex3D* TexF,
		UMBD::FGrid GridB,
		UMBD::FGrid GridC,
		uint64_t L,
		uint64_t D,
		bool bDebugInfo) override;

	FNerualFirstOrderFunction::FData FunctionData;
};

class INerualIterationCallback: public UMBD::IIterationCallback
{
public:
	INerualIterationCallback(FNeuralSolver* solver)
	{
		this->solver = solver;
	}
	virtual ECallbackReturnType operator()(const UMBD::FIterationSummary& summary)
	{
		solver->FunctionData.Iterations = summary.Iteration;
		return ECallbackReturnType::SolverContinue;
	}
	FNeuralSolver* solver;
};
