#include <UMBD/UMBDSolver.h>
#include <UMBD_ext/UMBDCeresSolver.h>
//#include "glog/logging.h"

#include <fstream>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    //google::InitGoogleLogging(argv[0]);

    UCommon::FThreadPool ThreadPool;
    UCommon::FThreadPoolRegistry::GetInstance().Register(&ThreadPool);

    UMBD::FCeresSolver CeresSolver;
    UMBD::FSolverRegistry::GetInstance().Register(&CeresSolver);

    const size_t D = 3;
    const size_t L = 2;
    UMBD::FGrid GridF(128, 128, 1);
    UMBD::FGrid GridB(16, 16, 1);
    UMBD::FGrid GridC = GridF;
    UMBD::FTex3D f(GridF, D, UMBD::EElementType::Float);

    for (size_t x = 0; x < f.GetGrid().Width; x++) {
        for (size_t y = 0; y < f.GetGrid().Height; y++) {
            float u = x / (float)(f.GetGrid().Width - 1);
            float v = y / (float)(f.GetGrid().Height - 1);
            f.At<UMBD::FLinearColorRGB>({ x,y,0 }) = { u,v * v,std::sqrt((u + v * v) / 2) };
        }
    }
    UMBD::FTex3D half_f = f.DownSampleOnPlane();

    UMBD::FSolverConfig config;
    config.bDebugInfo = true;
    config.InitMode = UMBD::FSolverConfig::EInitMode::GlobalPCA;
    config.MaxNumIterations = 1024;
    UMBD::FCompressedData compressedData(D, L, GridB, GridC, UMBD::EElementType::Uint8);
    UMBD::Solve(config, f, compressedData);

    UMBD::FTex3D approx_f = compressedData.GetApproxF(f.GetGrid());

    UMBD::FCompressedData half_compressedData = compressedData.DownSampleCoeffOnPlane();
    UMBD::FTex3D approx_half_f = half_compressedData.GetApproxF(half_f.GetGrid());

    double error1 = 0.;
    double error2 = 0.;
    for (size_t i = 0; i < f.GetGrid().GetOuterVolume() * f.GetNumChannel(); i++) {
        error1 += std::abs(approx_f.At<float>(i) - f.At<float>(i));
        error2 += std::pow(approx_f.At<float>(i) - f.At<float>(i), 2);
    }
    double avg_error1 = error1 / f.GetGrid().GetOuterVolume() / f.GetNumChannel();
    double avg_error2 = error2 / f.GetGrid().GetOuterVolume() / f.GetNumChannel();

    std::cout << "avg_error1 : " << avg_error1 << std::endl;
    std::cout << "avg_error2 : " << avg_error2 << std::endl;

    { // output f
        std::ofstream out_f("out.ppm");
        out_f << "P3" << std::endl
            << f.GetGrid().Width << " " << f.GetGrid().Height << std::endl
            << "255" << std::endl;
        for (size_t x = 0; x < f.GetGrid().Width; x++) {
            for (size_t y = 0; y < f.GetGrid().Height; y++) {
                UMBD::FUint64Vector RGB(f.At<UMBD::FLinearColorRGB>({ x, y, 0 }).Clamp(0.f, 1.f) * 255.999f);
                out_f << RGB.X << " " << RGB.Y << " " << RGB.Z << std::endl;
            }
        }
    }

    { // output half f

        std::ofstream out_half_f("out_half.ppm");
        out_half_f << "P3" << std::endl
            << half_f.GetGrid().Width << " " << half_f.GetGrid().Height << std::endl
            << "255" << std::endl;
        for (size_t x = 0; x < half_f.GetGrid().Width; x++) {
            for (size_t y = 0; y < half_f.GetGrid().Height; y++) {
                UMBD::FUint64Vector RGB(half_f.At<UMBD::FLinearColorRGB>({ x, y, 0 }).Clamp(0.f, 1.f) * 255.999f);
                out_half_f << RGB.X << " " << RGB.Y << " " << RGB.Z << std::endl;
            }
        }
    }

    { // output approx_f
        std::ofstream out_approx_f("out_approx_f.ppm");
        out_approx_f << "P3" << std::endl
            << approx_f.GetGrid().Width << " " << approx_f.GetGrid().Height << std::endl
            << "255" << std::endl;
        for (size_t x = 0; x < approx_f.GetGrid().Width; x++) {
            for (size_t y = 0; y < approx_f.GetGrid().Height; y++) {
                UMBD::FUint64Vector RGB(approx_f.At<UMBD::FLinearColorRGB>({ x, y, 0 }).Clamp(0.f, 1.f) * 255.999f);
                out_approx_f << RGB.X << " " << RGB.Y << " " << RGB.Z << std::endl;
            }
        }
    }

    { // output approx_half_f
        std::ofstream out_approx_half_f("out_approx_half_f.ppm");
        out_approx_half_f << "P3" << std::endl
            << approx_half_f.GetGrid().Width << " " << approx_half_f.GetGrid().Height << std::endl
            << "255" << std::endl;
        for (size_t x = 0; x < approx_half_f.GetGrid().Width; x++) {
            for (size_t y = 0; y < approx_half_f.GetGrid().Height; y++) {
                UMBD::FUint64Vector RGB(approx_half_f.At<UMBD::FLinearColorRGB>({ x, y, 0 }).Clamp(0.f, 1.f) * 255.999f);
                out_approx_half_f << RGB.X << " " << RGB.Y << " " << RGB.Z << std::endl;
            }
        }
    }

    // output B

    for (size_t l = 0; l < L; l++) {
        const UMBD::FTex3D& B = compressedData.GetB();
        std::ofstream out_B_l("out_B_" + std::to_string(l) + ".ppm");
        out_B_l << "P3" << std::endl
            << B.GetGrid().Width << " " << B.GetGrid().Height << std::endl
            << "255" << std::endl;
        for (size_t x = 0; x < B.GetGrid().Width; x++) {
            for (size_t y = 0; y < B.GetGrid().Height; y++) {
                size_t r = static_cast<size_t>(255.999f * UCommon::Clamp(B.GetFloat({ x, y, 0 }, l * 3 + 0), 0.f, 1.f));
                size_t g = static_cast<size_t>(255.999f * UCommon::Clamp(B.GetFloat({ x, y, 0 }, l * 3 + 1), 0.f, 1.f));
                size_t b = static_cast<size_t>(255.999f * UCommon::Clamp(B.GetFloat({ x, y, 0 }, l * 3 + 2), 0.f, 1.f));
                out_B_l << r << " " << g << " " << b << std::endl;
            }
        }
    }

    // output c
    for (size_t l = 0; l < L; l++) {
        std::ofstream out_c_l("out_c_" + std::to_string(l) + ".ppm");
        out_c_l << "P3" << std::endl
            << compressedData.GetC().GetGrid().Width << " " << compressedData.GetC().GetGrid().Height << std::endl
            << "255" << std::endl;
        for (size_t x = 0; x < compressedData.GetC().GetGrid().Width; x++) {
            for (size_t y = 0; y < compressedData.GetC().GetGrid().Height; y++) {
                size_t r = static_cast<size_t>(255.999f * UCommon::Clamp(compressedData.GetC().GetFloat({ x, y, 0 }, l), 0.f, 1.f));
                size_t g = static_cast<size_t>(255.999f * UCommon::Clamp(compressedData.GetC().GetFloat({ x, y, 0 }, l), 0.f, 1.f));
                size_t b = static_cast<size_t>(255.999f * UCommon::Clamp(compressedData.GetC().GetFloat({ x, y, 0 }, l), 0.f, 1.f));
                out_c_l << r << " " << g << " " << b << std::endl;
            }
        }
    }

    UMBD::FSolverRegistry::GetInstance().Deregister();
    UCommon::FThreadPoolRegistry::GetInstance().Deregister();

    return 0;
}
