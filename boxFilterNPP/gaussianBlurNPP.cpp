#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>
#include <npp.h>
#include <FreeImage.h>

#include <helper_cuda.h>
#include <helper_string.h>

namespace fs = std::filesystem;

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    // Print out the NPP library version being used
    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    // Print out CUDA driver and runtime versions
    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

bool processImage(const std::string &sFilename, const std::string &outputFolder)
{
    try
    {
        std::cout << "Processing image: " << sFilename << std::endl;

        // Declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // Load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);

        // Check if image loaded correctly
        if (oHostSrc.data() == nullptr)
        {
            std::cerr << "Error loading image: " << sFilename << std::endl;
            return false;
        }

        // Validate image dimensions
        if (oHostSrc.width() <= 0 || oHostSrc.height() <= 0)
        {
            std::cerr << "Invalid image dimensions for file: " << sFilename << std::endl;
            return false;
        }

        std::cout << "Image dimensions: " << oHostSrc.width() << "x" << oHostSrc.height() << std::endl;

        // Declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // Check for device image data
        if (oDeviceSrc.data() == nullptr)
        {
            std::cerr << "Error uploading image to device: " << sFilename << std::endl;
            return false;
        }

        // Create struct with Gaussian filter mask size
        NppiSize oMaskSize = {5, 5};

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};

        // Create struct with ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        // Allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

        // Check for device destination image data
        if (oDeviceDst.data() == nullptr)
        {
            std::cerr << "Error allocating device destination image: " << sFilename << std::endl;
            return false;
        }

        // Set mask size
        NppiMaskSize eMaskSize = NPP_MASK_SIZE_5_X_5;

        std::cout << "Running Gaussian blur filter..." << std::endl;

        // Run Gaussian blur filter
        NppStatus status = nppiFilterGaussBorder_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, eMaskSize,
            NPP_BORDER_REPLICATE);

        if (status != NPP_SUCCESS)
        {
            std::cerr << "NPP_CHECK_NPP - eStatusNPP = " << status << " for file: " << sFilename << std::endl;
            return false;
        }

        std::cout << "Filter applied successfully." << std::endl;

        // Synchronize CUDA operations
        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "CUDA synchronization error: " << cudaGetErrorString(cudaStatus) << " for file: " << sFilename << std::endl;
            return false;
        }

        // Declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // And copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        // Save the image using FreeImage
        std::string sResultFilename = outputFolder + "/" + fs::path(sFilename).stem().string() + "_gaussianBlur.png";
        FIBITMAP *image = FreeImage_ConvertFromRawBits(oHostDst.data(), oHostDst.width(), oHostDst.height(), oHostDst.pitch(),
                                                       8, 0, 0, 0, TRUE);
        FreeImage_Save(FIF_PNG, image, sResultFilename.c_str(), 0);
        FreeImage_Unload(image);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        // Free NPP image memory
        cudaFree(oDeviceSrc.data());
        cudaFree(oDeviceDst.data());

        // Wait for a short period to ensure all operations are complete
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Reset CUDA device to clear any residual states
        cudaDeviceReset();

        std::cout << "Finished processing image: " << sFilename << std::endl;

        return true;
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        return false;
    }
    catch (std::exception &e)
    {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        std::cerr << "Aborting." << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        return false;
    }
}

void printCudaDeviceProperties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        // Print out the CUDA device properties
        std::cout << "Device " << device << " name: " << deviceProp.name << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Memory pitch: " << deviceProp.memPitch << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dim: [" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", "
                  << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max grid size: [" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << "]" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    if (printfNPPinfo(argc, argv) == false)
    {
        exit(EXIT_SUCCESS);
    }

    printCudaDeviceProperties();

    if (!checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
        std::cerr << "Input folder is required.\n";
        exit(EXIT_FAILURE);
    }

    char *inputFolderPath;
    getCmdLineArgumentString(argc, (const char **)argv, "input", &inputFolderPath);
    std::string outputFolder = "output";

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
        char *outputFolderPath;
        getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFolderPath);
        outputFolder = outputFolderPath;
    }

    fs::create_directory(outputFolder);

    const int maxRetries = 3; // Number of retries for each image

    for (const auto &entry : fs::directory_iterator(inputFolderPath))
    {
        if (fs::is_regular_file(entry) && entry.path().extension() == ".pgm")
        {
            bool success = false;
            for (int attempt = 1; attempt <= maxRetries; ++attempt)
            {
                success = processImage(entry.path().string(), outputFolder);
                if (success)
                    break;
                std::cerr << "Retrying (" << attempt << "/" << maxRetries << ") for image: " << entry.path().string() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait for 1 second before retrying
            }
            if (!success)
            {
                std::cerr << "Failed to process image after " << maxRetries << " attempts: " << entry.path().string() << std::endl;
            }
        }
    }

    exit(EXIT_SUCCESS);
}
