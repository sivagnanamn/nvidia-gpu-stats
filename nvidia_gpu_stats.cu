// ================================================================================================
// A simple script to get memory usage & properties of CUDA supported NVIDIA devices
//
// Author: Sivagnanam Namasivayamurthy
//
// ================================================================================================

#include <stdio.h>
#include <time.h>

// CUDA C headers
#include <cuda.h>
#include <cuda_runtime.h>

/*
 *
 */
void printCurrentTime(){
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);

  printf("\n%d-%d-%d %d:%d:%d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

/*
 * Get number of CUDA supported devices available
 *
 * returns: the number of CUDA supported devices
 */
int getCudaDevicesCount(){

  int deviceCount;

  cudaError_t cu_err = cudaGetDeviceCount(&deviceCount);
  if(cudaSuccess != cu_err){
    printf("Unable to get cudaGetDeviceCount : error num %d - %s\n", (int) cu_err, cudaGetErrorString(cu_err));
    exit(EXIT_FAILURE);
  }
  return deviceCount;
}

/*
 * Print the number of CUDA supported devices available
 */
void printDevicesCount(){

  int deviceCount = getCudaDevicesCount();

  if(0 == deviceCount){
    printf("No CUDA supported device(s) found !!! \n");
    exit(EXIT_FAILURE);
  } else {
    printf("Found %d CUDA supported device(s)\n", deviceCount);
  }
}

/*
 * Get device properties
 *
 * deviceId : ID of the CUDA supported device

 * returns: the cudaDeviceProp struct that contains the CUDA device properties
 */
cudaDeviceProp getCudaDeviceProps(int deviceId){

  cudaDeviceProp deviceProps;

  cudaError_t cu_err = cudaGetDeviceProperties(&deviceProps, deviceId);
  if(cudaSuccess != cu_err){
    printf("Unable to get cudaGetDeviceProperties for device ID %d : error num %d - %s\n", deviceId, (int) cu_err, cudaGetErrorString(cu_err));
    exit(EXIT_FAILURE);
  }

  return deviceProps;
}

/*
 * Print the CUDA device properties
 *
 * deviceId: ID of the CUDA supported device
 */
void printCudaDeviceProps(int deviceId)
{
  cudaDeviceProp deviceProps = getCudaDeviceProps(deviceId);
  printf("\n Device ID: %d       Name:      %s\n",  deviceId, deviceProps.name);
  printf("--------------------------------------------------------------\n");
  printf("CUDA capability Major/Minor version number:         %d.%d\n",  deviceProps.major, deviceProps.minor);
  printf("Total global memory:                                %0.f MB\n", (float)deviceProps.totalGlobalMem/(1048576.0f));
  printf("Total shared memory per block:                      %lu bytes\n", deviceProps.sharedMemPerBlock);
  printf("Total registers per block:                          %d\n",  deviceProps.regsPerBlock);
  printf("Warp size:                                          %d\n",  deviceProps.warpSize);
  printf("Maximum memory pitch:                               %lu bytes\n", deviceProps.memPitch);
  printf("Maximum threads per block:                          %d\n",  deviceProps.maxThreadsPerBlock);
  printf("Maximum sizes of each dimension of a block:         %d x %d x %d \n", deviceProps.maxThreadsDim[0], deviceProps.maxThreadsDim[1], deviceProps.maxThreadsDim[2]);
  printf("Maximum sizes of each dimension of a grid:          %d x %d x %d  \n", deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
  printf("Clock rate:                                         %d\n",  deviceProps.clockRate);
  printf("Total constant memory:                              %lu\n", deviceProps.totalConstMem);
  printf("Texture alignment:                                  %lu bytes\n", deviceProps.textureAlignment);
  printf("Concurrent copy and execution:                      %s\n",  (deviceProps.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:                          %d\n",  deviceProps.multiProcessorCount);
  printf("Kernel execution timeout:                           %s\n",  (deviceProps.kernelExecTimeoutEnabled ? "Yes" : "No"));
  printf("--------------------------------------------------------------\n");
}

/*
 * Find all CUDA supported devices & print their properties
 */
void printAllCudaDeviceProps(){
  int deviceCount = getCudaDevicesCount();
  printCurrentTime();
  printf("\nFound %d CUDA supported device(s)\n", deviceCount);

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId){
    printCudaDeviceProps(deviceId);
  }
}

/*
 * Print the free & used memory information of all NVIDIA CUDA supported devices
 */
void printNvidiaDevicesMemoryInfo(){

  int driverVersion, runTimeVersion, deviceCount;
  size_t mem_available, mem_free;

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runTimeVersion);
  deviceCount = getCudaDevicesCount();

  printCurrentTime();
  printf("\n+----------------------------------------------------------------------+\n");
  printf("| CUDA Driver version: %d.%d               Runtime Version: %d.%d          |\n", driverVersion/1000, (driverVersion % 100)/10, runTimeVersion/1000, (runTimeVersion%100)/10);

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId){

    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProps = getCudaDeviceProps(deviceId);
    cudaMemGetInfo(&mem_free, &mem_available);
    printf("+----------------------------------------------------------------------------------------+\n");
    printf("| Device ID: %d      Name: %s           %.0f MB (free) / %.0f MB (total)    |\n", deviceId, deviceProps.name, (float)mem_free/(1024*1024.), (float)mem_available/(1024*1024.));
    printf("+----------------------------------------------------------------------------------------+\n");

  }

}

int main(int argc, char *argv[])
{
  if(argc < 2) {
    printNvidiaDevicesMemoryInfo(); // Print memory information if not argument is passed
  } else {
    if (0 == strcmp(argv[1], "-mem")){
      printNvidiaDevicesMemoryInfo();
    } else if (0 == strcmp(argv[1], "-props")) {
      printAllCudaDeviceProps();
    }
  }
  return 0;
}
