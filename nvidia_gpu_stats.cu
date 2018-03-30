// ================================================================================================
// A simple script to get memory usage & properties of CUDA supported NVIDIA devices
//
// Author: Sivagnanam Namasivayamurthy
//
// ================================================================================================

#include <stdio.h>

// CUDA C headers
#include <cuda.h>
#include <cuda_runtime.h>

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
  printf("\nFound %d CUDA supported device(s)\n", deviceCount);

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId){
    printCudaDeviceProps(deviceId);
  }
}

void printDeviceMemory(){
  size_t mem_available, mem_free;
  cudaMemGetInfo(&mem_free, &mem_available);
  printf("Memory available (MB):             %f\n",  mem_available/(1024*1024.));
  printf("Memory free (MB):             %f\n",  mem_free/(1024*1024.));
}

int main()
{
    printAllCudaDeviceProps();
    return 0;
}
