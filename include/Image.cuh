#ifndef IMAGE_CUH
#define IMAGE_CUH

#include "common_includes.h"
#include "io_util.h"
#include "cuda_util.cuh"
#include "Quadtree.cuh"
#include "Unity.cuh"

namespace jax{
  struct Image_Descriptor{
    int id;
    uint2 size;
    float3 cam_pos;
    float3 cam_vec;
    float fov;
    float foc;
    float dpix;
    long long int timeStamp;//seconds since Jan 01, 1070
    __device__ __host__ Image_Descriptor();
    __device__ __host__ Image_Descriptor(int id, uint2 size);
    __device__ __host__ Image_Descriptor(int id, uint2 size, float3 cam_pos, float3 camp_dir);
  };


  class Image{

  public:

    Image_Descriptor descriptor;
    std::string filePath;
    Quadtree<unsigned char>* quadtree;//holds pixels

    Image();
    Image(std::string filePath, int id = -1);
    Image(std::string filePath, unsigned int convertColorDepthTo, int id = -1);
    Image(std::string filePath, unsigned int convertColorDepthTo, unsigned int quadtreeBinDepth, int id = -1);
    ~Image();
  };

  void get_cam_params2view(Image_Descriptor &cam1, Image_Descriptor &cam2, std::string infile);
  void convertToBW(Unity<unsigned char>* pixels, unsigned int colorDepth);

  /* CUDA variable, method and kernel defintions */

  __device__ __forceinline__ unsigned long getGlobalIdx_2D_1D();
  __device__ __forceinline__ unsigned char bwaToBW(const uchar2 &color);
  __device__ __forceinline__ unsigned char rgbToBW(const uchar3 &color);
  __device__ __forceinline__ unsigned char rgbaToBW(const uchar4 &color);
  __global__ void generateBW(int numPixels, unsigned int colorDepth, unsigned char* colorPixels, unsigned char* bwPixels);

}

#endif /* IMAGE_CUH */
