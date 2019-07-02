#ifndef UNITY_CUH
#define UNITY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
      file, line, cudaGetErrorString(err));
      exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

namespace jax{
  struct IllegalUnityTransition{
    std::string msg;
    IllegalUnityTransition(){
      msg = "Illegal Unity memory transfer";
    }
    IllegalUnityTransition(std::string msg) : msg("Illegal Unity memory transfer: " + msg){}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };

  struct NullUnityException{
    std::string msg;
    NullUnityException(){
      msg = "Illegal attempt to use null set Unity";
    }
    NullUnityException(std::string msg) : msg("Illegal attempt to use null set Unity: " + msg){}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };

  typedef enum MemoryState{
    null = 0,
    cpu = 1,
    gpu = 2,
    both = 3,
    pinned = 4
  } MemoryState;

  template<typename T>
  class Unity{

  public:
    MemoryState fore;//can be used to keep track of recently updated memory
    MemoryState state;

    T* device;
    T* host;
    unsigned long numElements;

    Unity();
    Unity(T* data, unsigned long numElements, MemoryState state);
    ~Unity();

    void setData(T* data, unsigned long numElements, MemoryState state);//hard set
    void setMemoryState(MemoryState state);//hard set
    void clear();//hard clear
    void transferMemoryTo(MemoryState state);//soft set
    void clearHost();//hard clear
    void clearDevice();//hard clear
  };

  template<typename T>
  Unity<T>::Unity(){
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->numElements = 0;
  }
  template<typename T>
  Unity<T>::Unity(T* data, unsigned long numElements, MemoryState state){
    this->host = nullptr;
    this->device = nullptr;
    this->state = state;
    this->fore = state;
    this->numElements = numElements;
    if(state == cpu) this->host = data;
    else if(state == gpu) this->device = data;
    else{
      throw IllegalUnityTransition("cannot instantiate memory on device and host with only one pointer");
    }
  }
  template<typename T>
  Unity<T>::~Unity(){
    this->clear();
  }

  template<typename T>
  void Unity<T>::setData(T* data, unsigned long numElements, MemoryState state){
    this->clear();
    this->state = state;
    this->fore = state;
    this->numElements = numElements;
    if(state == cpu) this->host = data;
    else if(state == gpu) this->device = data;
    else{
      throw IllegalUnityTransition("cannot instantiate memory on device and host with only one pointer");
    }
  }
  template<typename T>
  void Unity<T>::setMemoryState(MemoryState state){
    if((this->state == null || this->numElements == 0) && state != null){
      throw NullUnityException();
    }
    else if(this->state == state) return;
    else if(state == null) this->clear();
    else if(state == both){
      if(this->state == cpu){
        if(this->device == nullptr){
          CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
        }
        CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      }
      if(this->state == gpu){
        if(this->host == nullptr){
          this->host = new T[this->numElements];
        }
        CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      }
    }
    else if(state == gpu){
      if(this->device == nullptr){
        CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
      }
      CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      delete[] this->host;
      this->host = nullptr;
    }
    else if(state == cpu){
      if(this->host == nullptr){
        this->host = new T[this->numElements];
      }
      CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(this->device));
      this->device = nullptr;
    }
    else{
      throw IllegalUnityTransition("unkown memory state");
    }
    this->state = state;
    this->fore = state;
  }
  template<typename T>
  void Unity<T>::clear(){
    switch(this->state){
      case null:
        break;
      case cpu:
        if(this->host != nullptr){
          delete[] this->host;
        }
        break;
      case gpu:
        if(this->device != nullptr){
          CudaSafeCall(cudaFree(this->device));
        }
        break;
      case both:
        if(host != nullptr){
          delete[] this->host;
        }
        if(device != nullptr){
          CudaSafeCall(cudaFree(this->device));
        }
        break;
      default:
        throw IllegalUnityTransition("unkown memory state");
    }
    this->host = nullptr;
    this->device = nullptr;
    this->state = null;
    this->fore = null;
    this->numElements = 0;
  }
  template<typename T>
  void Unity<T>::transferMemoryTo(MemoryState state){
    if((this->state == null || sizeof(T)*this->numElements == 0) && state != null){
      throw NullUnityException();
    }
    else if(state == null){
      throw IllegalUnityTransition("Cannot transfer unity memory to null");
    }
    else if(this->state == state) return;
    else if(state == both){
      if(this->state == cpu){
        if(this->device == nullptr){
          CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
        }
        CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      }
      if(this->state == gpu){
        if(this->host == nullptr){
          this->host = new T[this->numElements];
        }
        CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      }
      this->state = both;
      this->fore = both;
    }
    else if(state == gpu){
      if(this->device == nullptr){
        CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
      }
      CudaSafeCall(cudaMemcpy(this->device, this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      this->state = both;
      this->fore = both;
    }
    else if(state == cpu){
      if(this->host == nullptr){
        this->host = new T[this->numElements];
      }
      CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      this->state = both;
      this->fore = both;
    }
    else{
      throw IllegalUnityTransition("unkown memory state");
    }
  }
  template<typename T>
  void Unity<T>::clearHost(){
    if(this->host != nullptr){
      operator delete(this->host);
    }
    if(this->state == cpu){
      this->state = null;
      this->fore = null;
    }
    else if(this->state == both){
      this->state = gpu;
      this->fore = gpu;
    }
  }
  template<typename T>
  void Unity<T>::clearDevice(){
    if(this->device != nullptr){
      CudaSafeCall(cudaFree(this->device));
      this->device = nullptr;
    }
    if(this->state == gpu){
      this->state = null;
      this->fore = null;
    }
    else if(this->state == both){
      this->state = cpu;
      this->fore = cpu;
    }
  }

}

#endif /*UNITY_CUH*/
