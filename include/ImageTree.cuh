#ifndef IMAGETREE_CUH
#define IMAGETREE_CUH

#include "common_includes.h"
#include "Unity.cuh"
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

template<typename D>
struct LocalizedData{
  float2 loc;
  D data;
};


//TODO make ImageTree exceptions and add to a namespace

__global__ void generateLeafNodes(Node* leafNodes, unsigned int width, uint2 imageSize, int depth);



__global__ void getNodeKeys(float2* nodeCenters, int* nodeKeys, unsigned int width, uint2 imageSize, bool* hashMap);
template<typename L>
__global__ void getNodeKeys(float2* nodeCenters, int* nodeKeys, unsigned int width, unsigned int numElements, L* locations);
template<typename D>
__global__ void getNodeKeys(float2* nodeCenters, int* nodeKeys, unsigned int width, unsigned int numElements, LocalizedData<D>* LocalizedData);


//TODO make depth variable

template<typename T>
class ImageTree{

  void generateLeafNodes(int depth = -1);
  void generateLeafNodes(jax::Unity<bool>* hashMap);

  void generateParentNodes();


public:

  uint2 imageSize;
  int width;

  struct Node;
  struct Vertex;
  struct Edge;

  jax::Unity<T>* data;
  jax::Unity<Node>* nodes;
  jax::Unity<Vertex>* vertices;
  jax::Unity<Edge>* edges;

  jax::Unity<unsigned int>* nodeDepthIndex;
  jax::Unity<unsigned int>* vertexDepthIndex;
  jax::Unity<unsigned int>* edgeDepthIndex;

  ImageTree();

  ImageTree(uint2 imageSize, jax::Unity<T>* data);
  ImageTree(uint2 imageSize, jax::Unity<bool>* hashMap, jax::Unity<T>* data);
  ImageTree(jax::Unity<int2>* data);
  ImageTree(jax::Unity<float2>* data);
  ImageTree(jax::Unity<LocalizedData<T>>* data);

  ~ImageTree();

};

template<typename T>
struct ImageTree<T>::Vertex{

  float2 loc;
  int nodes[4];
  int depth;

  __device__ __host__ Vertex();
};

template<typename T>
struct ImageTree<T>::Edge{

  int2 vertices;
  int nodes[2];
  int depth;

  __device__ __host__ Edge();
};

template<typename T>
struct ImageTree<T>::Node{
  int key;

  int dataIndex;
  int numElements;
  float2 center;
  int depth;//maybe remove as its derivable or change to char

  int parent;
  int children[4];
  int neighbors[9];

  int edges[4];
  int vertices[4];

  __device__ __host__ Node();
};


#endif /* IMAGETREE_CUH */
