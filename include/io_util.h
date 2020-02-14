/** 
* \file io_util.h
* \brief This file contains image io methods.
*/
#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include "Unity.cuh"
#include "tinyply.h"
#include "Octree.cuh"
#include <stdio.h>

namespace jax{
  /**
  * \defgroup file_io 
  * \{
  */
  /*
  ARG PARSING
  */
  /**
   * \brief Map for command line flag and argument type association.
   * \details This map is filled out in io_util.cpp and contains all 
   * possible command line argument flags and the associated argument identifier. 
   * This map is primarily purposed to help parsArgs() method.
   * \see parseArgs
   */
extern std::map<std::string, std::string> cl_args;

void toLower(std::string &str);

/**
   * \brief Determines if a file path exists. 
   * \details This method takes in an absolute path and 
   * returns true if there is infact a file with that path. 
   * \param fileName - the absolute path to the file in question
   * \returns true if the path is a file, false otherwise
   */
bool fileExists(std::string fileName);

/**
   * \brief Determines if a directory path exists.
   * \details This method takes in an aboslute path 
   * to a directory and returns true if it exists. 
   * \param dirPath - the aboslute path to the directory
   * \returns true if the path is a directory, false otherwise
   */
bool directoryExists(std::string dirPath);

/**
   * \brief Extracts the file extension from a file path.
   * \details This method simply takes the string that comes 
   * after the last occurance of a '.'
   * \param path - the path to a file, could be absolute or relative 
   * \returns a string containing the file extension\
   * \todo add a fileExists() check to this 
   */
std::string getFileExtension(std::string path);

/**
   * \brief Returns the folder of a file give a fully qualified filepath.
   * \param path a string representing a fully qualified filepath
   * \return string which is the fully qualified folder path
   */
std::string getFolderFromFilePath(std::string path);

/**
   * \brief Returns the filename from a fully qualified filepath.
   * \param path a string representing a fully qualified filepath
   * \return string which is the filename only
   */
std::string getFileFromFilePath(std::string path);

void getImagePaths(std::string dirPath, std::vector<std::string> &imagePaths);

/**
   * \brief Base arg struct for arg parsing purposes. 
   */
struct arg
{};

  /**
   * \brief arg containing an image path 
   */
  struct img_arg : public arg{
    std::string path;
    img_arg(char* path);
  };
  /**
   * \brief arg containing a vector of image paths
   */
  struct img_dir_arg : public arg{
    std::vector<std::string> paths;
    img_dir_arg(char* path);
  };
  /**
   * \brief arg containing a floating point value 
   */
  struct flt_arg : public arg{
    float val;
    flt_arg(char* val);
  };
  /**
   * \brief arg containing an integer value 
   */
  struct int_arg : public arg{
    int val;
    int_arg(char* val);
  };

  typedef std::pair<std::string, arg*> arg_pair;

  std::vector<std::string> findFiles(std::string path);//going to be deprecated

  /**
   * \brief Parses command line arguments into a map that can be easily processed in main().
   * \details 
   * \returns a map of string argument type identifiers and the argument  
   */
  std::map<std::string, arg*> parseArgs(int numArgs, char* args[]);


  /*
  IMAGE IO
  */
  
  /**
  * \brief Get pixel array from a row of pointers generated from jax::readPNG utilizing png.h.
  * \details This method is utilized by readPNG and writePNG for help in writing images. 
  * \param row_pointers - pointers to rows of pixels in an image
  * \param width - width of image in pixels
  * \param height - height of image in pixels 
  * \param numValues - value to help with colorDepth determination
  * \returns a row-wise flattened pixel array
  */
  unsigned char* getPixelArray(unsigned char** &row_pointers, const unsigned int &width, const unsigned int &height, const int numValues);

  /**
  * \brief Reads a png image and generates a pixel array.
  * \details This method will read a PNG image utilizing libpng and fill in the passed-by-reference arguments 
  * height, width and colorDepth. 
  * \param filePath - const char* filePath for location of image (<string>.c_str() is easiest way to use a string path)
  * \param height - image height in pixels (reference argument that will be filled in during reading of image)
  * \param width - image width in pixels (reference argument that will be filled in during reading of image)
  * \param colorDepth - number of unsigned char values per pixel (reference argument that will be filled in during reading of image)
  * \returns a pixel array flattened row-wise with dimensions filled out in width, height, and colorDepth reference arguments
  */
  unsigned char* readPNG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);

  /**
  * \brief Writes a png image from a pixel array.
  * \details This method will write a PNG image from a row-wise flattened pixel array using libpng.
  * \param filePath const char* filePath where image is to be written (<string>.c_str() is easiest way to use a string path)
  * \param image - unsigned char array with pixels values flattened row wise
  * \param colorDepth - number of unsigned chars per pixel value
  * \param width - number of pixels in a row
  * \param height - number of rows in an image
  */
  void writePNG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);

  /**
  * \brief Reads a tiff image and generates a pixel array. 
  * \details This method will read a JPG/JPEG image utilizing libjpeg and fill in the passed-by-reference arguments 
  * height, width and colorDepth. 
  * \param filePath - const char* filePath for location of image (<string>.c_str() is easiest way to use a string path)
  * \param height - image height in pixels (reference argument that will be filled in during reading of image)
  * \param width - image width in pixels (reference argument that will be filled in during reading of image)
  * \param colorDepth - number of unsigned char values per pixel (reference argument that will be filled in during reading of image)
  * \returns a pixel array flattened row-wise with dimensions filled out in width, height, and colorDepth reference arguments
  */
  unsigned char* readTIFF(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);
  /**
  * \brief Writes a tiff image from a pixel array.
  * \details This method will write a JPG/JPEG image from a row-wise flattened pixel array using libjpeg.
  * \param filePath - const char* filePath where image is to be written (<string>.c_str() is easiest way to use a string path)
  * \param image - unsigned char array with pixels values flattened row wise
  * \param colorDepth - number of unsitin compression parameters
  */
  void writeTIFF(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);

  /**
  * \brief Reads a jpeg image and generates a pixel array.
  * \details This method will read a TIF/TIFF image utilizing libtiff and fill in the passed-by-reference arguments 
  * height, width and colorDepth. 
  * \param filePath - const char* filePath for location of image (<string>.c_str() is easiest way to use a string path)
  * \param height - image height in pixels (reference argument that will be filled in during reading of image)
  * \param width - image width in pixels (reference argument that will be filled in during reading of image)
  * \param colorDepth - number of unsigned char values per pixel (reference argument that will be filled in during reading of image)
  * \returns a pixel array flattened row-wise with dimensions filled out in width, height, and colorDepth reference arguments
  */
  unsigned char* readJPEG(const char* filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);
  /**
  * \brief Writes a jpeg image from a pixel array.
  * \details This method will write a TIF/TIFF image from a row-wise flattened pixel array using libtiff.
  * \param filePath - const char* filePath where image is to be written (<string>.c_str() is easiest way to use a string path)
  * \param image - unsigned char array with pixels values flattened row wise
  * \param colorDepth - number of unsigned chars per pixel value
  * \param width - number of pixels in a row
  * \param height - number of rows in an image
  */
  void writeJPEG(const char* filePath, unsigned char* image, const unsigned int &colorDepth, const unsigned int &width, const unsigned int &height);

  /**
  * \brief Reads an image and generates a pixel array.
  * \details This method will read a PNG, JPG/JPEG or TIF/TIFF image and fill in the passed-by-reference arguments 
  * height, width and colorDepth. 
  * \param filePath - const char* filePath for location of image (<string>.c_str() is easiest way to use a string path)
  * \param height - image height in pixels (reference argument that will be filled in during reading of image)
  * \param width - image width in pixels (reference argument that will be filled in during reading of image)
  * \param colorDepth - number of unsigned char values per pixel (reference argument that will be filled in during reading of image)
  * \returns a pixel array flattened row-wise with dimensions filled out in width, height, and colorDepth reference arguments
  */
  unsigned char* readImage(const char *filePath, unsigned int &height, unsigned int &width, unsigned int &colorDepth);

  /*
  FEATURE AND MATCH IO
  */
  //add match writes here

  /*
  PLY IO
  */

  //TODO make readPLY
  /**
  * \brief Will write a ply file based on a set of float3 values.
  * \details This method will write a ply in the specified location and can 
  * be written in binary or ASCII format.
  * \param filePath - path where image will be written (<string>.c_str() is easiest way to use a string path)
  * \param points - a Unity<float3>* where the points are listed
  * \param binary - bool signifying if ply should be written in binary or ASCII format. (optional, default is ASCII)
  * \see Unity
  */
  void writePLY(const char* filePath, Unity<float3>* points, bool binary = false);

  /*
  CSV and Misc Debug IO
  */

  /**
   * \brief Takes in an array of floats and writes them to a CSV.
   * \param values a set of float elements as a float array that are written in csv format on one line
   * \param num the number of elements in the float array
   * \param filename a string representing the desired filename of the csv output
   */
  void writeCSV(float* values, int num, std::string filename);

  //
  // Binary files - Gitlab #58
  //

  /*
   * Since the Image class and Image_Descriptor struct are both CUDified, I'll leave them that way for now and make this struct here
   *
   * Until further documentation is established, this struct declaration is the official specification of the .bcp format.
   * A .bcp file will be a binary serialization of the structure below in order from top to bottom.
   *
   * This definition is coupled with the reading method in io_util.cpp and the loading method in Image.cuh
   */
  struct bcpFormat {
  	float pos[3];
  	float vec[3];
  	float fov[2];
    float foc;
    float dpix[2];
  };

  bool readImageMeta(std::string imgpath, bcpFormat & out);



  /**\}*/ 
}

#endif /* IO_UTIL_H */
