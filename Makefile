CUDA_INSTALL_PATH := /usr/local/cuda

# CUDA stuff
CXX := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I./include -I/usr/local/cuda/include

# Common flags
COMMONFLAGS += ${INCLUDES}
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -std=c++11
# compute_<#> and sm_<#> will need to change depending on the device
# if this is not done you will receive a no kernel image is availabe error
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_61,code=sm_61

LIB_CUDA := -L/usr/local/cuda/lib64 -lcublas -lcudart


SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS = ImageTree.cu.o
_OBJS += QuadTreeTest.cu.o



OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = reconstruction.exe
LINKLINE = ${LINK} -gencode=arch=compute_61,code=sm_61 ${OBJS} ${LIB_CUDA} -o ${BINDIR}/${TARGET}


.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET}

$(OBJDIR):
	    -mkdir -p $(OBJDIR)

$(BINDIR):
	    -mkdir -p $(BINDIR)

#-------------------------------------------------------------
#  Cuda Cuda Reconstruction
#
${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} -dc $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

${BINDIR}/${TARGET}: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f out/*.ply
	rm -f bin/*
	rm -f out/*
	rm -f src/*~
	rm -f util/*~
	rm -f obj/*
	rm -f util/*.o
	rm -f util/io/*~
	rm -f util/examples/*~
	rm -f uril/CI/*~
	rm -f .DS_Store
	rm -f *._*
	rm -f *.~
	rm -f *.kp
	rm -f *.txt
