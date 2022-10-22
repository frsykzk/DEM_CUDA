#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <omp.h> 
#include <iostream>
#include <vector>
#include <fstream>
#include <time.h>
#include <iomanip>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono>
#include <random>


#define IN_FILE_1 "DEM_physical.txt"
#define IN_FILE_2 "initial.txt"

#define GST -1//�͈͊O���q
#define SLD 0//�ő̗��q
#define WLL 1//�Ǘ��q
#define OBJ 2//�Ǘ��q

#define NCP 20//�ő�ڐG��

#define pi 3.141592f

#define THREADS 256

#define freq 0.5f //�ǂ̉�]��/�b �ォ�猩�ā@+���v��� -�����v���
#define X 0.250f
#define Z 0.250f //��]���S���W

typedef float real;
//typedef double real;

typedef struct {
	real x, y, z;
}treal3;

typedef struct {
	real max, min;
}treal_m2;

typedef struct {
	real* x;
	real* y;
	real* z;
}areal3;


