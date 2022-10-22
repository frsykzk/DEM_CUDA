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

#define GST -1//範囲外粒子
#define SLD 0//固体粒子
#define WLL 1//壁粒子
#define OBJ 2//壁粒子

#define NCP 20//最大接触数

#define pi 3.141592f

#define THREADS 256

#define freq 0.5f //壁の回転数/秒 上から見て　+時計回り -反時計回り
#define X 0.250f
#define Z 0.250f //回転中心座標

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


