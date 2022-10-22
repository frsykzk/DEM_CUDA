#pragma once

#include "const.cuh"

class DEM {
public:
	///////////関数////////////////
	void RdDat();
	void WrtDat();
	void WrtDatWLL();
	void WrtDat2();
	void AlcBkt();
	void SetPara();
	void MkBkt();
	void ClcDEM();
	void ColForce();
	void update();
	void memory_free();
	void Output_vtk(char, int);
	void Output_vtk2(char, int);
	void DevicetoHost();
	void checkPCL();
	///////////関数////////////////

	///////////physical.txt////////////////
	real PCL_DST;//粒子サイズ
	treal3 MINc;//計算範囲min
	treal3 MAXc;//計算範囲max
	treal3 G;//重力
	real FIN_TIM;//計算終了タイム
	real dt;//時間刻み間隔
	real output_time;//ファイル出力時間間隔
	real eta;//反発係数
	real mu;//摩擦係数
	real dns;//固体密度
	///////////physical.tst////////////////

	///////////粒子データ////////////////
	areal3 Pos;//座標
	areal3 d_Pos;
	areal3 Vel;//速度
	areal3 d_Vel;
	areal3 Ftotal;//接触力
	areal3 d_Ftotal;
	areal3 Omega;//角速度
	areal3 d_Omega;
	areal3 Torque;//トルク
	areal3 d_Torque;
	real* wallangle;//壁回転角
	real* d_wallangle;
	real* rot_rad;//壁回転半径
	real* d_rot_rad;

	areal3 ep;
	areal3 d_ep;
	real* ep_r;
	real* d_ep_r;
	int* pair;
	int* d_pair;

	char* Typ;//粒子タイプ　固体０　壁１
	char* d_Typ;
	///////////粒子データ////////////////


	int iF;//ファイル番号
	real TIM;//現在の計算時間
	real outtime = 0.0f;

	///////////粒子探索用////////////////
	real DB, DB2, DBinv;//粒子探索の分割法のブロックの幅
	int nBx, nBy, nBz, nBxy, nBxyz;//リンクリスト用
	//int* bfst, * blst, * nxt;
	int* d_bfst, * d_blst, * d_nxt;//リンクリスト用
	///////////粒子探索用////////////////

	real m;//粉体粒子質量
	int nP;//初期粒子数
	int nPWLL;
	int nPSLD;
	int nPOBJ;

	real eta_n;//粘性減衰係数
	real eta_t;
	real eta_r;

	real k;
	real kn;
	real kt;
	real kr;

	real I;//粉体慣性モーメント

	char outout_filename[256];

	real* D;//粒子直径配列
	real* d_D;

	real umax;

	real mu_r;//回転摩擦係数

	real w;//回転壁角速度

	FILE* fp;
};

