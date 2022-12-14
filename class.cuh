#pragma once

#include "const.cuh"

class DEM {
public:
	///////////Φ////////////////
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
	///////////Φ////////////////

	///////////physical.txt////////////////
	real PCL_DST;//±qTCY
	treal3 MINc;//vZΝΝmin
	treal3 MAXc;//vZΝΝmax
	treal3 G;//dΝ
	real FIN_TIM;//vZIΉ^C
	real dt;//ΤέΤu
	real output_time;//t@CoΝΤΤu
	real eta;//½­W
	real mu;//CW
	real dns;//ΕΜ§x
	///////////physical.tst////////////////

	///////////±qf[^////////////////
	areal3 Pos;//ΐW
	areal3 d_Pos;
	areal3 Vel;//¬x
	areal3 d_Vel;
	areal3 Ftotal;//ΪGΝ
	areal3 d_Ftotal;
	areal3 Omega;//p¬x
	areal3 d_Omega;
	areal3 Torque;//gN
	areal3 d_Torque;
	real* wallangle;//Ηρ]p
	real* d_wallangle;
	real* rot_rad;//Ηρ]Όa
	real* d_rot_rad;

	areal3 ep;
	areal3 d_ep;
	real* ep_r;
	real* d_ep_r;
	int* pair;
	int* d_pair;

	char* Typ;//±q^Cv@ΕΜO@ΗP
	char* d_Typ;
	///////////±qf[^////////////////


	int iF;//t@CΤ
	real TIM;//»έΜvZΤ
	real outtime = 0.0f;

	///////////±qTυp////////////////
	real DB, DB2, DBinv;//±qTυΜͺ@ΜubNΜ
	int nBx, nBy, nBz, nBxy, nBxyz;//NXgp
	//int* bfst, * blst, * nxt;
	int* d_bfst, * d_blst, * d_nxt;//NXgp
	///////////±qTυp////////////////

	real m;//²Μ±qΏΚ
	int nP;//ϊ±q
	int nPWLL;
	int nPSLD;
	int nPOBJ;

	real eta_n;//S«ΈW
	real eta_t;
	real eta_r;

	real k;
	real kn;
	real kt;
	real kr;

	real I;//²Μ΅«[g

	char outout_filename[256];

	real* D;//±qΌazρ
	real* d_D;

	real umax;

	real mu_r;//ρ]CW

	real w;//ρ]Ηp¬x

	FILE* fp;
};

