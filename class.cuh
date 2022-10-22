#pragma once

#include "const.cuh"

class DEM {
public:
	///////////�֐�////////////////
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
	///////////�֐�////////////////

	///////////physical.txt////////////////
	real PCL_DST;//���q�T�C�Y
	treal3 MINc;//�v�Z�͈�min
	treal3 MAXc;//�v�Z�͈�max
	treal3 G;//�d��
	real FIN_TIM;//�v�Z�I���^�C��
	real dt;//���ԍ��݊Ԋu
	real output_time;//�t�@�C���o�͎��ԊԊu
	real eta;//�����W��
	real mu;//���C�W��
	real dns;//�ő̖��x
	///////////physical.tst////////////////

	///////////���q�f�[�^////////////////
	areal3 Pos;//���W
	areal3 d_Pos;
	areal3 Vel;//���x
	areal3 d_Vel;
	areal3 Ftotal;//�ڐG��
	areal3 d_Ftotal;
	areal3 Omega;//�p���x
	areal3 d_Omega;
	areal3 Torque;//�g���N
	areal3 d_Torque;
	real* wallangle;//�ǉ�]�p
	real* d_wallangle;
	real* rot_rad;//�ǉ�]���a
	real* d_rot_rad;

	areal3 ep;
	areal3 d_ep;
	real* ep_r;
	real* d_ep_r;
	int* pair;
	int* d_pair;

	char* Typ;//���q�^�C�v�@�ő̂O�@�ǂP
	char* d_Typ;
	///////////���q�f�[�^////////////////


	int iF;//�t�@�C���ԍ�
	real TIM;//���݂̌v�Z����
	real outtime = 0.0f;

	///////////���q�T���p////////////////
	real DB, DB2, DBinv;//���q�T���̕����@�̃u���b�N�̕�
	int nBx, nBy, nBz, nBxy, nBxyz;//�����N���X�g�p
	//int* bfst, * blst, * nxt;
	int* d_bfst, * d_blst, * d_nxt;//�����N���X�g�p
	///////////���q�T���p////////////////

	real m;//���̗��q����
	int nP;//�������q��
	int nPWLL;
	int nPSLD;
	int nPOBJ;

	real eta_n;//�S�������W��
	real eta_t;
	real eta_r;

	real k;
	real kn;
	real kt;
	real kr;

	real I;//���̊������[�����g

	char outout_filename[256];

	real* D;//���q���a�z��
	real* d_D;

	real umax;

	real mu_r;//��]���C�W��

	real w;//��]�Ǌp���x

	FILE* fp;
};

