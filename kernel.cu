#include "class.cuh"/////////////////////////////////////////////////////////////////////pairにする

void DEM::RdDat() {
	//////////////////////physical.txt///////////////////////////////
	FILE* in;
	if (fopen_s(&in, IN_FILE_1, "r") != 0) {
		printf("DEM_physical.txtが開けません\n");
	}
	else {
		real scan[50];
		fscanf_s(in, "%f %f %f", &scan[0], &scan[1], &scan[2]);//最小範囲
		fscanf_s(in, "%f %f %f", &scan[3], &scan[4], &scan[5]);//最大範囲
		fscanf_s(in, "%f %f %f", &scan[6], &scan[7], &scan[8]);//重力加速度
		fscanf_s(in, "%f", &scan[9]);//計算終了時間
		fscanf_s(in, "%f", &scan[10]);//時間刻み幅
		fscanf_s(in, "%f", &scan[11]);//ファイル吐き出し感覚
		fscanf_s(in, "%f", &scan[12]);//ばね定数
		fscanf_s(in, "%f", &scan[13]);//摩擦係数
		fscanf_s(in, "%f", &scan[14]);//回転摩擦係数
		fscanf_s(in, "%f", &scan[15]);//粒子密度
		fscanf_s(in, "%f", &scan[16]);//粒子サイズ
		fclose(in);

		PCL_DST = scan[16];
		MINc.x = real(scan[0] - 5.0f * PCL_DST);
		MINc.y = real(scan[1] - 5.0f * PCL_DST);
		MINc.z = real(scan[2] - 5.0f * PCL_DST);
		MAXc.x = real(scan[3] + 5.0f * PCL_DST);
		MAXc.y = real(scan[4] + 5.0f * PCL_DST);
		MAXc.z = real(scan[5] + 5.0f * PCL_DST);
		G.x = real(scan[6]);
		G.y = real(scan[7]);
		G.z = real(scan[8]);
		FIN_TIM = real(scan[9]);
		dt = real(scan[10]);
		output_time = real(scan[11]);
		k = real(scan[12]);
		mu = real(scan[13]);
		mu_r = real(scan[14]);
		dns = real(scan[15]);

	}
	fclose(in);

	printf("MINc.x = %f  MINc.y = %f  MINc.z = %f\n", MINc.x, MINc.y, MINc.z);
	printf("MAXc.x = %f  MAXc.y = %f  MAXc.z = %f\n", MAXc.x, MAXc.y, MAXc.z);
	printf("G.x = %f  G.y = %f  G.z = %f\n", G.x, G.y, G.z);
	printf("FIN_TIM = %f\n", FIN_TIM);
	printf("dt = %.10f\n", dt);
	printf("output_time = %f\n", output_time);
	printf("k = %f\n", k);
	printf("mu = %f\n", mu);
	printf("mu_r = %f\n", mu_r);
	printf("dns = %f\n", dns);
	printf("PCL_DST = %f\n", PCL_DST);

	//////////////////////physical.txt///////////////////////////////

	//////////////////////initial.txt///////////////////////////////
	FILE* in2;
	if (fopen_s(&in2, IN_FILE_2, "r") != 0) {
		printf("initial.txtが開けません\n");
	}
	else {
		fscanf_s(in2, "%d", &nP);//総粒子数取得
		//int nPnew = memory;
		Pos.x = (real*)malloc(sizeof(real) * (nP));
		Pos.y = (real*)malloc(sizeof(real) * (nP));
		Pos.z = (real*)malloc(sizeof(real) * (nP));

		Vel.x = (real*)malloc(sizeof(real) * (nP));
		Vel.y = (real*)malloc(sizeof(real) * (nP));
		Vel.z = (real*)malloc(sizeof(real) * (nP));

		Omega.x = (real*)malloc(sizeof(real) * (nP));
		Omega.y = (real*)malloc(sizeof(real) * (nP));
		Omega.z = (real*)malloc(sizeof(real) * (nP));

		Ftotal.x = (real*)malloc(sizeof(real) * (nP));
		Ftotal.y = (real*)malloc(sizeof(real) * (nP));
		Ftotal.z = (real*)malloc(sizeof(real) * (nP));

		Torque.x = (real*)malloc(sizeof(real) * (nP));
		Torque.y = (real*)malloc(sizeof(real) * (nP));
		Torque.z = (real*)malloc(sizeof(real) * (nP));

		Typ = (char*)malloc(sizeof(char) * (nP));

		D = (real*)malloc(sizeof(real) * (nP));

		wallangle = (real*)malloc(sizeof(real) * (nP));
		rot_rad = (real*)malloc(sizeof(real) * (nP));

		int nPsolid = 0; int nPwall = 0; int nPobj = 0;
		int nPtmp = 0;
		for (int i = 0; i < nP; i++) {
			int a[1];
			float b[11];
			int c[1];
			float g[1];
			fscanf_s(in2, " %d %d %f %f %f %f %f %f %f", &a[0], &c[0], &b[0], &b[1], &b[2], &b[8], &b[9], &b[10], &g[0]);
			const treal3 pos = { b[0], b[1], b[2] };
			if (pos.x<MAXc.x && pos.x>MINc.x && pos.y<MAXc.y && pos.y>MINc.y && pos.z<MAXc.z && pos.z>MINc.z) {
				Typ[nPtmp] = char(c[0]);
				Pos.x[nPtmp] = real(b[0]); Pos.y[nPtmp] = real(b[1]); Pos.z[nPtmp] = real(b[2]);
			}
			if (Typ[nPtmp] == SLD) { nPsolid += 1; }
			if (Typ[nPtmp] == WLL) { nPwall += 1; }
			if (Typ[nPtmp] == OBJ) { nPobj += 1; }
			nPtmp += 1;
		}
		nP = nPtmp;
		nPWLL = nPwall;
		nPSLD = nPsolid;
		nPOBJ = nPobj;
		std::cout << "総固体粒子数 nPSLD = " << nPsolid << std::endl;
		std::cout << "総壁粒子数 nPWLL = " << nPwall << std::endl;
		std::cout << "総動壁粒子数 nPOBJ = " << nPobj << std::endl;
		std::cout << "総粒子数 nP = " << nP << std::endl;


		ep.x = (real*)malloc(sizeof(real) * (NCP * nPSLD));
		ep.y = (real*)malloc(sizeof(real) * (NCP * nPSLD));
		ep.z = (real*)malloc(sizeof(real) * (NCP * nPSLD));
		ep_r = (real*)malloc(sizeof(real) * (NCP * nPSLD));
		pair = (int*)malloc(sizeof(int) * (NCP * nPSLD));//粉体粒子数i*最大接触数k i+k*i

	}
	fclose(in2);
	//////////////////////initial.txt///////////////////////////////
}

void DEM::Output_vtk(const char typ, const int outputflg) {
	if (outputflg == 0)
	{
		sprintf(outout_filename, "./vtk_cuda/output%05d.csv", iF);
		printf("Filename = %s\n", outout_filename);

		if (fopen_s(&fp, outout_filename, "w") != 0) {
			printf("%sが開けません\n", outout_filename);
		}
		/*else {
			fprintf_s(fp, "Pos.x,Pos.y,Pos.z\n");
			for (int i = 0; i < nP; i++) {
				if (Typ[i] == typ) {
					fprintf(fp, "%f,%f,%f,%f\n", Pos.x[i], Pos.y[i], Pos.z[i]);
				}
			}
		}*/
		else {
			fprintf_s(fp, "Pos.x,Pos.y,Pos.z,Vel\n");
			for (int i = 0; i < nP; i++) {
				if (Typ[i] == typ) {
					fprintf(fp, "%f,%f,%f,%f\n", Pos.x[i], Pos.y[i], Pos.z[i], sqrt(Vel.x[i] * Vel.x[i] + Vel.y[i] * Vel.y[i] + Vel.z[i] * Vel.z[i]));
				}
			}
		}
		fclose(fp);
	}

	if (outputflg == 1)
	{
		sprintf(outout_filename, "./vtk_cuda/outputwall%05d.csv", iF);
		printf("Filename = %s\n", outout_filename);

		if (fopen_s(&fp, outout_filename, "w") != 0) {
			printf("%sが開けません\n", outout_filename);
		}
		else {
			fprintf_s(fp, "Pos.x,Pos.y,Pos.z\n");
			for (int i = 0; i < nP; i++) {
				if (Typ[i] == typ) {
					fprintf_s(fp, "%f,%f,%f\n", Pos.x[i], Pos.y[i], Pos.z[i]);
				}
			}
		}
		fclose(fp);
	}

	if (outputflg == 2)
	{
		sprintf(outout_filename, "./vtk_cuda/outputobj%05d.csv", iF);
		printf("Filename = %s\n", outout_filename);

		if (fopen_s(&fp, outout_filename, "w") != 0) {
			printf("%sが開けません\n", outout_filename);
		}
		else {
			fprintf_s(fp, "Pos.x,Pos.y,Pos.z\n");
			for (int i = 0; i < nP; i++) {
				if (Typ[i] == typ) {
					fprintf_s(fp, "%f,%f,%f\n", Pos.x[i], Pos.y[i], Pos.z[i]);
				}
			}
		}
		fclose(fp);
	}
}

void DEM::Output_vtk2(const char typ, const int outputflg) {//txt seitei
	if (outputflg == 0)
	{
		sprintf(outout_filename, "./vtk_cuda/seitei.txt");
		printf("Filename = %s\n", outout_filename);

		if (fopen_s(&fp, outout_filename, "w") != 0) {
			printf("%sが開けません\n", outout_filename);
		}
		else {
			int sldnp = 0;
			for (int i = 0; i < nP; i++) {
				if (Typ[i] == SLD) { sldnp += 1; }//固体粒子カウント
			}
			fprintf(fp, "%d\n", sldnp);
			for (int i = 0; i < nP; i++) {
				if (Typ[i] == SLD) {
					fprintf(fp, "%d %d %f %f %f %f %f %f %f\n", i - nPWLL, Typ[i], Pos.x[i], Pos.y[i], Pos.z[i], 1.0, 1.0, 1.0, 1.0);
				}
			}
		}
		fclose(fp);
	}
}

void DEM::WrtDat(void) {
	printf("WrtDat_start\n");

	Output_vtk(SLD, 0);
//	Output_vtk(OBJ, 2);

}

void DEM::WrtDatWLL(void) {
	printf("WrtDatWLL_start\n");

	Output_vtk(WLL, 1);
}

void DEM::WrtDat2(void) {
	printf("WrtDat2_start\n");

	Output_vtk2(SLD, 0);
}

void DEM::AlcBkt() {//バケット作成

	DB = PCL_DST * 2.1f;
	DB2 = DB * DB;
	DBinv = 1.0f / DB;

	nBx = (int)((MAXc.x - MINc.x) * DBinv) + 3;
	nBy = (int)((MAXc.y - MINc.y) * DBinv) + 3;
	nBz = (int)((MAXc.z - MINc.z) * DBinv) + 3;

	nBxy = nBx * nBy;
	nBxyz = nBx * nBy * nBz;
	printf("nBx:%d  nBy:%d  nBz:%d  nBxy:%d  nBxyz:%d\n", nBx, nBy, nBz, nBxy, nBxyz);

	(cudaMalloc((void**)&d_bfst, sizeof(int) * nBxyz));
	(cudaMalloc((void**)&d_blst, sizeof(int) * nBxyz));
	(cudaMalloc((void**)&d_nxt, sizeof(int) * (nP)));


}

void DEM::SetPara() {

	m = dns * pi * PCL_DST * PCL_DST * PCL_DST / 6.0f;

	I = 0.1f * m * PCL_DST * PCL_DST;

	real alpha = 0.27;

	kn = k;
	kt = kn / (2.0f * (1.0f + alpha));
	//kr = kt;
	kr = 0.0f;

	eta_n = 2.0f * sqrt(m * kn);//5.0を2.0に戻した！！！！！！！！！摩擦弱めて摩擦角16度で試し中8/14
	eta_t = eta_n / sqrt(2.0f * (1.0f + alpha));
	//eta_r = eta_t;
	eta_r = 0.0f;

	w = 2.0f * pi * freq;

	kn = 1000;
	kt = kn * 0.25f;
	//kr = kt;
	kr = 0.0f;

	eta_n = sqrt(2.0f * m * kn);//5.0を2.0に戻した！！！！！！！！！摩擦弱めて摩擦角16度で試し中8/14
	eta_t = sqrt(2.0f * m * kt);
	//eta_r = eta_t;
	eta_r = 0.0f;


	printf("m:%.10f\nkn:%f  kt:%f  kr:%f\neta_n:%f  eta_t:%f  eta_r:%f\nfreq:%f\n\n", m, kn, kt, kr, eta_n, eta_t, eta_r, freq);

	iF = 0;
	TIM = 0.0f;

#pragma omp parallel for
	for (int i = 0; i < nP; i++) {
		Vel.x[i] = Vel.y[i] = Vel.z[i] = 0.0f;
		Ftotal.x[i] = Ftotal.y[i] = Ftotal.z[i] = 0.0f;
		Omega.x[i] = Omega.y[i] = Omega.z[i] = 0.0f;//初期値与えてランダム性出せる？
		Torque.x[i] = Torque.y[i] = Torque.z[i] = 0.0f;

		if (Typ[i] == WLL) { D[i] = 1.0f * PCL_DST; }
		else if (Typ[i] == OBJ) { D[i] = 1.0f * PCL_DST; }//壁粒子でかく
		else if (Typ[i] == SLD) { D[i] = PCL_DST; }

		wallangle[i] = atan2(Pos.z[i] - Z, Pos.x[i] - X);//初期角度計算　-pi ~ +piで角度返す
		rot_rad[i] = sqrt((Pos.x[i] - X) * (Pos.x[i] - X) + (Pos.z[i] - Z) * (Pos.z[i] - Z));
	}
	
	//音速とタイムステップ
	real max_height = MINc.y;
	real min_height = MAXc.y;
	for (int i = 0; i < nP; i++) {//初期化
		if (Typ[i] == SLD) {
			if (max_height < Pos.y[i]) { max_height = Pos.y[i]; }
			else if (min_height > Pos.y[i]) { min_height = Pos.y[i]; }
		}
	}
	umax = sqrt(2.0f * abs(G.y) * (max_height - min_height));

#pragma omp parallel for
	for (int i = 0; i < nPSLD; i++) { for (int k = 0; k < NCP; k++) { ep.x[k + i * NCP] = 0; } }
#pragma omp parallel for
	for (int i = 0; i < nPSLD; i++) { for (int k = 0; k < NCP; k++) { ep.y[k + i * NCP] = 0; } }
#pragma omp parallel for
	for (int i = 0; i < nPSLD; i++) { for (int k = 0; k < NCP; k++) { ep.z[k + i * NCP] = 0; } }
#pragma omp parallel for
	for (int i = 0; i < nPSLD; i++) { for (int k = 0; k < NCP; k++) { ep_r[k + i * NCP] = 0; } }
#pragma omp parallel for
	for (int i = 0; i < nPSLD; i++) { for (int k = 0; k < NCP; k++) { pair[k + i * NCP] = -2; } }

	(cudaMalloc((void**)&d_Typ, sizeof(char) * nP));
	(cudaMalloc((void**)&d_D, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Pos.x, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Pos.y, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Pos.z, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Vel.x, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Vel.y, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Vel.z, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Ftotal.x, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Ftotal.y, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Ftotal.z, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Omega.x, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Omega.y, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Omega.z, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Torque.x, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Torque.y, sizeof(real) * nP));
	(cudaMalloc((void**)&d_Torque.z, sizeof(real) * nP));
	(cudaMalloc((void**)&d_ep.x, sizeof(real) * (NCP * nPSLD)));
	(cudaMalloc((void**)&d_ep.y, sizeof(real) * (NCP * nPSLD)));
	(cudaMalloc((void**)&d_ep.z, sizeof(real) * (NCP * nPSLD)));
	(cudaMalloc((void**)&d_ep_r, sizeof(real) * (NCP * nPSLD)));
	(cudaMalloc((void**)&d_pair, sizeof(int) * (NCP * nPSLD)));
	(cudaMalloc((void**)&d_wallangle, sizeof(real) * nP));
	(cudaMalloc((void**)&d_rot_rad, sizeof(real) * nP));

	(cudaMemcpy(d_Typ, Typ, sizeof(char) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_D, D, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Pos.x, Pos.x, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Pos.y, Pos.y, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Pos.z, Pos.z, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Vel.x, Vel.x, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Vel.y, Vel.y, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Vel.z, Vel.z, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Ftotal.x, Ftotal.x, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Ftotal.y, Ftotal.y, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Ftotal.z, Ftotal.z, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Omega.x, Omega.x, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Omega.y, Omega.y, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Omega.z, Omega.z, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Torque.x, Torque.x, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Torque.y, Torque.y, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_Torque.z, Torque.z, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ep.x, ep.x, sizeof(real) * (NCP * nPSLD), cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ep.y, ep.y, sizeof(real) * (NCP * nPSLD), cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ep.z, ep.z, sizeof(real) * (NCP * nPSLD), cudaMemcpyHostToDevice));
	(cudaMemcpy(d_ep_r, ep_r, sizeof(real) * (NCP * nPSLD), cudaMemcpyHostToDevice));
	(cudaMemcpy(d_pair, pair, sizeof(int) * (NCP * nPSLD), cudaMemcpyHostToDevice));
	(cudaMemcpy(d_wallangle, wallangle, sizeof(real) * nP, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_rot_rad, rot_rad, sizeof(real) * nP, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
}

__global__ void d_initialize_int_array(const int n, int* i_array, const int a) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		i_array[i] = a;
	}
}

__global__ void d_MkBkt(const int nP, const  int nBx, const  int nBxy, const  real DBinv,
	int* d_bfst, int* d_blst, int* d_nxt,
	const char* d_Typ, const  areal3 d_Pos, const treal3 MINc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nP) {
		if (d_Typ[i] == GST) { return; }
		int ix = (int)((d_Pos.x[i] - MINc.x) * DBinv) + 1;
		int iy = (int)((d_Pos.y[i] - MINc.y) * DBinv) + 1;
		int iz = (int)((d_Pos.z[i] - MINc.z) * DBinv) + 1;
		int ib = iz * nBxy + iy * nBx + ix;
		const int j = atomicExch(&d_blst[ib], i);
		if (j == -1) { d_bfst[ib] = i; }
		else { d_nxt[j] = i; }
	}
}

void DEM::MkBkt() {//粒子をバケットに収納
	//printf("MkBkt\n");
	////////////////cudaスレッド設定/////////////////////
	dim3 threads(THREADS, 1, 1);
	int TOTAL_THREADS = nBxyz;	int BLOCKS = TOTAL_THREADS / THREADS + 1;
	dim3 blocks_nBxyz(BLOCKS, 1, 1);
	TOTAL_THREADS = (nP);	BLOCKS = TOTAL_THREADS / THREADS + 1;
	dim3 blocks_nP(BLOCKS, 1, 1);
	//////////////////////////////////////////////
	((d_initialize_int_array << <blocks_nBxyz, threads >> > (nBxyz, d_bfst, -1)));
	((d_initialize_int_array << <blocks_nBxyz, threads >> > (nBxyz, d_blst, -1)));
	((d_initialize_int_array << <blocks_nP, threads >> > (nP, d_nxt, -1)));
	cudaDeviceSynchronize();

	d_MkBkt << <blocks_nP, threads >> > (nP, nBx, nBxy, DBinv, d_bfst, d_blst, d_nxt, d_Typ, d_Pos, MINc);
}

__device__ int sign(real a, real b) {//aの絶対値にbの符号をつける
	int c = (b >= 0.0f) - (b <= 0.0f);
	return abs(a) * c;
}

__device__ real PairNumber(int* pair, int nPSLD, int nPWLL, int i, int j) {
	int contact;
	int count = 0;
	int k = i - nPWLL;//粉体粒子の前に壁粒子があるから

	for (int p = 0; p < NCP; p++) {//接触履歴あり　場所特定
		if (pair[p + k * NCP] == j) {
			contact = p;
			break;
		}
		else { count += 1; }
	}

	if (count == NCP) {//接触履歴なし　新規登録
		for (int q = 0; q < NCP; q++) {
			if (pair[q + k * NCP] == -2) {
				pair[q + k * NCP] = j;
				contact = q;
				break;
			}
		}
	}
	return contact;
}

__global__ void d_ColForce(const int nP, const int nPSLD, const int nPWLL, const real* d_D, const char* d_Typ, areal3 d_Pos, areal3 d_Vel, areal3 d_Ftotal, areal3 d_Omega, areal3 d_Torque,
	areal3 d_ep, real* d_ep_r, int* d_pair, const real m, const real eta_n, const real eta_t, real eta_r, real kn, real kt, real kr, const real mu, const real mu_r, const real dt,
	const treal3 MINc, const real DBinv, const int nBx, const int nBxy, const int* d_bfst, const int* d_blst, const int* d_nxt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nP) {
		if (d_Typ[i] == SLD) {
			real Di = d_D[i];
			treal3 Posi;	Posi.x = d_Pos.x[i];	Posi.y = d_Pos.y[i];	Posi.z = d_Pos.z[i];
			treal3 Veli;		Veli.x = d_Vel.x[i];		Veli.y = d_Vel.y[i];		Veli.z = d_Vel.z[i];
			treal3 Omegai;	Omegai.x = d_Omega.x[i];	Omegai.y = d_Omega.y[i];	Omegai.z = d_Omega.z[i];

			int ix = (int)((Posi.x - MINc.x) * DBinv) + 1;
			int iy = (int)((Posi.y - MINc.y) * DBinv) + 1;
			int iz = (int)((Posi.z - MINc.z) * DBinv) + 1;
			for (int jz = iz - 1; jz <= iz + 1; jz++) {
				for (int jy = iy - 1; jy <= iy + 1; jy++) {
					for (int jx = ix - 1; jx <= ix + 1; jx++) {
						int jb = jz * nBxy + jy * nBx + jx;
						int j = d_bfst[jb];
						if (j == -1) continue;
						for (;;) {//粒子iの近傍粒子jのループ開始
							if (j != i) {
								if (d_Typ[j] == GST) { continue; }
								treal3 r_delta_i;
								treal3 r_delta_j;
								treal3 Xi;

								treal3 dist;
								real L;
								real L_2;
								treal3 n;

								treal3 Fcol;
								real Tr = 0.0f;
								treal3 dp;
								real dp_r;
								real phi = 0.0f;

								real Dj = d_D[j];
								treal3 Posj;	Posj.x = d_Pos.x[j];	Posj.y = d_Pos.y[j];	Posj.z = d_Pos.z[j];
								treal3 Velj;		Velj.x = d_Vel.x[j];		Velj.y = d_Vel.y[j];		Velj.z = d_Vel.z[j];
								treal3 Omegaj;	Omegaj.x = d_Omega.x[j];	Omegaj.y = d_Omega.y[j];	Omegaj.z = d_Omega.z[j];

								dist.x = Posj.x - Posi.x;   dist.y = Posj.y - Posi.y;   dist.z = Posj.z - Posi.z;

								L = sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);

								n.x = dist.x / L;//l   
								n.y = dist.y / L;//m
								n.z = dist.z / L;//n

								L_2 = sqrt(n.x * n.x + n.y * n.y);//方向余弦の中身

								if (L - 0.5f * (Di + Dj) < 0.0f) {//接触判定

									int couple = PairNumber(d_pair, nPSLD, nPWLL, i, j) + (i - nPWLL) * NCP; //粒子i,jのpair配列番号
									treal3 ep;	ep.x = d_ep.x[couple];		ep.y = d_ep.y[couple];		ep.z = d_ep.z[couple];		real ep_r= d_ep_r[couple];

									if (L_2 == 0) {//ジンバルロック対策
										if (i < j) {
											r_delta_i.x = -Omegai.z * dt;
											r_delta_i.y = Omegai.y * dt;
											r_delta_i.z = Omegai.x * dt;
											r_delta_j.x = -Omegaj.z * dt;
											r_delta_j.y = Omegaj.y * dt;
											r_delta_j.z = Omegaj.x * dt;//角変位

											Xi.x = -(Veli.z - Velj.z) * dt;
											Xi.y = (Veli.y - Velj.y) * dt + (r_delta_i.z * Di + r_delta_j.z * Dj) * 0.5f;
											Xi.z = (Veli.x - Velj.x) * dt - (r_delta_i.y * Di + r_delta_j.y * Dj) * 0.5f;

											//////////////////////ローカルx///////////////////////////////
											ep.x += kn * Xi.x;
											dp.x = eta_n * Xi.x / dt;							
											if (ep.x < 0.0f) { ep.x = dp.x = 0.0f; }
											d_ep.x[couple] = ep.x;//バネ更新
											Fcol.x = ep.x + dp.x;
											//////////////////////ローカルx///////////////////////////////

											//////////////////////ローカルy///////////////////////////////
											ep.y += kt * Xi.y;
											dp.y = eta_t * Xi.y / dt;
											if (ep.x < 0.0f) { ep.y = dp.y = 0.0f; }
											if (abs(ep.y) > mu * ep.x) { ep.y = mu * sign(ep.x, ep.y);		dp.y = 0.0f; }
											d_ep.y[couple] = ep.y;
											Fcol.y = ep.y + dp.y;
											//////////////////////ローカルy///////////////////////////////

											//////////////////////ローカルz///////////////////////////////
											ep.z += kt * Xi.z;
											dp.z = eta_t * Xi.z / dt;
											if (ep.x < 0.0f) { ep.z = dp.z = 0.0f; }
											if (abs(ep.z) > mu * ep.x) { ep.z = mu * sign(ep.x, ep.z);		dp.z = 0.0f; }
											d_ep.z[couple] = ep.z;
											Fcol.z = ep.z + dp.z;
											//////////////////////ローカルz///////////////////////////////


											//////////////////////ワールド///////////////////////////////
											d_Ftotal.x[i] -= Fcol.z;
											d_Ftotal.y[i] -= Fcol.y;
											d_Ftotal.z[i] -= -Fcol.x;

											d_Torque.x[i] -= Fcol.y * Di * 0.5f;
											d_Torque.y[i] -= -Fcol.z * Di * 0.5f;
											d_Torque.z[i] -= -Tr;
											//////////////////////ワールド///////////////////////////////
										}
										else {
											r_delta_i.x = Omegai.z * dt;
											r_delta_i.y = Omegai.y * dt;
											r_delta_i.z = -Omegai.x * dt;
											r_delta_j.x = Omegaj.z * dt;
											r_delta_j.y = Omegaj.y * dt;
											r_delta_j.z = -Omegaj.x * dt;//角変位

											Xi.x = (Veli.z - Velj.z) * dt;
											Xi.y = (Veli.y - Velj.y) * dt + (r_delta_i.z * Di + r_delta_j.z * Dj) * 0.5f;
											Xi.z = -(Veli.x - Velj.x) * dt - (r_delta_i.y * Di + r_delta_j.y * Dj) * 0.5f;

											//////////////////////ローカルx///////////////////////////////
											ep.x += kn * Xi.x;
											dp.x = eta_n * Xi.x / dt;
											if (ep.x < 0.0f) { ep.x = dp.x = 0.0f; }
											d_ep.x[couple] = ep.x;//バネ更新
											Fcol.x = ep.x + dp.x;
											//////////////////////ローカルx///////////////////////////////

											//////////////////////ローカルy///////////////////////////////
											ep.y += kt * Xi.y;
											dp.y = eta_t * Xi.y / dt;
											if (ep.x < 0.0f) { ep.y = dp.y = 0.0f; }
											if (abs(ep.y) > mu * ep.x) { ep.y = mu * sign(ep.x, ep.y);		dp.y = 0.0f; }
											d_ep.y[couple] = ep.y;
											Fcol.y = ep.y + dp.y;
											//////////////////////ローカルy///////////////////////////////

											//////////////////////ローカルz///////////////////////////////
											ep.z += kt * Xi.z;
											dp.z = eta_t * Xi.z / dt;
											if (ep.x < 0.0f) { ep.z = dp.z = 0.0f; }
											if (abs(ep.z) > mu * ep.x) { ep.z = mu * sign(ep.x, ep.z);		dp.z = 0.0f; }
											d_ep.z[couple] = ep.z;
											Fcol.z = ep.z + dp.z;
											//////////////////////ローカルz///////////////////////////////


											//////////////////////ワールド///////////////////////////////
											d_Ftotal.x[i] -= -Fcol.z;
											d_Ftotal.y[i] -= Fcol.y;
											d_Ftotal.z[i] -= Fcol.x;

											d_Torque.x[i] -= -Fcol.y * Di * 0.5f;
											d_Torque.y[i] -= -Fcol.z * Di * 0.5f;
											d_Torque.z[i] -= Tr;
											//////////////////////ワールド///////////////////////////////
										}
									}
									else {//通常
										if (i < j) {
											r_delta_i.x = (n.x * Omegai.x + n.y * Omegai.y + n.z * Omegai.z) * dt;
											r_delta_i.y = (-n.y * Omegai.x / L_2 + n.x * Omegai.y / L_2) * dt;
											r_delta_i.z = (-n.x * n.z * Omegai.x / L_2 - n.y * n.z * Omegai.y / L_2 + L_2 * Omegai.z) * dt;
											r_delta_j.x = (n.x * Omegaj.x + n.y * Omegaj.y + n.z * Omegaj.z) * dt;
											r_delta_j.y = (-n.y * Omegaj.x / L_2 + n.x * Omegaj.y / L_2) * dt;
											r_delta_j.z = (-n.x * n.z * Omegaj.x / L_2 - n.y * n.z * Omegaj.y / L_2 + L_2 * Omegaj.z) * dt;//角変位

											Xi.x = (n.x * (Veli.x - Velj.x) + n.y * (Veli.y - Velj.y) + n.z * (Veli.z - Velj.z)) * dt;
											Xi.y = (-n.y * (Veli.x - Velj.x) / L_2 + n.x * (Veli.y - Velj.y) / L_2) * dt + (r_delta_i.z * Di + r_delta_j.z * Dj) * 0.5f;
											Xi.z = (-n.x * n.z * (Veli.x - Velj.x) / L_2 - n.y * n.z * (Veli.y - Velj.y) / L_2 + (Veli.z - Velj.z) * L_2) * dt - (r_delta_i.y * Di + r_delta_j.y * Dj) * 0.5f;

											//////////////////////ローカルx///////////////////////////////
											ep.x += kn * Xi.x;
											dp.x = eta_n * Xi.x / dt;
											if (ep.x < 0.0f) { ep.x = dp.x = 0.0f; }
											d_ep.x[couple] = ep.x;//バネ更新
											Fcol.x = ep.x + dp.x;
											//////////////////////ローカルx///////////////////////////////

											//////////////////////ローカルy///////////////////////////////
											ep.y += kt * Xi.y;
											dp.y = eta_t * Xi.y / dt;
											if (ep.x < 0.0f) { ep.y = dp.y = 0.0f; }
											if (abs(ep.y) > mu * ep.x) { ep.y = mu * sign(ep.x, ep.y);		dp.y = 0.0f; }
											d_ep.y[couple] = ep.y;
											Fcol.y = ep.y + dp.y;
											//////////////////////ローカルy///////////////////////////////

											//////////////////////ローカルz///////////////////////////////
											ep.z += kt * Xi.z;
											dp.z = eta_t * Xi.z / dt;
											if (ep.x < 0.0f) { ep.z = dp.z = 0.0f; }
											if (abs(ep.z) > mu * ep.x) { ep.z = mu * sign(ep.x, ep.z);		dp.z = 0.0f; }
											d_ep.z[couple] = ep.z;
											Fcol.z = ep.z + dp.z;
											//////////////////////ローカルz///////////////////////////////


											//////////////////////ワールド///////////////////////////////
											d_Ftotal.x[i] -= n.x * Fcol.x - n.y * Fcol.y / L_2 - n.x * n.z * Fcol.z / L_2;
											d_Ftotal.y[i] -= n.y * Fcol.x + n.x * Fcol.y / L_2 - n.y * n.z * Fcol.z / L_2;
											d_Ftotal.z[i] -= n.z * Fcol.x + Fcol.z * L_2;

											d_Torque.x[i] -= n.x * Tr - (-n.y * Fcol.z / L_2 + n.x * n.z * Fcol.y / L_2) * Di * 0.5f;
											d_Torque.y[i] -= n.y * Tr - (n.x * Fcol.z / L_2 + n.y * n.z * Fcol.y / L_2) * Di * 0.5f;
											d_Torque.z[i] -= n.z * Tr + Fcol.y * L_2 * Di * 0.5f;
											//////////////////////ワールド///////////////////////////////
										}
										else {
											r_delta_i.x = (n.x * Omegai.x + n.y * Omegai.y + n.z * Omegai.z) * dt;
											r_delta_i.y = (n.y * Omegai.x / L_2 - n.x * Omegai.y / L_2) * dt;
											r_delta_i.z = (n.x * n.z * Omegai.x / L_2 + n.y * n.z * Omegai.y / L_2 - L_2 * Omegai.z) * dt;
											r_delta_j.x = (n.x * Omegaj.x + n.y * Omegaj.y + n.z * Omegaj.z) * dt;
											r_delta_j.y = (n.y * Omegaj.x / L_2 - n.x * Omegaj.y / L_2) * dt;
											r_delta_j.z = (n.x * n.z * Omegaj.x / L_2 + n.y * n.z * Omegaj.y / L_2 - L_2 * Omegaj.z) * dt;//角変位

											Xi.x = (n.x * (Veli.x - Velj.x) + n.y * (Veli.y - Velj.y) + n.z * (Veli.z - Velj.z)) * dt;
											Xi.y = (n.y * (Veli.x - Velj.x) / L_2 - n.x * (Veli.y - Velj.y) / L_2) * dt + (r_delta_i.z * Di + r_delta_j.z * Dj) * 0.5f;
											Xi.z = (n.x * n.z * (Veli.x - Velj.x) / L_2 + n.y * n.z * (Veli.y - Velj.y) / L_2 - (Veli.z - Velj.z) * L_2) * dt - (r_delta_i.y * Di + r_delta_j.y * Dj) * 0.5f;

											//////////////////////ローカルx///////////////////////////////
											ep.x += kn * Xi.x;
											dp.x = eta_n * Xi.x / dt;
											if (ep.x < 0.0f) { ep.x = dp.x = 0.0f; }
											d_ep.x[couple] = ep.x;//バネ更新
											Fcol.x = ep.x + dp.x;
											//////////////////////ローカルx///////////////////////////////

											//////////////////////ローカルy///////////////////////////////
											ep.y += kt * Xi.y;
											dp.y = eta_t * Xi.y / dt;
											if (ep.x < 0.0f) { ep.y = dp.y = 0.0f; }
											if (abs(ep.y) > mu * ep.x) { ep.y = mu * sign(ep.x, ep.y);		dp.y = 0.0f; }
											d_ep.y[couple] = ep.y;
											Fcol.y = ep.y + dp.y;
											//////////////////////ローカルy///////////////////////////////

											//////////////////////ローカルz///////////////////////////////
											ep.z += kt * Xi.z;
											dp.z = eta_t * Xi.z / dt;
											if (ep.x < 0.0f) { ep.z = dp.z = 0.0f; }
											if (abs(ep.z) > mu * ep.x) { ep.z = mu * sign(ep.x, ep.z);		dp.z = 0.0f; }
											d_ep.z[couple] = ep.z;
											Fcol.z = ep.z + dp.z;
											//////////////////////ローカルz///////////////////////////////



											//////////////////////ワールド///////////////////////////////
											d_Ftotal.x[i] -= n.x * Fcol.x + n.y * Fcol.y / L_2 + n.x * n.z * Fcol.z / L_2;
											d_Ftotal.y[i] -= n.y * Fcol.x - n.x * Fcol.y / L_2 + n.y * n.z * Fcol.z / L_2;
											d_Ftotal.z[i] -= n.z * Fcol.x - Fcol.z * L_2;

											d_Torque.x[i] -= n.x * Tr - (n.y * Fcol.z / L_2 - n.x * n.z * Fcol.y / L_2) * Di * 0.5f;
											d_Torque.y[i] -= n.y * Tr + (n.x * Fcol.z / L_2 + n.y * n.z * Fcol.y / L_2) * Di * 0.5f;
											d_Torque.z[i] -= n.z * Tr - Fcol.y * L_2 * Di * 0.5;
											//////////////////////ワールド///////////////////////////////
										}
									}
								}
								else {//接触していない時,epを0,pairを-1にしておく。
									for (int k = 0; k < NCP; k++) {
										int kiNPWLLNCP = k + (i - nPWLL) * NCP;
										if (d_pair[kiNPWLLNCP] == j) {
											d_ep.x[kiNPWLLNCP] = 0.0f;
											d_ep.y[kiNPWLLNCP] = 0.0f;
											d_ep.z[kiNPWLLNCP] = 0.0f;
											d_ep_r[kiNPWLLNCP] = 0.0f;
											d_pair[kiNPWLLNCP] = -2;
											break;
										}
									}
								}
							}
							j = d_nxt[j];
							if (j == -1) break;
						}//粒子iの近傍粒子jのループ終了
					}
				}
			}
		}

	}
}

void DEM::ColForce() {
	//printf("ColForce\n");
	////////////////cudaスレッド設定/////////////////////
	dim3 threads(THREADS, 1, 1);
	int TOTAL_THREADS = (nP);	int BLOCKS = TOTAL_THREADS / THREADS + 1;
	dim3 blocks_nP(BLOCKS, 1, 1);
	//////////////////////////////////////////////
	d_ColForce << <blocks_nP, threads >> > (nP, nPSLD, nPWLL, d_D, d_Typ, d_Pos, d_Vel, d_Ftotal, d_Omega, d_Torque, d_ep, d_ep_r, d_pair, m, eta_n, eta_t, eta_r, kn, kt, kr, mu, mu_r, dt, MINc, DBinv, nBx, nBxy, d_bfst, d_blst, d_nxt);
	cudaDeviceSynchronize();
}

__global__ void d_update(const int nP, char* d_Typ, areal3 d_Pos, areal3 d_Vel, areal3 d_Ftotal, areal3 d_Omega, areal3 d_Torque, const real umax,
	const real m, const real dt, const treal3 MINc, const treal3 MAXc, const treal3 G, const real PCL_DST, const real I, const real d_w, real* d_wallangle, real* d_rot_rad) {
	//printf("update\n");
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nP) {
		if (d_Typ[i] == SLD) {//粉体速度更新

			/*treal3 Vtmp;
			Vtmp.x = d_Vel.x[i];		Vtmp.y = d_Vel.y[i];		Vtmp.z = d_Vel.z[i];*/

			d_Vel.x[i] += d_Ftotal.x[i] * dt / m;
			d_Vel.y[i] += d_Ftotal.y[i] * dt / m + G.y * dt;
			d_Vel.z[i] += d_Ftotal.z[i] * dt / m;

			treal3 Utmp;
			Utmp.x = d_Vel.x[i];	Utmp.y = d_Vel.y[i];	Utmp.z = d_Vel.z[i];
			real U = Utmp.x * Utmp.x + Utmp.y * Utmp.y + Utmp.z * Utmp.z;
			U = sqrt(U);
			if (U > umax) {
				Utmp.x *= umax / U;	Utmp.y *= umax / U;	Utmp.z *= umax / U;
				d_Vel.x[i] = Utmp.x;		d_Vel.y[i] = Utmp.y;		d_Vel.z[i] = Utmp.z;
			}

			/*d_Pos.x[i] += 0.5f * (Vtmp.x + d_Vel.x[i]) * dt;//前ステップと現在の速度の平均値分だけ移動させる
			d_Pos.y[i] += 0.5f * (Vtmp.y + d_Vel.y[i]) * dt;
			d_Pos.z[i] += 0.5f * (Vtmp.z + d_Vel.z[i]) * dt;*/

			d_Pos.x[i] += d_Vel.x[i] * dt;
			d_Pos.y[i] += d_Vel.y[i] * dt;
			d_Pos.z[i] += d_Vel.z[i] * dt;

			if (d_Pos.x[i] >= MAXc.x - 5.0 * PCL_DST) { d_Typ[i] = GST; }
			else if (d_Pos.y[i] >= MAXc.y - 5.0 * PCL_DST) { d_Typ[i] = GST; }
			else if (d_Pos.z[i] >= MAXc.z - 5.0 * PCL_DST) { d_Typ[i] = GST; }
			else if (d_Pos.x[i] <= MINc.x + 5.0 * PCL_DST) { d_Typ[i] = GST; }
			else if (d_Pos.y[i] <= MINc.y + 5.0 * PCL_DST) { d_Typ[i] = GST; }
			else if (d_Pos.z[i] <= MINc.z + 5.0 * PCL_DST) { d_Typ[i] = GST; }

			d_Omega.x[i] += d_Torque.x[i] * dt / I;
			d_Omega.y[i] += d_Torque.y[i] * dt / I;
			d_Omega.z[i] += d_Torque.z[i] * dt / I;

			d_Ftotal.x[i] = 0.0f;
			d_Ftotal.y[i] = 0.0f;
			d_Ftotal.z[i] = 0.0f;

			d_Torque.x[i] = 0.0f;
			d_Torque.y[i] = 0.0f;
			d_Torque.z[i] = 0.0f;
		}
		/*else if (d_Typ[i] == OBJ) {//壁速度更新
			d_Vel.x[i] = -d_w * d_rot_rad[i] * sin(d_wallangle[i]);
			//d_Vel.y[i] = 0.0f;
			d_Vel.z[i] = d_w * d_rot_rad[i] * cos(d_wallangle[i]);
			

			d_Pos.x[i] += d_Vel.x[i] * dt;
			//d_Pos.y[i] += d_Vel.y[i] * dt;
			d_Pos.z[i] += d_Vel.z[i] * dt;

			d_wallangle[i] += d_w * dt;
			if (d_wallangle[i] > 2.0f * pi) { d_wallangle[i] -= 2.0f * pi; }
			else if (d_wallangle[i] < -2.0f * pi) { d_wallangle[i] += 2.0f * pi; }//if(<pi)になってた
		}
		*/

	}
}

void DEM::update() {
	//printf("update\n");
	////////////////cudaスレッド設定/////////////////////
	dim3 threads(THREADS, 1, 1);
	int TOTAL_THREADS = nBxyz;	int BLOCKS = TOTAL_THREADS / THREADS + 1;
	dim3 blocks_nBxyz(BLOCKS, 1, 1);
	TOTAL_THREADS = (nP);	BLOCKS = TOTAL_THREADS / THREADS + 1;
	dim3 blocks_nP(BLOCKS, 1, 1);
	//////////////////////////////////////////////
	d_update << <blocks_nP, threads >> > (nP, d_Typ, d_Pos, d_Vel, d_Ftotal, d_Omega, d_Torque, umax, m, dt, MINc, MAXc, G, PCL_DST, I, w, d_wallangle, d_rot_rad);
	cudaDeviceSynchronize();
}

void DEM::DevicetoHost() {

	cudaMemcpy(Typ, d_Typ, sizeof(char) * nP, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pos.x, d_Pos.x, sizeof(real) * nP, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pos.y, d_Pos.y, sizeof(real) * nP, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pos.z, d_Pos.z, sizeof(real) * nP, cudaMemcpyDeviceToHost);
	cudaMemcpy(Vel.x, d_Vel.x, sizeof(real) * nP, cudaMemcpyDeviceToHost);
	cudaMemcpy(Vel.y, d_Vel.y, sizeof(real) * nP, cudaMemcpyDeviceToHost);
	cudaMemcpy(Vel.z, d_Vel.z, sizeof(real) * nP, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf_s("%f:\n ", TIM);
	/*cudaMemcpy(pair, d_pair, sizeof(int) * (NCP * nPSLD), cudaMemcpyDeviceToHost);
	for (int k = 0; k < NCP; k++) { printf_s("%d ", pair[k + (220000 - nPWLL) * NCP]); }
	printf_s("\n");  //pair確認用
	*/
}

void DEM::ClcDEM() {
	//real outtime = 0.0;

	WrtDatWLL();
	WrtDat();
	
	iF++;

	while (1) {

		MkBkt();

		ColForce();

		update();

		if (outtime >= output_time) {
			outtime -= output_time;
			DevicetoHost();
			WrtDat();
			iF++;
		}

		if (TIM >= FIN_TIM) { WrtDat2(); break; }

		outtime += dt;
		TIM += dt;
		//printf("TIM = %f\n", TIM);
	}
}

void DEM::memory_free() {
	free(Pos.x); free(Pos.y); free(Pos.z);
	free(Vel.x); free(Vel.y); free(Vel.z);
	free(Omega.x); free(Omega.y); free(Omega.z);
	free(Ftotal.x); free(Ftotal.y); free(Ftotal.z);
	free(Torque.x); free(Torque.y); free(Torque.z);
	free(Typ);
	free(ep.x); free(ep.y); free(ep.z);
	free(ep_r);
	free(D);
	free(pair);
	free(wallangle);
	free(rot_rad);

	cudaFree(d_Pos.x); cudaFree(d_Pos.y); cudaFree(d_Pos.z);
	cudaFree(d_Vel.x); cudaFree(d_Vel.y); cudaFree(d_Vel.z);
	cudaFree(d_Omega.x); cudaFree(d_Omega.y); cudaFree(d_Omega.z);
	cudaFree(d_Ftotal.x); cudaFree(d_Ftotal.y); cudaFree(d_Ftotal.z);
	cudaFree(d_Torque.x); cudaFree(d_Torque.y); cudaFree(d_Torque.z);
	cudaFree(d_Typ);
	cudaFree(d_ep.x); cudaFree(d_ep.y); cudaFree(d_ep.z);
	cudaFree(d_ep_r);
	cudaFree(d_D);
	cudaFree(d_pair);
	cudaFree(d_wallangle);
	cudaFree(d_rot_rad);
}
