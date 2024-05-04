//
// Created by ldx on 24-2-22.
//
#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<cassert>
#include<Eigen/Core>
#include<Eigen/Dense>

struct Camera {
    double focal_length = 0.0;// 焦距
    double distortion[2];//畸变系数
    double translation[3];//平移
    double rotation[9];//旋转
    bool is_constant = false;
};
struct Point3D {
    double pos[3];
    bool is_constant = false;
};
struct Observation {
    double pos[2];
    int camera_id;
    int point_id;
};

//全局变量
std::vector<Camera> cameras;
std::vector<Point3D> points_3d;
std::vector<Observation> observations; // 两个视图的同名点

#define TRUST_REGION_RADIUS_INIT (1000)
#define TRUST_REGION_RADIUS_DECREMENT (1.0 / 10.0)
#define TRUST_REGION_RADIUS_GAIN (10.0)

const int lm_max_iterations = 100; // lm算法迭代次数
double initial_mse = 0.0;
double final_mse = 0.0;

// lm 算法终止条件
double lm_mse_threshold = 1e-16;
double lm_delta_threshold = 1e-8;

// 信赖域大小
double trust_region_radius = 1000;
int cg_max_iterations = 1000;

const int num_cam_params = 15;


void load_data(const std::string &file) {
    /* 加载数据 */
    std::ifstream in(file);
    assert(in.is_open());
    std::string line, word;

    // 加载相机参数
    {
        int n_cams = 0; //相机编号
        getline(in, line);
        std::stringstream stream(line); //从字符流中读取
        stream >> word >> n_cams;
        cameras.resize(n_cams);
        for (int i = 0; i < cameras.size(); i++) {
            getline(in, line);
            std::stringstream stream(line);
            stream >> cameras[i].focal_length;
            stream >> cameras[i].distortion[0] >> cameras[i].distortion[1];
            for (int j = 0; j < 3; j++) stream >> cameras[i].translation[j];
            for (int j = 0; j < 9; j++) stream >> cameras[i].rotation[j];
        }
    }

    //加载三维点
    {
        int n_points = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_points;
        points_3d.resize(n_points);
        for (int i = 0; i < n_points; i++) {
            getline(in, line);
            std::stringstream stream(line);
            stream >> points_3d[i].pos[0] >> points_3d[i].pos[1] >> points_3d[i].pos[2];
        }
    }

    // 加载观察点（两个视图的同名点）
    {
        int n_observations = 0;
        getline(in, line);
        std::stringstream stream(line);
        stream >> word >> n_observations;
        observations.resize(n_observations);
        for (int i = 0; i < observations.size(); i++) {
            getline(in, line);
            std::stringstream stream(line);
            stream >> observations[i].camera_id >> observations[i].point_id >> observations[i].pos[0] >> observations[i].pos[1];
        }
    }
}

/**
 * \desciption 径向畸变矫正
 * @param x
 * @param y
 * @param dist
 */
void radial_distort(double &x, double &y, const double *dist) {
    double const radius = x * x + y * y;
    double const factor = 1.0 + radius * (dist[0] + dist[1] * radius);
    x *= factor;
    y *= factor;
}

/**
 * \description 计算重投影误差(将三维点映射回二维点计算误差)
 *
 */
void compute_reprojection_errors(std::vector<double> &vector_f) {
    for (int i = 0; i < observations.size(); i++) {
        Observation const &obs = observations[i];
        Point3D const &p3d = points_3d[obs.point_id];
        Camera const &cam = cameras[obs.camera_id];

        const double *f = &cam.focal_length;
        const double *dist = cam.distortion;
        const double *rot = cam.rotation;
        const double *trans = cam.translation;
        const double *point = p3d.pos;


        double rp[] = {0, 0, 0};
        for (int d = 0; d < 3; ++d) {
            rp[0] += rot[0 + d] * point[d];
            rp[1] += rot[3 + d] * point[d];
            rp[2] += rot[6 + d] * point[d];
        }
        rp[2] = (rp[2] + trans[2]);
        rp[0] = (rp[0] + trans[0]) / rp[2];
        rp[1] = (rp[1] + trans[1]) / rp[2];

        radial_distort(rp[0], rp[1], dist);
        vector_f.push_back(rp[0] * (*f) - obs.pos[0]);
        vector_f.push_back(rp[1] * (*f) - obs.pos[1]);
    }
}

/**
 * \description 计算均方误差
 * @param F
 * @return 误差总和
 */

double compute_mse(const std::vector<double> &F) {
    double mse = 0.0;
    for (int i = 0; i < F.size(); i++) {
        mse += F[i] * F[i];
    }
    return mse / (F.size() / 2.0);
}

/**
 * \求雅可比矩阵
 * @param cam 相机参数
 * @param p3d 三维点
 * @param cam_x_ptr [f,k0,k1,t0,t1,t2,r0,r1,r2,r3,r4,r5,r6,r7,r8]
 * @param cam_y_ptr [X,Y,Z]
 * @param point_x_ptr
 * @param point_y_ptr
 */
void my_jacobian(const Camera &cam, const Point3D &p3d, double *cam_x_ptr, double *cam_y_ptr, double *point_x_ptr, double *point_y_ptr) {
    const double f = cam.focal_length;
    const double *R = cam.rotation;
    const double *t = cam.translation;
    const double *pos = p3d.pos;
    const double X = pos[0];
    const double Y = pos[1];
    const double Z = pos[2];
    const double k0 = cam.distortion[0];
    const double k1 = cam.distortion[1];
    const double r0 = R[0];
    const double r1 = R[1];
    const double r2 = R[2];
    const double r3 = R[3];
    const double r4 = R[4];
    const double r5 = R[5];
    const double r6 = R[6];
    const double r7 = R[7];
    const double r8 = R[8];

    const double xc = r0 * X + r1 * Y + r2 * Z + t[0];
    const double yc = r3 * X + r4 * Y + r5 * Z + t[1];
    const double zc = r6 * X + r7 * Y + r8 * Z + t[2];

    const double x = xc / zc;
    const double y = yc / zc;

    const double rr = x * x + y * y;
    const double d = 1.0 + (k0 + k1 * rr) * rr;

    const double u = f * d * x;
    const double v = f * d * y;

    /* 关于焦距的偏导*/
    cam_x_ptr[0] = d * x;
    cam_y_ptr[0] = d * y;

    /* 关于畸变系数k0,k1的偏导*/
    const double u_deriv_d = f * x;
    const double v_deriv_d = f * y;
    const double d_deriv_k0 = rr;
    const double d_deriv_k1 = rr * rr;

    cam_x_ptr[1] = u_deriv_d * d_deriv_k0;
    cam_x_ptr[2] = u_deriv_d * d_deriv_k1;
    cam_y_ptr[1] = v_deriv_d * d_deriv_k0;
    cam_y_ptr[2] = v_deriv_d * d_deriv_k1;

    /* 关于平移向量t0,t1,t2的偏导*/
    const double xc_deriv_t0 = 1;
    const double yc_deriv_t1 = 1;
    const double zc_deriv_t2 = 1;

    const double x_deriv_xc = 1 / zc;
    const double y_deriv_yc = 1 / zc;
    const double x_deriv_zc = -xc / (zc * zc);
    const double y_deriv_zc = -yc / (zc * zc);

    const double x_deriv_t0 = x_deriv_xc * xc_deriv_t0;
    const double y_deriv_t1 = y_deriv_yc * yc_deriv_t1;
    const double x_deriv_t2 = x_deriv_zc * zc_deriv_t2;
    const double y_deriv_t2 = y_deriv_zc * zc_deriv_t2;

    const double rr_deriv_x = 2 * x;
    const double rr_deriv_y = 2 * y;
    const double rr_deriv_t0 = rr_deriv_x * x_deriv_t0;
    const double rr_deriv_t1 = rr_deriv_y * y_deriv_t1;
    const double rr_deriv_t2 = rr_deriv_x * x_deriv_t2 + rr_deriv_y * y_deriv_t2;

    const double d_deriv_rr = k0 + 2 * k1 * rr;
    const double d_deriv_t0 = d_deriv_rr * rr_deriv_t0;
    const double d_deriv_t1 = d_deriv_rr * rr_deriv_t1;
    const double d_deriv_t2 = d_deriv_rr * rr_deriv_t2;

    const double d_deriv_x = d_deriv_rr * rr_deriv_x;
    const double d_deriv_y = d_deriv_rr * rr_deriv_y;
    const double u_deriv_x = f * d + f * x * d_deriv_x;
//    const double u_deriv_x = f * d ;
    const double v_deriv_y = f * d + f * y * d_deriv_y;

    cam_x_ptr[3] = u_deriv_d * d_deriv_t0 * x + u_deriv_x * x_deriv_t0 * d;
    cam_x_ptr[4] = u_deriv_d * d_deriv_t1 * x;
    cam_x_ptr[5] = u_deriv_d * d_deriv_t2 * x + u_deriv_x * x_deriv_t2 * d;
    cam_y_ptr[3] = v_deriv_d * d_deriv_t0 * y;
    cam_y_ptr[4] = v_deriv_d * d_deriv_t1 * y + v_deriv_y * y_deriv_t1 * d;
    cam_y_ptr[5] = v_deriv_d * d_deriv_t2 * y + v_deriv_y * y_deriv_t2 * d;

    /* 关于平移向量r0,r1,r2,r3,r4,r5,r6,r7,r8的偏导*/
    const double xc_deriv_r0 = X;
    const double yc_deriv_r3 = X;
    const double zc_deriv_r6 = X;
    const double xc_deriv_r1 = Y;
    const double yc_deriv_r4 = Y;
    const double zc_deriv_r7 = Y;
    const double xc_deriv_r2 = Z;
    const double yc_deriv_r5 = Z;
    const double zc_deriv_r8 = Z;

    const double rr_deriv_r0 = rr_deriv_x * x_deriv_xc * xc_deriv_r0;
    const double rr_deriv_r1 = rr_deriv_x * x_deriv_xc * xc_deriv_r1;
    const double rr_deriv_r2 = rr_deriv_x * x_deriv_xc * xc_deriv_r2;
    const double rr_deriv_r3 = rr_deriv_y * y_deriv_yc * yc_deriv_r3;
    const double rr_deriv_r4 = rr_deriv_y * y_deriv_yc * yc_deriv_r4;
    const double rr_deriv_r5 = rr_deriv_y * y_deriv_yc * yc_deriv_r5;
    const double rr_deriv_r6 = rr_deriv_x * x_deriv_zc * zc_deriv_r6 + rr_deriv_y * y_deriv_zc * zc_deriv_r6;
    const double rr_deriv_r7 = rr_deriv_x * x_deriv_zc * zc_deriv_r7 + rr_deriv_y * y_deriv_zc * zc_deriv_r7;
    const double rr_deriv_r8 = rr_deriv_x * x_deriv_zc * zc_deriv_r8 + rr_deriv_y * y_deriv_zc * zc_deriv_r8;

    const double d_deriv_r0 = d_deriv_rr * rr_deriv_r0;
    const double d_deriv_r1 = d_deriv_rr * rr_deriv_r1;
    const double d_deriv_r2 = d_deriv_rr * rr_deriv_r2;
    const double d_deriv_r3 = d_deriv_rr * rr_deriv_r3;
    const double d_deriv_r4 = d_deriv_rr * rr_deriv_r4;
    const double d_deriv_r5 = d_deriv_rr * rr_deriv_r5;
    const double d_deriv_r6 = d_deriv_rr * rr_deriv_r6;
    const double d_deriv_r7 = d_deriv_rr * rr_deriv_r7;
    const double d_deriv_r8 = d_deriv_rr * rr_deriv_r8;

    const double x_deriv_r0 = x_deriv_xc * xc_deriv_r0;
    const double x_deriv_r1 = x_deriv_xc * xc_deriv_r1;
    const double x_deriv_r2 = x_deriv_xc * xc_deriv_r2;
    const double x_deriv_r6 = x_deriv_zc * zc_deriv_r6;
    const double x_deriv_r7 = x_deriv_zc * zc_deriv_r7;
    const double x_deriv_r8 = x_deriv_zc * zc_deriv_r8;
    const double y_deriv_r3 = y_deriv_yc * yc_deriv_r3;
    const double y_deriv_r4 = y_deriv_yc * yc_deriv_r4;
    const double y_deriv_r5 = y_deriv_yc * yc_deriv_r5;
    const double y_deriv_r6 = y_deriv_zc * zc_deriv_r8;
    const double y_deriv_r7 = y_deriv_zc * zc_deriv_r8;
    const double y_deriv_r8 = y_deriv_zc * zc_deriv_r8;

    cam_x_ptr[6] = f * x * d_deriv_r0 + f * d * x_deriv_r0;
    cam_x_ptr[7] = f * x * d_deriv_r1 + f * d * x_deriv_r1;
    cam_x_ptr[8] = f * x * d_deriv_r2 + f * d * x_deriv_r2;
    cam_x_ptr[9] = f * x * d_deriv_r3;
    cam_x_ptr[10] = f * x * d_deriv_r4;
    cam_x_ptr[11] = f * x * d_deriv_r5;
    cam_x_ptr[12] = f * x * d_deriv_r6 + f * d * x_deriv_r6;
    cam_x_ptr[13] = f * x * d_deriv_r7 + f * d * x_deriv_r7;
    cam_x_ptr[14] = f * x * d_deriv_r8 + f * d * x_deriv_r8;

    cam_y_ptr[6] = f * y * d_deriv_r0;
    cam_y_ptr[7] = f * y * d_deriv_r1;
    cam_y_ptr[8] = f * y * d_deriv_r2;
    cam_y_ptr[9] = f * y * d_deriv_r3 + f * d * y_deriv_r3;
    cam_y_ptr[10] = f * y * d_deriv_r4 + f * d * y_deriv_r4;
    cam_y_ptr[11] = f * y * d_deriv_r5 + f * d * y_deriv_r5;
    cam_y_ptr[12] = f * y * d_deriv_r6 + f * d * y_deriv_r6;
    cam_y_ptr[13] = f * y * d_deriv_r7 + f * d * y_deriv_r7;
    cam_y_ptr[14] = f * y * d_deriv_r8 + f * d * y_deriv_r8;

    /* 关于平移向量X,Y,Z的偏导*/
    const double xc_deriv_X = r0;
    const double xc_deriv_Y = r1;
    const double xc_deriv_Z = r2;
    const double yc_deriv_X = r3;
    const double yc_deriv_Y = r4;
    const double yc_deriv_Z = r5;
    const double zc_deriv_X = r6;
    const double zc_deriv_Y = r7;
    const double zc_deriv_Z = r8;

    const double x_deriv_X = x_deriv_xc * xc_deriv_X + x_deriv_zc * zc_deriv_X;
    const double y_deriv_X = y_deriv_yc * yc_deriv_X + y_deriv_zc * zc_deriv_X;
    const double x_deriv_Y = x_deriv_xc * xc_deriv_Y + x_deriv_zc * zc_deriv_Y;
    const double y_deriv_Y = y_deriv_yc * yc_deriv_Y + y_deriv_zc * zc_deriv_Y;
    const double x_deriv_Z = x_deriv_xc * xc_deriv_Z + x_deriv_zc * zc_deriv_Z;
    const double y_deriv_Z = y_deriv_yc * yc_deriv_Z + y_deriv_zc * zc_deriv_Z;

    const double rr_deriv_X = rr_deriv_x * x_deriv_X + rr_deriv_y * y_deriv_X;
    const double rr_deriv_Y = rr_deriv_x * x_deriv_Y + rr_deriv_y * y_deriv_Y;
    const double rr_deriv_Z = rr_deriv_x * x_deriv_Z + rr_deriv_y * y_deriv_Z;

    const double d_deriv_X = d_deriv_rr * rr_deriv_X;
    const double d_deriv_Y = d_deriv_rr * rr_deriv_Y;
    const double d_deriv_Z = d_deriv_rr * rr_deriv_Z;

    point_x_ptr[0] = f * x * d_deriv_X + f * d * x_deriv_X;
    point_x_ptr[1] = f * x * d_deriv_Y + f * d * x_deriv_Y;
    point_x_ptr[2] = f * x * d_deriv_Z + f * d * x_deriv_Z;
    point_y_ptr[0] = f * y * d_deriv_X + f * d * y_deriv_X;
    point_y_ptr[1] = f * y * d_deriv_Y + f * d * y_deriv_Y;
    point_y_ptr[2] = f * y * d_deriv_Z + f * d * y_deriv_Z;
}

/**
 * \description 构造雅阁比矩阵
 *      相机参数的雅阁比矩阵大小为： (2*observations.size() * (num_cameras*9)
 *      三维点参数的雅阁比矩阵大小为： (2*observations.size() * (num_points*3)
 * @param jac_cam 关于相机参数的雅阁比矩阵
 * @param jac_points 关于三维点的雅阁比矩阵
 */
void analytic_jacobian(Eigen::MatrixXd &jac_cam, Eigen::MatrixXd &jac_points) {
    const int camera_cols = cameras.size() * num_cam_params;
    const int point_cols = points_3d.size() * 3;
    const int jacobi_rows = observations.size() * 2;

    jac_cam.resize(jacobi_rows, camera_cols);
    jac_points.resize(jacobi_rows, point_cols);

    double cam_x_ptr[15], cam_y_ptr[15], point_x_ptr[3], point_y_ptr[3];
    // 遍历每一个观察到的二维点
    for (int i = 0; i < observations.size(); i++) {
        const Observation &obs = observations[i];
        //三维点坐标
        const Point3D &p3p = points_3d[obs.point_id];
        //相机参数
        const Camera &cam = cameras[obs.camera_id];

        /*对相机和三维点求偏导*/
        my_jacobian(cam, p3p, cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr);

        /*观察点对应雅阁比矩阵的行，第i个观察点对应的位置是2*i,2*i+1*/
        int row_x = 2 * i + 0;
        int row_y = 2 * i + 1;

        //jac_cam中相机对应的列数为camera_id * n_cam_params
        int cam_col = obs.camera_id * num_cam_params;
        //jac_points中三维点对应的列数为point_id * 3
        int point_col = obs.camera_id * 3;

        for (int j = 0; j < num_cam_params; j++) {
            jac_cam(row_x, cam_col + j) = cam_x_ptr[j];
            jac_cam(row_y, cam_col + j) = cam_y_ptr[j];
        }
        for (int j = 0; j < 3; j++) {
            jac_points(row_x, point_col + j) = point_x_ptr[j];
            jac_points(row_y, point_col + j) = point_y_ptr[j];
        }
    }
}

/**
 *\descrition 使用高斯牛顿法进行求解
 * @param Jc 相机参数雅可比矩阵
 * @param Jp 三维点雅可比矩阵
 * @param F y‘ - y
 */
Eigen::MatrixXd my_solve_schur(const Eigen::MatrixXd Jc, const Eigen::MatrixXd Jp, const std::vector<double> &F) {
    /**
     *  雅可比矩阵：
     *      J = [Jc Jp]
     *
     */
    Eigen::MatrixXd J(Jc.rows(), Jc.cols() + Jp.cols());
    Eigen::VectorXd err = Eigen::VectorXd::Zero(F.size());

    for (int i = 0; i < F.size(); i++) {
        err(i) = F[i];
    }
    J << Jc, Jp;
    const Eigen::MatrixXd JT = J.transpose();
    const Eigen::MatrixXd JTJ = JT * J;
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(JTJ.rows(), JTJ.rows());
    identity *= 0.1;
    /* B = JTJ + uI */
    const Eigen::MatrixXd B = JTJ + identity;
    const Eigen::MatrixXd inv_B = B.inverse();
    const Eigen::MatrixXd E = -JT * err;
    const Eigen::VectorXd delta = inv_B * E;
    int camId = 0;
    while (camId < cameras.size()) {
        // [f,k0,k1,t0,t1,t2,r0,r1,r2,r3,r4,r5,r6,r7,r8]
        cameras[camId].focal_length += delta(camId * num_cam_params + 0);
        cameras[camId].distortion[0] += delta(camId * num_cam_params + 1);
        cameras[camId].distortion[1] += delta(camId * num_cam_params + 2);
        cameras[camId].translation[0] += delta(camId * num_cam_params + 3);
        cameras[camId].translation[1] += delta(camId * num_cam_params + 4);
        cameras[camId].translation[2] += delta(camId * num_cam_params + 5);
        cameras[camId].rotation[0] += delta(camId * num_cam_params + 6);
        cameras[camId].rotation[1] += delta(camId * num_cam_params + 7);
        cameras[camId].rotation[2] += delta(camId * num_cam_params + 8);
        cameras[camId].rotation[3] += delta(camId * num_cam_params + 9);
        cameras[camId].rotation[4] += delta(camId * num_cam_params + 10);
        cameras[camId].rotation[5] += delta(camId * num_cam_params + 11);
        cameras[camId].rotation[6] += delta(camId * num_cam_params + 12);
        cameras[camId].rotation[7] += delta(camId * num_cam_params + 13);
        cameras[camId].rotation[8] += delta(camId * num_cam_params + 14);
        camId++;
    }
    int pointId = 0;
    while (pointId < points_3d.size()) {
        points_3d[pointId].pos[0] += delta(cameras.size() * num_cam_params + pointId * 3 + 0);
        points_3d[pointId].pos[1] += delta(cameras.size() * num_cam_params + pointId * 3 + 1);
        points_3d[pointId].pos[2] += delta(cameras.size() * num_cam_params + pointId * 3 + 2);
        pointId++;
    }
}


void lm_optimization() {
    std::vector<double> F; // 重投影误差
    compute_reprojection_errors(F);
    // 计算初始均方误差
    double current_mse = compute_mse(F);
    final_mse = initial_mse = current_mse;

    double new_mse, delta_mse, delta_mse_ratio = 1.0;
    /* Levenberg-Marquard 算法 */
    for (int lm_iter = 0; lm_iter < lm_max_iterations; lm_iter++) {
        // 当均方误差小于一定阈值时停止
        if (current_mse < lm_mse_threshold) {
            std::cout << "BA: Satisfied MSE threshold" << std::endl;
        }
        // 1、计算雅阁比矩阵
        Eigen::MatrixXd Jc, Jp;
        analytic_jacobian(Jc, Jp);

        // 2、高斯牛顿法进行求解
        Eigen::MatrixXd delta = my_solve_schur(Jc, Jp, F);
        // 计算重投影误差
        std::vector<double> F_new;
        compute_reprojection_errors(F_new);
        new_mse = compute_mse(F_new);
        delta_mse = current_mse - new_mse;
//
//        bool successful_iteration = delta_mse > 0.0;
//        if (successful_iteration) {
//            std::cout << "\titer: " << lm_iter << "\tpre_mse: " << current_mse << "\tcur_mse: " << new_mse << std::endl;
//            current_mse = new_mse;
//        }
        std::cout << "\titer: " << lm_iter << "\tpre_mse: " << current_mse << "\tcur_mse: " << new_mse << std::endl;
        current_mse = new_mse;
        F = F_new;
    }

}

int main() {
    load_data("../test_ba.txt");
    lm_optimization();
    return 0;
}