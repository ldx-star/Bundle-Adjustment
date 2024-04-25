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

const int num_cam_params = 9;


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

void my_jacobian(const Camera &cam, const Point3D &p3d, double *cam_x_ptr, double *cam_y_ptr, double *point_x_ptr, double *point_y_ptr) {
    const double f = cam.focal_length;
    const double *R = cam.rotation;
    const double *t = cam.translation;
    const double *X = p3d.pos;
    const double k0 = cam.distortion[0];
    const double k1 = cam.distortion[1];

    const double xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const double yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const double zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];

    const double x = xc / zc;
    const double y = yc / zc;

    const double r2 = x * x + y * y;
    const double distort = 1.0 + (k0 + k1 * r2) * r2;

    const double u = f * distort * x;
    const double v = f * distort * y;

    /* 关于焦距的偏导*/
    cam_x_ptr[0] = distort * x;
    cam_y_ptr[0] = distort * y;

    /* 关于畸变系数k0,k1的偏导*/
    // 计算中间变量
    const double u_deriv_distort = f * x;
    const double v_deriv_distort = f * y;
    const double distort_deriv_k0 = r2;
    const double distort_deriv_k1 = r2 * r2;

    cam_x_ptr[1] = u_deriv_distort * distort_deriv_k0;
    cam_x_ptr[2] = u_deriv_distort * distort_deriv_k1;
    cam_y_ptr[1] = v_deriv_distort * distort_deriv_k0;
    cam_y_ptr[2] = v_deriv_distort * distort_deriv_k1;

    // 计算中间变量x,y关于xc,yc,zc的偏导
    const double x_deriv_xc = 1 / zc;
    const double x_deriv_yc = 0;
    const double x_deriv_zc = -x / zc;
    const double y_deriv_xc = 0;
    const double y_deriv_yc = 1 / zc;
    const double y_deriv_zc = -y / zc;

    // 计算中间变量u,v关于x,y的偏导
    const double u_deriv_x = f * distort;
    const double v_deriv_y = f * distort;

    // 计算中间变量distort关于r2的偏导
    const double distort_deriv_r2 = k0 + 2 * k1 * r2;

    // 计算中间变量r2关于xc,yc,zc的偏导
    const double r2_deriv_xc = 2 * x / zc;
    const double r2_deriv_yc = 2 * y / zc;
    const double r2_deriv_zc = -2 * r2 / zc;

    // 计算中间变量distort关于xc,yc,zc的偏导
    const double distort_deriv_xc = distort_deriv_r2 * r2_deriv_xc;
    const double distort_deriv_yc = distort_deriv_r2 * r2_deriv_yc;
    const double distort_deriv_zc = distort_deriv_r2 * r2_deriv_zc;

    // 计算中间变量u,v关于xc,yc,zc的偏导
    const double u_deriv_xc = u_deriv_distort * distort_deriv_xc + u_deriv_x * x_deriv_xc;
    const double u_deriv_yc = u_deriv_distort * distort_deriv_yc + u_deriv_x * x_deriv_yc;
    const double u_deriv_zc = u_deriv_distort * distort_deriv_zc + u_deriv_x * x_deriv_zc;

    const double v_deriv_xc = v_deriv_distort * distort_deriv_xc + v_deriv_y * y_deriv_xc;
    const double v_deriv_yc = v_deriv_distort * distort_deriv_yc + v_deriv_y * y_deriv_yc;
    const double v_deriv_zc = v_deriv_distort * distort_deriv_zc + v_deriv_y * y_deriv_zc;

    /*关于平移向量t0,t1,t2的偏导*/
    const double xc_deriv_t0 = 1;
    const double yc_deriv_t1 = 1;
    const double zc_deriv_t2 = 1;

    cam_x_ptr[3] = u_deriv_xc * xc_deriv_t0;
    cam_x_ptr[4] = u_deriv_yc * yc_deriv_t1;
    cam_x_ptr[5] = u_deriv_zc * zc_deriv_t2;

    cam_y_ptr[3] = v_deriv_xc * xc_deriv_t0;
    cam_y_ptr[4] = v_deriv_yc * yc_deriv_t1;
    cam_y_ptr[5] = v_deriv_zc * zc_deriv_t2;

    /*计算关于旋转矩阵（表示为角轴向量w0,w1,w2）的偏导*/
    const double rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2];
    const double ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2];
    const double rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
    const double xc_deriv_w0 = 0;
    const double xc_deriv_w1 = rz;
    const double xc_deriv_w2 = -ry;
    const double yc_deriv_w0 = -rz;
    const double yc_deriv_w1 = 0;
    const double yc_deriv_w2 = rx;
    const double zc_deriv_w0 = ry;
    const double zc_deriv_w1 = -rx;
    const double zc_deriv_w2 = 0;

    cam_x_ptr[6] = u_deriv_yc * yc_deriv_w0 + u_deriv_zc * zc_deriv_w0;
    cam_x_ptr[7] = u_deriv_xc * xc_deriv_w1 + u_deriv_zc * zc_deriv_w1;
    cam_x_ptr[8] = u_deriv_xc * xc_deriv_w2 + u_deriv_yc * yc_deriv_w2;

    cam_x_ptr[6] = v_deriv_yc * yc_deriv_w0 + v_deriv_zc * zc_deriv_w0;
    cam_x_ptr[7] = v_deriv_xc * xc_deriv_w1 + v_deriv_zc * zc_deriv_w1;
    cam_x_ptr[8] = v_deriv_xc * xc_deriv_w2 + v_deriv_yc * yc_deriv_w2;

    /*计算关于三维点坐标X,Y,Z的偏导*/
    const double xc_deriv_X = R[0];
    const double xc_deriv_Y = R[1];
    const double xc_deriv_Z = R[2];
    const double yc_deriv_X = R[3];
    const double yc_deriv_Y = R[4];
    const double yc_deriv_Z = R[5];
    const double zc_deriv_X = R[6];
    const double zc_deriv_Y = R[7];
    const double zc_deriv_Z = R[8];

    point_x_ptr[0] = u_deriv_xc * xc_deriv_X + u_deriv_yc * yc_deriv_X + u_deriv_zc * zc_deriv_X;
    point_x_ptr[1] = u_deriv_xc * xc_deriv_Y + u_deriv_yc * yc_deriv_Y + u_deriv_zc * zc_deriv_Y;
    point_x_ptr[2] = u_deriv_xc * xc_deriv_Z + u_deriv_yc * yc_deriv_Z + u_deriv_zc * zc_deriv_Z;

    point_y_ptr[0] = v_deriv_xc * xc_deriv_X + v_deriv_yc * yc_deriv_X + v_deriv_zc * zc_deriv_X;
    point_y_ptr[1] = v_deriv_xc * xc_deriv_Y + v_deriv_yc * yc_deriv_Y + v_deriv_zc * zc_deriv_Y;
    point_y_ptr[2] = v_deriv_xc * xc_deriv_Z + v_deriv_yc * yc_deriv_Z + v_deriv_zc * zc_deriv_Z;
}

/**
 * \description 构造雅阁比矩阵
 *      相机参数的雅阁比矩阵大小为： (2*observations.size() * (num_cameras*9)
 *      三维点参数的雅阁比矩阵大小为： (2*observations.size() * (num_points*3)
 * @param jac_cam 关于相机参数的雅阁比矩阵
 * @param jac_points 关于三维点的雅阁比矩阵
 */
void analytic_jacobian(Eigen::MatrixXd &jac_cam, Eigen::MatrixXd &jac_points) {
    const int camera_cols = cameras.size() * 9;
    const int point_cols = points_3d.size() * 3;
    const int jacobi_rows = observations.size() * 2;

    jac_cam.resize(jacobi_rows,camera_cols);
    jac_points.resize(jacobi_rows,point_cols);

    double cam_x_ptr[9], cam_y_ptr[9], point_x_ptr[3], point_y_ptr[3];
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

        for(int j = 0;j < num_cam_params;j++){
            jac_cam(row_x,cam_col+j) = cam_x_ptr[j];
            jac_cam(row_y,cam_col+j) = cam_y_ptr[j];
        }
        for(int j = 0; j < 3; j++){
            jac_points(row_x,point_col+j) = point_x_ptr[j];
            jac_points(row_y,point_col+j) = point_y_ptr[j];
        }
    }
}

void lm_optimization() {
    std::vector<double> F; // 重投影误差
    compute_reprojection_errors(F);
    // 计算初始均方误差
    double current_mse = compute_mse(F);
    final_mse = initial_mse = current_mse;

    //设置共轭梯度法的相关参数
    trust_region_radius = TRUST_REGION_RADIUS_INIT;

    /* Levenberg-Marquard 算法 */
    for (int lm_iter = 0;; lm_iter++) {
        // 当均方误差小于一定阈值时停止
        if (current_mse < lm_mse_threshold) {
            std::cout << "BA: Satisfied MSE threshold" << std::endl;
        }
        // 1、计算雅阁比矩阵
        Eigen::MatrixXd Jc,Jp;
        analytic_jacobian(Jc,Jp);
        std::cout << Jc(0,0 )<< std::endl;

        // 2、共轭梯度法进行求解


    }

}

int main() {
    load_data("../test_ba.txt");
    lm_optimization();
    return 0;
}