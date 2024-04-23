//
// Created by ldx on 24-2-22.
//
#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<cassert>

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

const int lm_max_iterations = 100; // lm算法迭代次数
double initial_mse = 0.0;
double final_mse = 0.0;


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

double compute_mse(const std::vector<double> &F){
    double mse = 0.0;
    for(int i = 0; i < F.size(); i++){
        mse += F[i] * F[i];
    }
    return mse / (F.size() / 2.0);
}

void lm_optimization() {
    std::vector<double> F; // 重投影误差
    compute_reprojection_errors(F);
    // 计算初始均方误差
    double current_mse = compute_mse(F);
    final_mse = initial_mse = current_mse;
}

int main() {
    load_data("../test_ba.txt");
    lm_optimization();
    return 0;
}