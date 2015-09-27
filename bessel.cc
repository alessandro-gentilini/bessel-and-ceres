#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include <functional>
#include <random>
std::vector< double > data_x,data_y;

void prepare_data()
{
   std::default_random_engine re(42);   
   std::normal_distribution<double> nd(0,0.05);
   auto gaussian_noise = std::bind(nd, re);

   const double delta = 0.1;
   const double m = 0.3;
   const double c = 0.1;
   for ( int i = 0; i < 100; i++ ) {
      const double x = i*delta;
      data_x.push_back(x);
      data_y.push_back(ceres::BesselJ1(m*x+c)+gaussian_noise());
   }
}

struct MyResidual {
   MyResidual(double x, double y)
      : x_(x), y_(y) {}
   template <typename T> bool operator()(const T* const m,
      const T* const c,
      T* residual) const {
         residual[0] = T(y_) - ceres::BesselJ1(m[0] * T(x_) + c[0]);
         return true;
   }
private:
   const double x_;
   const double y_;
};

int main(int argc, char** argv) {
   prepare_data();

   google::InitGoogleLogging(argv[0]);
   const double m_0 = .0;
   const double c_0 = .0;
   double m = m_0;
   double c = c_0;
   Problem problem;
   for (int i = 0; i < data_x.size(); ++i) {
      problem.AddResidualBlock(
         new AutoDiffCostFunction<MyResidual, 1, 1, 1>(
         new MyResidual(data_x[i], data_y[i])),
         NULL,
         &m, &c);
   }
   Solver::Options options;
   options.max_num_iterations = 25;
   options.linear_solver_type = ceres::DENSE_QR;
   options.minimizer_progress_to_stdout = true;
   Solver::Summary summary;
   Solve(options, &problem, &summary);
   std::cout << summary.BriefReport() << "\n";
   std::cout << "Initial m: " << m_0 << " c: " << c_0 << "\n";
   std::cout << "Final   m: " << m << " c: " << c << "\n";

   return 0;
}