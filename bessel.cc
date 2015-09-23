#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include <cmath>

namespace ceres {

   // j0 is the Bessel functions of the first kind with integer order equal to 0
   inline double Bessel_J_0     (double x) { return j0(x);      } 

   // j1 is the Bessel functions of the first kind with integer order equal to 1
   inline double Bessel_J_1     (double x) { return j1(x);      } 

   // jn is the Bessel functions of the first kind with integer order equal to n
   inline double Bessel_J_2     (double x) { return jn(2,x);    } 

   // http://dlmf.nist.gov/10.6#E3
   // j0(a + h) ~= j0(a) - j1(a) h
   template <typename T, int N> inline
      Jet<T, N> Bessel_J_0(const Jet<T, N>& f) {
         return Jet<T, N>(Bessel_J_0(f.a), -Bessel_J_1(f.a) * f.v);
   }

   // http://dlmf.nist.gov/10.6#E1
   // j1(a + h) ~= j1(a) + 0.5 ( j0(a) - j2(a) ) h
   template <typename T, int N> inline
      Jet<T, N> Bessel_J_1(const Jet<T, N>& f) {
         return Jet<T, N>(Bessel_J_1(f.a), T(0.5) * ( Bessel_J_0(f.a)-Bessel_J_2(f.a) ) * f.v);
   }

   // http://dlmf.nist.gov/10.6#E2
   // j2(a + h) ~= j2(a) + ( j1(a) - (2/a) j2(a) ) h
   template <typename T, int N> inline
      Jet<T, N> Bessel_J_2(const Jet<T, N>& f) {
         return Jet<T, N>(Bessel_J_2(f.a), ( Bessel_J_1(f.a)-(T(2)/f.a)*Bessel_J_2(f.a) ) * f.v);
   }
}


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
      data_y.push_back(ceres::Bessel_J_1(m*x+c)+gaussian_noise());
   }
}

struct MyResidual {
   MyResidual(double x, double y)
      : x_(x), y_(y) {}
   template <typename T> bool operator()(const T* const m,
      const T* const c,
      T* residual) const {
         residual[0] = T(y_) - ceres::Bessel_J_1(m[0] * T(x_) + c[0]);
         return true;
   }
private:
   const double x_;
   const double y_;
};

int main(int argc, char** argv) {
   prepare_data();

   google::InitGoogleLogging(argv[0]);
   double m = 0.;
   double c = 0.;
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
   std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
   std::cout << "Final   m: " << m << " c: " << c << "\n";

   return 0;
}