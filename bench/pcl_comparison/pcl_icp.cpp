// Compile ICP from the installed PCL headers; no separate registration .so required.
#include <pcl/registration/icp.h>
#include <pcl/pcl_config.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

template class pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ,float>;
template class pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ,float>;

class Icp : public pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> {
public:
  int updates() const { return nr_iterations_; }
};

int main() {
  std::cout << std::setprecision(17);
  std::size_t n, m;
  double gate;
  int budget;
  // Persistent stdin protocol avoids measuring process creation per registration.
  while (std::cin >> n >> m >> gate >> budget) {
    if (n < 3 || m < 3 || n > 1000000 || m > 1000000 ||
        !std::isfinite(gate) || gate <= 0 || budget < 1 || budget > 1000)
      throw std::runtime_error("invalid ICP request");
    auto source = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto target = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    source->resize(n); target->resize(m);
    for (auto cloud : {source,target}) {
      for (auto &point : *cloud) {
        if (!(std::cin >> point.x >> point.y >> point.z) ||
            !std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
          throw std::runtime_error("invalid XYZ payload");
      }
      cloud->is_dense = true;
    }
    Icp icp;
    icp.setInputSource(source); icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(gate);
    icp.setMaximumIterations(budget);
    icp.setTransformationEpsilon(0);
    icp.setTransformationRotationEpsilon(1); // cosine(0) angular threshold
    icp.setEuclideanFitnessEpsilon(0);
    // ICP resets translation/rotation/relative thresholds inside align; set
    // absolute MSE too. Equal exact transforms can still stop at zero deltas.
    icp.getConvergeCriteria()->setAbsoluteMSE(0);
    pcl::PointCloud<pcl::PointXYZ> output;
    auto start = std::chrono::steady_clock::now();
    icp.align(output);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    auto pose = icp.getFinalTransformation();
    std::cout << PCL_VERSION_PRETTY << ' ' << icp.updates() << ' ' << seconds;
    for (int row=0;row<4;++row) for (int col=0;col<4;++col) std::cout << ' ' << pose(row,col);
    std::cout << '\n' << std::flush;
  }
}
