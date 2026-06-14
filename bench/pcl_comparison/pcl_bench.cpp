// Times the same core operations as the SpatialRust `bench_ops` example, using
// PCL, on the same input PCD. Prints `operation,seconds,output_points` lines.
//
// Build (see run.sh):
//   g++ -O2 -std=c++17 -I/usr/include/pcl-1.14 -I/usr/include/eigen3 \
//       pcl_bench.cpp -o pcl_bench \
//       -lpcl_common -lpcl_io -lpcl_filters -lpcl_features -lpcl_search \
//       -lpcl_kdtree -lpcl_octree
#include <chrono>
#include <cstdio>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: pcl_bench <cloud.pcd>\n"); return 1; }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) < 0) {
    std::fprintf(stderr, "failed to read %s\n", argv[1]); return 1;
  }
  std::fprintf(stderr, "loaded %zu points from %s\n", cloud->size(), argv[1]);

  // Voxel-grid downsample (leaf 0.05).
  {
    pcl::PointCloud<pcl::PointXYZ> out;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    auto t0 = Clock::now();
    vg.filter(out);
    auto t1 = Clock::now();
    std::printf("voxel_downsample,%.4f,%zu\n", secs(t0, t1), out.size());
  }

  // Normal estimation (k = 10), single-threaded.
  {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(10);
    pcl::PointCloud<pcl::Normal> normals;
    auto t0 = Clock::now();
    ne.compute(normals);
    auto t1 = Clock::now();
    std::printf("normal_estimation,%.4f,%zu\n", secs(t0, t1), normals.size());
  }

  // Statistical Outlier Removal (k = 16, std = 1.0).
  {
    pcl::PointCloud<pcl::PointXYZ> out;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(16);
    sor.setStddevMulThresh(1.0);
    auto t0 = Clock::now();
    sor.filter(out);
    auto t1 = Clock::now();
    std::printf("statistical_outlier_removal,%.4f,%zu\n", secs(t0, t1), out.size());
  }

  // Radius Outlier Removal (radius 0.1, min 4).
  {
    pcl::PointCloud<pcl::PointXYZ> out;
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(0.1);
    ror.setMinNeighborsInRadius(4);
    auto t0 = Clock::now();
    ror.filter(out);
    auto t1 = Clock::now();
    std::printf("radius_outlier_removal,%.4f,%zu\n", secs(t0, t1), out.size());
  }
  return 0;
}
