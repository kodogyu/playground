#include <iostream>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/BetweenFactor.h>

using namespace std;
using namespace gtsam;

int main() {
    // Create an empty nonlinear factor graph
    NonlinearFactorGraph graph;

    // Add a Gaussian prior on pose x_1
    gtsam::Pose2 priorMean(0.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr priorNoise =
    noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
    graph.add(PriorFactor<Pose2>(1, priorMean, priorNoise));

    // Add two odometry factors
    Pose2 odometry(2.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr odometryNoise =
    noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
    graph.add(BetweenFactor<Pose2>(1, 2, odometry, odometryNoise));
    graph.add(BetweenFactor<Pose2>(2, 3, odometry, odometryNoise));

    // create (deliberately inaccurate) initial estimate
    Values initial;
    initial.insert(1, Pose2(0.5, 0.0, 0.2));
    initial.insert(2, Pose2(2.3, 0.1, -0.2));
    initial.insert(3, Pose2(4.1, 0.1, 0.1));

    // optimize using Levenberg-Marquardt optimization
    Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();
    
    // Print the optimized values
    // cout << "Optimized result:" << endl;
    // cout << "first = " << result.at<Pose2>(1) << endl;
    // cout << "second = " << result.at<Pose2>(2) << endl;
    // cout << "third = " << result.at<Pose2>(3) << endl;
    result.print("Final result:\n");
    
    return 0;
}
