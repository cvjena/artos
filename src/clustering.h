#ifndef ARTOS_CLUSTERING_H
#define ARTOS_CLUSTERING_H

#include <Eigen/Core>
#include "Random.h"

namespace ARTOS
{

/**
* Applies Lloyd's k-means clustering algorithm to a set of m n-dimensional data points, grouping
* them into exactly k clusters.
*
* @param[in] dataPoints Eigen matrix with data points, one per row. So, for a total of m n-dimensional
* data points, this will be a m x n matrix.
*
* @param[in] k Number of clusters to form.
*
* @param[out] assignments Pointer to an Eigen vector with m integral elements, which will receive
* information about the assignment of each data point to a cluster. So, if the i-th data point belongs to
* the j-th cluster, assignments(i) will be j. May be NULL.
*
* @param[out] centroids Pointer to an k x n matrix, which the computed centroids of each cluster will
* be written to, one per row. May be NULL.
*/
template<typename Derived, typename DerivedCent>
void kMeansClustering(const Eigen::MatrixBase<Derived> & dataPoints, const unsigned int k,
                      Eigen::VectorXi * assignments, Eigen::MatrixBase<DerivedCent> * centroids = NULL)
{
    // Initialize centroids matrices and assignment vector
    typedef typename Derived::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
    const int numDataPoints = dataPoints.rows();
    Mat centr(k, dataPoints.cols());
    Eigen::VectorXi assign(numDataPoints);
    
    // Choose k initial centroids at random
    Random::seedOnce();
    {
        int r;
        Eigen::Matrix<bool, Eigen::Dynamic, 1> chosen(numDataPoints);
        chosen.setConstant(false);
        for (unsigned int i = 0; i < k; i++)
        {
            do
            {
                r = Random::getInt(numDataPoints - 1);
                for (unsigned int j = 0; j < i && !chosen(r); j++)
                    if (centr.row(j).isApprox(dataPoints.row(r), 1e-4))
                        chosen(r) = true;
            }
            while (chosen(r));
            chosen(r) = true;
            centr.row(i) = dataPoints.row(r);
        }
    }
    
    int c, d, l;
    Scalar distance, minDistance;
    int minDistanceIndex;
    Eigen::VectorXi numAssignments(k);
    bool assignmentsChanged;
    for (assignmentsChanged = true, l = 0; assignmentsChanged && l < 10000; l++)
    {
        // Assign data points to clusters
        assignmentsChanged = false;
        for (d = 0; d < numDataPoints; d++)
        {
            minDistanceIndex = -1;
            for (c = 0; c < k; c++)
            {
                distance = (centr.row(c) - dataPoints.row(d)).squaredNorm();
                if (minDistanceIndex < 0 || distance < minDistance)
                {
                    minDistance = distance;
                    minDistanceIndex = c;
                }
            }
            if (assign(d) != minDistanceIndex)
            {
                assign(d) = minDistanceIndex;
                assignmentsChanged = true;
            }
        }
        
        // Compute new centroids
        centr.setZero();
        numAssignments.setZero();
        for (d = 0; d < numDataPoints; d++)
        {
            centr.row(assign(d)) += dataPoints.row(d);
            numAssignments(assign(d)) += 1;
        }
        centr = centr.cwiseQuotient(numAssignments.cast<Scalar>().replicate(1, centr.cols()));
    }
    
    // Copy results to output parameters
    if (assignments != NULL)
        *assignments = assign;
    if (centroids != NULL)
        *centroids = centr;
}


/**
* Runs kMeansClustering() multiple times and returns the result with the least reconstruction error
* (i. e. the sum of the distances between the data points and their cluster's centroid).
*
* @param[in] dataPoints Eigen matrix with data points, one per row. So, for a total of m n-dimensional
* data points, this will be a m x n matrix.
*
* @param[in] k Number of clusters to form.
*
* @param[out] assignments Pointer to an Eigen vector with m integral elements, which will receive
* information about the assignment of each data point to a cluster. So, if the i-th data point belongs to
* the j-th cluster, assignments(i) will be j. May be NULL.
*
* @param[out] centroids Pointer to an k x n matrix, which the computed centroids of each cluster will
* be written to, one per row. May be NULL.
*
* @param[in] numRuns Number of times to run clustering.
*/
template<typename Derived, typename DerivedCent>
void repeatedKMeansClustering(const Eigen::MatrixBase<Derived> & dataPoints, const unsigned int k,
                      Eigen::VectorXi * assignments, Eigen::MatrixBase<DerivedCent> * centroids = NULL,
                      const unsigned int numRuns = 10)
{
    // Initialize centroids matrices and assignment vector
    typedef typename Derived::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
    const int numDataPoints = dataPoints.rows();
    Mat centr(k, dataPoints.cols());
    Eigen::VectorXi assign(numDataPoints);
    
    // Run kMeansClustering() multiple times
    double reconstError, minReconstError;
    for (unsigned int i = 0; i < numRuns; i++)
    {
        // Cluster
        kMeansClustering(dataPoints, k, &assign, &centr);
        // Compute reconstruction error
        reconstError = 0;
        for (unsigned int j = 0; j < numDataPoints; j++)
            reconstError += static_cast<double>((dataPoints.row(j) - centr.row(assign(j))).squaredNorm());
        // Update results
        if (i == 0 || reconstError < minReconstError)
        {
            minReconstError = reconstError;
            if (assignments != NULL)
                *assignments = assign;
            if (centroids != NULL)
                *centroids = centr;
        }
    }
}


/**
* Modifies the results of a clustering operation, so that clusters whose centroids are close to each other
* will be merged.
*
* @param[in,out] assignments Eigen vector of assignments of each data point to the cluster it belongs to,
* identified by the numerical ID of the cluster.
*
* @param[in,out] centroids The centroids of the clusters, one per row. Centroids of merged clusters will
* be re-calculated.
*
* @param[in] minDistance The minimum distance between two clusters. Each pair of clusters whose distance
* is less than this value will be merged.
*/
template<typename Derived>
void mergeNearbyClusters(Eigen::VectorXi & assignments, Eigen::MatrixBase<Derived> & centroids, const typename Derived::Scalar minDistance)
{
    typedef typename Derived::Scalar Scalar;
    const int numCentroids = centroids.rows();
    Eigen::VectorXi mapping = Eigen::VectorXi::LinSpaced(numCentroids, 0, numCentroids - 1);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> eliminated = Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(numCentroids, false);
    bool merged = true;
    Scalar distance, minDist;
    int i, j, c, minDistIndex, numCentr1, numCentr2;
    while (merged)
    {
        merged = false;
        for (i = 0; i < numCentroids; i++)
            if (!eliminated(i))
            {
                // Search for centroid closest to centroid i
                minDistIndex = -1;
                for (j = i + 1; j < numCentroids; j++)
                    if (!eliminated(j))
                    {
                        distance = (centroids.row(i) - centroids.row(j)).norm();
                        if (minDistIndex < 0 || distance < minDist)
                        {
                            minDistIndex = j;
                            minDist = distance;
                        }
                    }
                // If closer than minDistance, merge i and minDistIndex
                if (minDistIndex >= 0 && minDist < minDistance)
                {
                    // Re-compute centroid i
                    numCentr1 = numCentr2 = 0;
                    for (j = 0; j < assignments.size(); j++)
                        if (mapping(assignments(j)) == i)
                            numCentr1++;
                        else if (mapping(assignments(j)) == minDistIndex)
                            numCentr2++;
                    centroids.row(i) = (static_cast<Scalar>(numCentr1) * centroids.row(i) + static_cast<Scalar>(numCentr2) * centroids.row(minDistIndex))
                                   / static_cast<Scalar>(numCentr1 + numCentr2);
                    // Re-assign data points
                    mapping(minDistIndex) = i;
                    eliminated(minDistIndex) = true;
                    merged = true;
                }
            }
    }
    // Apply new centroids and mapping
    if (eliminated.any())
    {
        Derived newCentroids(numCentroids - eliminated.count(), centroids.cols());
        for (i = 0, c = 0; i < numCentroids; i++)
            if (!eliminated(i))
            {
                for (j = 0; j < assignments.size(); j++)
                    if (mapping(assignments(j)) == i)
                        assignments(j) = c;
                newCentroids.row(c++) = centroids.row(i);
            }
        centroids = newCentroids;
    }
}

}

#endif