// soft k-means clustering with k=2
// D.J.C. MacKay, Information Theory, Inference & Learning Algorithms, 2003, p.304
// Aug 2006

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "ind.h"
#include "random_ns.h"
#include "utils.h"
#include "kmeans.h"

// use RandomNS
// use utils1

extern struct random_ns rand_ns;


void kmeans3(
    int k,  // Number of clusters required
    double **pt,  // Points (ndim,npt)
    int npt,  // Number of points
    int ndim,  // Dimensionality
    double **means,  // OUT: Mean value for each cluster (k,ndim)
    int *cluster,  // OUT: Cluster membership of each point for k clusters (npt)
    int min_pt){  // Minimum number of points allowed in a cluster
    // Simple k-means:

    int i1, i2, i3;
  
    double **dis; // (min_pt,2)
    int *old_cluster, // (npt),
        **r,  // (npt,k),
        *totR,  // Number of points per cluster (k)
        *scrap;  // Indices of starting point for each cluster (k)
    double temp, dist, urv=0.0, d1;
    int i, j, l, x;
    int clstrd=0, flag;

    if (k > npt/min_pt + 1)
        k = npt/min_pt + 1;

    if (k==1){
        for (i=0; i<ndim; i++){
            means[0][i] = 0.0;
            for (l=0; l<npt; l++)
                means[0][i] += pt[i][l];
            means[0][i] /= (double)npt;
        }
        for (j=0; j<npt; j++)
            cluster[j] = 0;
        return;
    }

    dis = (double **)malloc(min_pt * sizeof(double *));
    dis[0] = (double *)malloc(min_pt*2 * sizeof(double));
    for (i=1; i<min_pt; i++)
        dis[i] = dis[0] + 2*i;
    /* scrap keeps starting random indices to ensure they are all different */
    scrap = (int *)malloc(k * sizeof(int));
        for (j=0; j<k; j++)
            scrap[j] = -1;
    totR = (int *)calloc(k, sizeof(int));

    old_cluster = (int *)malloc(npt * sizeof(int));
    for (i=0; i<npt; i++)
        old_cluster[i] = -1;

    r = (int **)malloc(npt * sizeof(int *));
    r[0] = (int *)malloc(npt*k * sizeof(int));
    for (i=1; i<npt; i++)
        r[i] = r[0] + k*i;

    // Choose random points as starting positions
    for (i=0; i<k; i++){
        while (1){
            flag = 0;
            urv = ranmarns(&rand_ns, 0);
            x = (int) ((double)npt*urv);
            for (j=0; j<i; j++)
                if (x == scrap[j]){
                    flag = 1;  /* Repeated index, draw again */
                    break;
                }
            if (flag){
                continue;
            }else{
                scrap[i] = x;
                break;
            }
        }
        for (j=0; j<ndim; j++)
            means[i][j] = pt[j][x];
    }

    while (1){
        for (i=0; i<npt; i++){
            temp = DBL_MAX;
            for (j=0; j<k; j++){
                dist = 0.0;
                for (l=0; l<ndim; l++)
                    dist += pow(means[j][l] - pt[l][i], 2.0);
                if (dist < temp){
                    temp = dist;
                    x = j;
                }
            }
            for (j=0; j<k; j++)
                r[i][j] = 0;
            r[i][x] = 1;
            cluster[i] = x;
        }

        clstrd = 1;
        /* Check cluster membership has changed */
        for (i=0; i<npt; i++)
            if (old_cluster[i] != cluster[i]){
                clstrd = 0;
                break;
            }

        if (clstrd){
            // Check if all the clusters have more than min_pt points
            for (i=0; i<k; i++)
                if (totR[i] < min_pt){
                    for (j=0; j<min_pt; j++)
                        for (l=0; l<2; l++)
                            dis[j][l] = DBL_MAX;
                    /* Number of points required to get to min_pt */
                    i1 = min_pt - totR[i];
                    /* Find i1 closest points to means[i] (dis[:][0]) */
                    /* and their indices (dis[:][1]) */
                    for (j=0; j<npt; j++)
                        if (cluster[j] != i && totR[cluster[j]] > min_pt){
                            d1 = 0.0;
                            for (l=0; l<ndim; l++)
                                d1 += pow(means[i][l]-pt[l][j], 2.0);
                            i3 = -1;
                            for (i2=i1-1; i2>=0; i2--){
                                if (d1 < dis[i2][0])
                                    i3 = i2;
                                else
                                    break;
                            }
                            if (i3 != -1){
                                for (l=i1-1; l<i3; l--){
                                    dis[l][0] = dis[l-1][0];
                                    dis[l][1] = dis[l-1][1];
                                }
                                dis[i3][0] = d1;
                                dis[i3][1] = (double)j;
                            }
                        }
                    /* Swap the points */
                    for (j=0; j<i1; j++){
                        i3 = (int)dis[j][1];
                        i2 = cluster[i3];
                        cluster[i3] = i;
                        totR[i] += 1;
                        totR[i2] -= 1;
                        for (l=0; l<k; l++)
                            r[i3][l] = 0;
                        r[i3][i] = 1;
                    }
                }

            for (i=0; i<k; i++)
                // Update means:
                for (j=0; j<ndim; j++){
                    means[i][j] = 0.0;
                    for (l=0; l<npt; l++)
                        means[i][j] += r[l][i] * pt[j][l];
                    means[i][j] /= totR[i];
                }
            break;
        }

        for (j=0; j<npt; j++)
            old_cluster[j] = cluster[j];

        for (i=0; i<k; i++){
            totR[i] = 0;
            for (j=0; j<npt; j++)
                totR[i] += r[j][i];
            if (totR[i] == 0)
                continue;
            // Update means:
            for (j=0; j<ndim; j++){
                means[i][j] = 0.0;
                for (l=0; l<npt; l++)
                    means[i][j] += r[l][i] * pt[j][l];
                means[i][j] /= totR[i];
            }
        }
    }

    free(old_cluster);
    free(totR);
    free(scrap);
    free(r[0]);
    free(r);
    free(dis[0]);
    free(dis);
}


double delF(
    int ndim,  // Dinemsionality
    int n1,
    int n2, //no. of points in each ellipsoid
    double mdis1,  // Mahalanobis distance of the point from both ellipsoids
    double mdis2,
    double detcov1, // Determinant of the covariance matrices of the ellipsoids
    double detcov2,
    double kfac1,  //overall enlargement factors of the ellipsoids
    double kfac2){
    //calculate the variation in weighted average of e-tightness functions of 2
    //components for re-assigning a point from component 1 to 2
    //Choi, Wang & Kim, Eurographics 2007, vol. 26, No. 3
    double delf =
        (pow(n1/(n1-1.0),3.0)*(1.0-mdis1/(n1-1.0))-1.0)
        * sqrt(detcov1*pow(kfac1,ndim))
        / (pow(n1/(n1-1.0),1.5) * sqrt(1.0-mdis1/(n1-1.0)) + 1.0);

    delf +=
        (pow(n2/(n2+1.0),3.0)*(1.0+mdis2/(n2+1.0))-1.0)
        * sqrt(detcov2*pow(kfac2,ndim))
        / (pow(n2/(n2+1.0),1.5) * sqrt(1.0+mdis2/(n2+1.0)) + 1.0);

    delf /= (double)(n1+n2);
    return delf;
}

