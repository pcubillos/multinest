// X-means, Pelleg, Moore
// module xmeans_clstr

// use kmeans_clstr
// use utils1

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "random_ns.h"
#include "utils.h"
#include "kmeans.h"
#include "xmeans.h"


int num_clusters,
    nclusters,
    pt_clustered;
double **p, **xclsMean, **xclsEval, **aux;
double  *xclsVol, *xclsKfac, *xclsEff, *xclsDetcov;
double ***xclsInvCov, ***xclsTMat, ***xclsCovmat, ***xclsEvec;

/* Gmeans global variables */
double
    **pt_k,  // points in clusters (ndim, npt)
    **aux_k,  // loglike, to change order only (naux,npt)
    **mean_k;  // k-cluster means (2, ndim)

int *xclsPos,
    *ppc;  // points per cluster

extern struct random_ns rand_ns;


void do_gmeans(
    double **points, // IN/OUT: Points, output array is sorted by cluster
    int npt,  // Number of points
    int ndim,  // Dimensionality of points
    int *nClstr, // OUT: Total clusters found
    int *pt_per_cluster,  // OUT: points per cluster
    int naux,  // Dimensionality of auxa
    double **auxa,  // Auxiliary-data array for each point
    int min_pt,  // Minimum number of points per cluster
    int max_clusters){  // Maximum number of clusters

    int i, j;
    nclusters = 1;  // Current number of clusters
    num_clusters = 0;  // Number of clusters created
    pt_clustered = 0;  // Number of points clustered

    if (npt < 2*min_pt){
        *nClstr = 1;
        pt_per_cluster[0] = npt;
        return;
    }

    /* Allocate global variables */
    p = (double **)malloc(ndim * sizeof(double *));
    p[0] = (double *)malloc(ndim*npt * sizeof(double));
    for (i=1; i<ndim; i++)
        p[i] = p[0] + npt*i;

    aux = (double **)malloc(naux * sizeof(double *));
    aux[0] = (double *)malloc(naux*npt * sizeof(double));
    for (i=1; i<naux; i++)
        aux[i] = aux[0] + npt*i;

    xclsPos = (int *)calloc(max_clusters, sizeof(int));
    ppc = (int *)malloc(max_clusters * sizeof(int));
    for (i=0; i<max_clusters; i++)
        ppc[i] = -1;

    // Points in clusters (ndim, npt)
    pt_k = (double **)malloc(ndim * sizeof(double *));
    pt_k[0] = (double *)malloc(ndim*npt * sizeof(double));
    for (i=1; i<ndim; i++){
        pt_k[i] = pt_k[0] + npt*i;
    }

    // Points in clusters (2, ndim, npt)
    //pt_k = (double ***)malloc(2 * sizeof(double **));
    //pt_k[0] = (double **)malloc(2*ndim * sizeof(double *));
    //pt_k[0][0] = (double *)malloc(2*ndim*npt * sizeof(double));
    //for (i=0; i<2; i++){
    //    if (i != 0)
    //        pt_k[i] = pt_k[0] + ndim*i;
    //    for (j=0; j<npt; j++)
    //        if (i != 0 || j != 0)
    //            pt_k[i][j] = pt_k[0][0] + npt*ndim*i + npt*j;
    //}

    // loglike, to change order only (naux,npt)
    aux_k = (double **)malloc(naux * sizeof(double *));
    aux_k[0] = (double *)malloc(naux*npt * sizeof(double));
    for (i=1; i<naux; i++){
        aux_k[i] = aux_k[0] + npt*i;
    }

    // Mean of each new kmean cluster (2,ndim)
    mean_k = (double **)malloc(2 * sizeof(double *));
    mean_k[0] = (double *)malloc(2*ndim * sizeof(double));
    mean_k[1] = mean_k[0] + ndim;

    gmeans(points, npt, ndim, naux, auxa, min_pt, max_clusters);

    j = 0;
    for (i=0; i<num_clusters; i++)
        if (ppc[xclsPos[i]] >= 0){
            pt_per_cluster[j] = ppc[xclsPos[i]];
            j += 1;
        }
    *nClstr = j;

    for (j=0; j<npt; j++){
        for (i=0; i<ndim; i++)
            points[i][j] = p[i][j];
        for (i=0; i<naux; i++)
            auxa[i][j] = aux[i][j];
    }

    free(p[0]);
    free(p);
    free(aux[0]);
    free(aux);
    free(xclsPos);
    free(ppc);
    free(pt_k[0]);
    free(pt_k);
    free(aux_k[0]);
    free(aux_k);
    free(mean_k[0]);
    free(mean_k);
}


void gmeans(
    double **pt, // points
    int npt,  // Number of points
    int ndim,  // Points dimensionality
    int naux, // auxa dimensionality
    double **auxa, // Auxilliary points
    int min_pt,
    int max_clusters){

    int *cluster;  // Cluster array having cluster num of each pt (npt)
    int i, j, ip, i1, k;
    int attempt_cluster = 1;  // attempt to cluster
    int npt_k[2] = {0, 0};  // no. of points in the clusters
    int npt_cluster=0;
    double ad=0.0;

    cluster = (int *)malloc(npt * sizeof(int));

    // Don't cluster if cluster has less than 2*(min_pt-1)+1 points
    // since it result in clusters having less than min_pt points
    if (npt < 2*min_pt || nclusters == max_clusters)
        attempt_cluster = 0;

    if (attempt_cluster == 1){
        // Breakup the points in 2 clusters
        i1 = 2;
        kmeans3(i1, pt, npt, ndim, mean_k, cluster, min_pt);
        // Get number of points in each cluster:
        for (i=0; i<npt; i++)
            npt_k[cluster[i]] += 1;

        // Don't cluster if either of the clusters has fewer than
        // min_pt points since that might be noise
        if (npt_k[0] < min_pt || npt_k[1] < min_pt){
            attempt_cluster = 0;
        }
    }
    // Calculate the means of the two clusters & their difference
    if (attempt_cluster == 1){
        // Recycle mean_k[0] as delta_mean:
        for (i=0; i<ndim; i++)
            mean_k[0][i] = mean_k[0][i] - mean_k[1][i];
        // Inference at alpha=0.0001
        //if (anderson_darling(npt, ndim, pt, mean_k[0]) < 1.8692)
        ad = anderson_darling(npt, ndim, pt, mean_k[0]);
        if (ad < 1.8692)
            attempt_cluster = 0;
    }

    if (attempt_cluster == 1){
        nclusters += 1;
        // Separate clusters and try further kmeans clustering:
        npt_cluster = npt_k[0];
        npt_k[0] = npt_k[1] = 0;
        for (j=0; j<npt; j++){
            k = cluster[j];
            for (i=0; i<ndim; i++)
                pt_k[i][npt_cluster*k + npt_k[k]] = pt[i][j];
            for (i=0; i<naux; i++)
                aux_k[i][npt_cluster*k + npt_k[k]] = auxa[i][j];
            npt_k[k] += 1;
        }
        free(cluster);
        for (j=0; j<npt_k[0]; j++){
            for (i=0; i<ndim; i++)
                pt[i][j] = pt_k[i][j];
            for (i=0; i<naux; i++)
                auxa[i][j] = aux_k[i][j];
        }
        gmeans(pt, npt_k[0], ndim, naux, auxa, min_pt, max_clusters);
        for (j=0; j<npt_k[1]; j++){
            for (i=0; i<ndim; i++)
                pt[i][j] = pt_k[i][npt_cluster+j];
            for (i=0; i<naux; i++)
                auxa[i][j] = aux_k[i][npt_cluster+j];
        }
        gmeans(pt, npt_k[1], ndim, naux, auxa, min_pt, max_clusters);
        return;
    }
    free(cluster);

    /* Do not attempt further clustering */
    for (j=0; j<npt; j++){
        for (i=0; i<ndim; i++)
            p[i][pt_clustered+j] = pt[i][j];
        for (i=0; i<naux; i++)
            aux[i][pt_clustered+j] = auxa[i][j];
    }
    ip = bin_search(ppc, num_clusters, npt);
    for (i=num_clusters; i>ip; i--)
        ppc[i] = ppc[i-1];
    ppc[ip] = npt;

    for (i=0; i<num_clusters; i++)
        if (xclsPos[i] >= ip)
            xclsPos[i] += 1;
    xclsPos[num_clusters] = ip;

    pt_clustered += npt;
    num_clusters += 1;
}


void rescale(int ndim, int npt, double **pt){
    // Rescale the points so that all have the same range
    int i, j;
    double d1, d2;

    for (i=0; i<ndim; i++){
        d1 = d2 = pt[i][0];
        for (j=1; j<npt; j++){
            if (d2 < pt[i][j])
                d2 = pt[i][j];
            if (d1 > pt[i][j])
                d1 = pt[i][j];
        }
        for (j=0; j<npt; j++)
            pt[i][j] = (pt[i][j]-d1) / (d2-d1);
    }
}


double set_weights(
    int npt,  // Number of points
    double **like,  // Log-like & log of dX of points
    double *weights,  // Weights
    int set_weight){  // Set the weights
    // Normalized normal probability for the points given a cluster
    int j;
    double locZ;  // Local evidence

    locZ = -DBL_MAX * DBL_EPSILON; // logZero
    for (j=0; j<npt; j++)
        if (weights[j] > 0.0)
            locZ = log_sum_exp(locZ, like[0][j] + like[1][j] + log(weights[j]));

    // Calculate the posterior probabilty weights
    if (set_weight)
        for (j=0; j<npt; j++)
            if (weights[j] > 0.0)
                weights[j] *= exp(like[0][j] + like[1][j] - locZ);

    return locZ;
}


void normal_cluster_probability(  // normalProbClsGivPt
    int nclusters, //no. of clusters
    int ndim, // Dimensionality
    double *p, // Point (ndim)
    double like, // log-like of the point
    double *lowlike, //lowest log-like of each cluster (nclusters)
    double **mean, //mean (nclusters,ndim)
    double ***invcov, //inverse covariance matrix (nclusters,ndim,ndim)
    double *detcov, //determinant of the covariance matrix (nclusters)
    double *cwt, //cluster prior probabilities (nclusters)
    double *prob){ // Output (nclusters)
    // Probability of each cluster given a particular point
    // assuming the model is Gaussian Mixture

    int i, j, k;
    double *a, **tpt, d2;
    int flag=0;

    a = (double *)calloc(nclusters, sizeof(double));
    tpt = (double **)malloc(nclusters * sizeof(double *));
    tpt[0] = (double *)malloc(nclusters*ndim * sizeof(double));
    for (i=1; i<nclusters; i++)
        tpt[i] = tpt[0] + ndim*i;

    // Calculate point-mean
    for (i=0; i<nclusters; i++)
        for (j=0; j<ndim; j++)
            tpt[i][j] = p[j] - mean[i][j];

    // a=sum(Transpose(pt-mean).invcov.(pt-mean))
    for (i=0; i<nclusters; i++)
        if (lowlike[i] >= like){
            for (j=0; j<ndim; j++){
                a[i] += tpt[i][j] * tpt[i][j] * invcov[i][j][j];
                for (k=j; k<ndim; k++)
                    a[i] += 2.0 * tpt[i][j] * tpt[i][k] * invcov[i][j][k];
            }
            prob[i] = log(cwt[i]) - 0.5*log(detcov[i]) - 0.5*a[i];
            flag = 1;
        }else
            prob[i] = -DBL_MAX * DBL_EPSILON;  // logZero

    free(a);
    free(tpt[0]);
    free(tpt);

    if (flag == 0){
        for (i=0; i<nclusters; i++)
            prob[i] = 0.0;
        return;
    }

    d2 = prob[0];
    for (i=1; i<nclusters; i++)
        d2 = log_sum_exp(d2, prob[i]);

    if (prob[i] == -DBL_MAX * DBL_EPSILON)
        prob[i] = 0.0;
    else
        for (i=1; i<nclusters; i++)
            prob[i] = exp(prob[i]-d2);
}


double anderson_darling(
    int npt,  // Number of points
    int ndim,  // Dimensionality
    double **pt,  // Points (ndim, npt)
    double *delta_mean){ // Difference between the two cluster means (ndim)
    // Anderson Darling test for normal distribution

    double *ppt;  // projected & normalized points (npt)
    double mean=0.0,  // mean of the projected & normalized points
        sigma=0.0,  // st.dev. of the projected & normalized points
        A,  // Anderson Darling statistic
        stn1, stn2,
        mode=0.0;
    int i, j, ii;

    ppt = (double *)calloc(npt, sizeof(double));
    // project the points onto delta_mean & calculate the mean & sigma
    // of the projected points
    for (j=0; j<ndim; j++)
        mode += pow(delta_mean[j], 2.0);
    mode = sqrt(mode);

    for (i=0; i<npt; i++){
        for (j=0; j<ndim; j++)
            ppt[i] += pt[j][i] * delta_mean[j] / mode;
        mean += ppt[i];
        sigma += pow(ppt[i], 2.0);
    }
    mean = mean / npt;
    sigma = sqrt(sigma/npt - pow(mean,2.0)) * npt / (npt-1.0);

    //transform the projected points into ones with mean=0 & sigma=1
    for (i=0; i<npt; i++)
        ppt[i] = (ppt[i]-mean) / sigma;

    // Sort ppt in ascending order
    quicksort(ppt, npt);

    // Calculate Anderson Darling statistic:
    A = 0.0;
    for (i=0; i<npt; i++){
        if (ppt[i] > 5.0)
            stn1 = 1.0;
        else if (ppt[i] < -5.0)
            stn1 = 0.0;
        else
            stn1 = st_normal_cdf(ppt[i]);

        ii = npt - i - 1;
        if (ppt[ii] > 5.0)
            stn2 = 1.0;
        else if (ppt[ii] < -5.0)
            stn2 = 0.0;
        else
            stn2 = st_normal_cdf(ppt[ii]);
        
        A += (2.0*i+1.0) * (log(stn1) + log(1.0-stn2));
    }
    A = -A / (1.0*npt) - npt;
    A *= (1.0 + 4.0/npt - 25.0/(npt*npt));

    free(ppt);

    return A;
    // Inference at alpha=0.0001
    //if (A > 1.8692)
    //    return 0;
    //return 1;
}


void gauss_prop(
    int npt,  // Number of points
    int ndim,  // Dimensionality
    double **pt,  // Points (ndim,npt)
    double *weights,  // Probability weights (npt)
    double *norm_weights,  // Evidence-normalized probability weights (npt)
    double cwt, //no. of points in the cluster
    //output variables
    double *mean,  // mean (ndim)
    double **covmat,  // Covariance matrix (ndim,ndim)
    double **invcov, // inverse covariance matrix(ndim,ndim)
    double *eigen_vals,  // Eigen values (ndim)
    double **eigen_vecs,  // Eigen vectors (ndim,ndim)
    double detcov, //determinant of the covariance matrix
    int init_eigen, //initialize the LAPACK eigenanalysis routine?
    int calc_mean){ // calculate the mean?

    int i, j, k;
    // total no. of points in the cluster
    cwt = 0.0;
    for (i=0; i<npt; i++)
        cwt += weights[i];

    if (calc_mean == 1)
        for (j=0; j<ndim; j++){
            mean[j] = 0.0;
            for (i=0; i<npt; i++)
                mean[j] += pt[j][i] * norm_weights[i];
        }

    // Covariance matrix:
    calc_covmat_wt(npt, ndim, pt, norm_weights, mean, covmat);
    // Eigen analysis:
    for (i=0; i<ndim; i++)
        for (j=0; j<ndim; j++)
            eigen_vecs[i][j] = covmat[i][j];
    diagonalize(eigen_vecs, eigen_vals, ndim, init_eigen);
    // Eigenvalues of covariance matrix can't be zero:
    for (j=0; j<ndim; j++)
        if (eigen_vals[j] <= 0.0)
            for (k=0; k<j; k++)
                eigen_vals[k] = eigen_vals[j+1] / 100.0;

    // Determinant of the covariance matrix:
    detcov = product(eigen_vals, ndim);

    // Inverse of covariance matrix:
    inverse_covariance(ndim, eigen_vecs, eigen_vals, invcov);
}


//----------------------------------------------------------------------

// Gaussian mixture by expectation maximization
//some points are constrained to be in specific clusters
//likelihood constraint
void GaussMixExpMaxLike(
    int ndim, // Dimensionality
    int nclusters, //no. of clusters
    int npt, //total no. of points
    int *ncon, //no. constrained points in each cluster (nclusters)
    int norm, // rescale the points so that all have the same range?
    double **pt,  // points(ndim,npt)
    double **like,  // log-likelihood and log of the dx (2,npt)
    double **weights, // OUT: points probability weights (npt,nclusters)
    double **norm_weights, // OUT:normalized probability weights (npt,nclusters)
    double *locZ,  // OUT: local evidence (nclusters)
    double *lowlike, // (nclusters)
    int init_eigen){ //initialize the LAPACK eigen-analysis routines?

    int i, j, k, count=0;
    double d1;
    double **pts;  // (ndim,npt)
    double **mean,  // (nclusters,ndim)
        **eigen_vals,  // (nclusters,ndim)
        ***eigen_vecs,  // (nclusters,ndim,ndim)
        *cwt;  // (nclusters)
    double *old_locZ,  // (nclusters)
        ***covmat,  // (nclusters,ndim,ndim)
        ***invcov,  // (nclusters,ndim,ndim)
        *detcov;  // (nclusters)
    double **t_weights, **t_norm_weights;  // transpose matrices of weights
    double *point;

    int calc_mean=1, set_weight=1;

    //ndim = ndim;

    t_norm_weights = (double **)malloc(nclusters * sizeof(double *));
    t_norm_weights[0] = (double *)calloc(nclusters*npt, sizeof(double));
    for (j=1; j<nclusters; j++)
        t_norm_weights[j] = t_norm_weights[0] + npt*j;

    t_weights = (double **)malloc(nclusters * sizeof(double *));
    t_weights[0] = (double *)calloc(nclusters*npt, sizeof(double));
    for (j=1; j<nclusters; j++)
        t_weights[j] = t_weights[0] + npt*j;

    point = (double *)malloc(ndim * sizeof(double));
    pts = (double **)malloc(ndim * sizeof(double *));
    pts[0] = (double *)malloc(ndim*npt * sizeof(double));
    for (i=1; i<ndim; i++)
        pts[i] = pts[0] + npt*i;

    for (i=0; i<ndim; i++)
        for (j=0; j<npt; j++)
            pts[i][j] = pt[i][j];

    // rescaling
    if (norm)
        rescale(ndim, npt, pts);

    for (i=0; i<npt; i++)
        for (j=0; j<nclusters; j++)
            weights[i][j] = 0.0;

    // Set the initial probability weights
    k = 0;
    for (j=0; j<nclusters; j++){
        for (i=k; i<k+ncon[j]; i++)
            t_norm_weights[j][i] = t_weights[j][i] = weights[i][j] = 1.0;
        k += ncon[j];
    }

    // Calculate the initial Gaussian mixture model
    for (j=0; j<nclusters; j++){
        // Calculate the evidence & normalized weights
        locZ[j] = set_weights(npt, like, t_norm_weights[j], set_weight);
        // now the Gaussian with normalized weights
        gauss_prop(
            npt, ndim, pts, t_weights[j], t_norm_weights[j], cwt[j],
            mean[j], covmat[j], invcov[j], eigen_vals[j], eigen_vecs[j],
            detcov[j], init_eigen, calc_mean);
        // FINDME: Might need some pointer wizardry
    }
    // Normalize cluster prior probabilities
    d1 = 0.0;
    for (j=0; j<nclusters; j++)
        d1 += cwt[j];
    for (j=0; j<nclusters; j++)
        cwt[j] /= d1;

    k = 0;
    for (j=0; j<nclusters; j++)
        k += ncon[j];
    // Expectation-maximization:
    while (1){
        count += 1;
        old_locZ = locZ;
        //check all the points now
        for (i=k; i<npt; i++)
            for (j=0; j<ndim; j++)
                point[j] = pts[j][i];
            // Calculate the probability of points lying in each cluster
            normal_cluster_probability(
                nclusters, ndim, point, like[0][i], lowlike, mean,
                invcov, detcov, cwt, weights[i]);

        // Re-calculate the Gaussian mixture model
        for (i=0; i<npt; i++)
            for (j=0; j<nclusters; j++)
                t_norm_weights[j][i] = t_weights[j][i] = weights[i][j];

        for (j=0; j<nclusters; j++){
            //calculate the evidence & normalized weights
            locZ[j] = set_weights(npt, like, t_norm_weights[j], set_weight);
            //now the Gaussian with normalized weights
            gauss_prop(npt, ndim, pts, t_weights[j], t_norm_weights[j],
                cwt[j], mean[j], covmat[j],
                invcov[j], eigen_vals[j], eigen_vecs[j],
                detcov[j], 0, calc_mean);
        }
        // Normalize cluster prior probabilities
        d1 = 0.0;
        for (j=0; j<nclusters; j++)
            d1 += cwt[j];
        for (j=0; j<nclusters; j++)
            cwt[j] /= d1;
        // Check for convergence
        d1 = 0.0;
        for (j=0; j<nclusters; j++)
            d1 += pow(old_locZ[j] - locZ[j], 2.0);
        d1 = sqrt(d1);
        if (d1 < 0.0001 || count==100){
            //weights = norm_weights
            break;
        }
    }

    for (i=0; i<npt; i++)
        for (j=0; j<nclusters; j++)
            norm_weights[i][j] = t_norm_weights[j][i];

    free(point);
    free(pts[0]);
    free(pts);
    free(t_norm_weights[0]);
    free(t_norm_weights);
    free(t_weights[0]);
    free(t_weights);
}

