// X-means, Pelleg, Moore
// module xmeans_clstr

void rescale(int ndim, int npt, double **pt);

double set_weights(int npt, double **like, double *weights, int set_weight);

void normal_cluster_probability(
    int nclusters, int ndim, double *p, double like, double *lowlike,
    double **mean, double ***invcov, double *detcov, double *cwt,
    double *prob);

double anderson_darling(int npt, int ndim, double **pt, double *delta_mean);

void Gmeans(double **pt, int npt, int naux, double **auxa, int min_pt);

void doGmeans(
    double **points, int npt, int np, int nClstr, int *pt_per_cluster,
    int naux, double **auxa, int min_pt, int maxC);

void gauss_prop(
    int npt, int ndim, double **pt, double *weights, double *norm_weights,
    double cwt, double *mean, double **covmat, double **invcov,
    double *eigen_vals, double **eigen_vecs, double detcov,
    int init_eigen, int calc_mean);

void GaussMixExpMaxLike(
    int ndim, int nclusters, int npt, int *ncon, int norm, double **pt,
    double **like, double **weights, double **norm_weights, double *locZ,
    double *lowlike, int init_eigen);
