// This is utils1.f90 in the original multinest code
// Covariance matrix, diagonalization

// use RandomNS

/* DSYEVR prototype */
void dsyevr(
    char *jobz, char *range, char *uplo, int *n, double *a,
    int *lda, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *w, double *z, int *ldz, int *isuppz, double *work,
    int *lwork, int *iwork, int *liwork, int *info);

double sum1d(double *arr, int n);
void swap(double *arr, int i, int j);
void quicksort(double *arr, int n);
double gammln(double xx);
double gcf(double a, double x);
double gser(double a, double x);
double gammp(double a, double x);
double gammq(double a, double x);
double erf(double x);
double st_normal_cdf(double x);
double log_sum_exp(double x, double y);
int bin_search(int *array, int npt, int x);
void diagonalize(double **mat, double *diag, int n, int init_eigen);
void calc_covmat_wt(int npoints, int d, double **p, double *weights,
    double *mean, double **covmat);
void inverse_covariance(int d, double **eigen_vecs, double *eigen_vals,
    double **invcov);

