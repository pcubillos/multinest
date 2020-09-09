// This is utils1.f90 in the original multinest code
// Covariance matrix, diagonalization

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "ind.h"
#include "random_ns.h"  // will be used when adding pt_in_spheroid()
#include "utils.h"
// use RandomNS

/* DSYEVR prototype */
extern void dsyevr(
    char* jobz, char* range, char* uplo, int* n, double* a,
    int* lda, double* vl, double* vu, int* il, int* iu, double* abstol,
    int* m, double* w, double* z, int* ldz, int* isuppz, double* work,
    int* lwork, int* iwork, int* liwork, int* info);


//int lwork, liwork
//data lwork,liwork /1,1/
int setBlk=0;


double sum1d(double *arr, int n){
    int i;
    double sum = 0.0;
    for (i=0; i<n; i++)
        sum += arr[i];
    return sum;
}


void swap(double *arr, int i, int j){
    double tmp=arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}


void quicksort(double *arr, int n){
    int NSTACK=64, M=7;
    int i, ir, j, k, jstack=-1, l=0;
    int *istack = (int *)malloc(NSTACK * sizeof(int));
    double a;

    ir = n - 1;
    while (1){
        if (ir-l < M){
            for (j=l+1; j<=ir; j++){
                a = arr[j];
                for (i=j-1; i>=l; i--){
                    if (arr[i] <= a)
                        break;
                    arr[i+1] = arr[i];
                }
                arr[i+1] = a;
            }
            if (jstack < 0)
                break;
            ir = istack[jstack--];
            l = istack[jstack--];
        }else{
            k = (l+ir) >> 1;
            swap(arr, k, l+1);

            if (arr[l] > arr[ir])
                swap(arr, l, ir);
            if (arr[l+1] > arr[ir])
                swap(arr, l+1, ir);
            if (arr[l] > arr[l+1])
                swap(arr, l, l+1);
            i = l + 1;
            j = ir;
            a = arr[l+1];
            while (1){
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i)
                    break;
                swap(arr, i, j);
            }
            arr[l+1] = arr[j];
            arr[j] = a;
            jstack += 2;
            if (jstack >= NSTACK){
                printf("STACK too small.\n");
                break;
            }
            if (ir-i+1 >= j-1){
                istack[jstack] = ir;
                istack[jstack-1] = i;
                ir = j - 1;
            }else{
                istack[jstack] = j-1;
                istack[jstack-1] = l;
                l = i;
            }
        }
    }
    free(istack);
}


double gammln(double xx){
    //Returns the value gamma(ln[(xx)]) for xx > 0.
    int j;
    double tmp, x, y;
    //Internal arithmetic will be done in double,
    //a nicety that you can omit if  ve- gure accuracy is good enough.
    double
        stp=2.5066282746310005,
        ser=1.000000000190015;
    double cof[6] = {    // FINDME: double-check DATA statement
         76.18009172947146,
        -86.50532032941677,
         24.01409824083091,
         -1.231739572450155,
          0.1208650973866179e-2,
         -0.5395239384953e-5};

    x = xx;
    y = x;
    tmp = x + 5.5;
    tmp = (x+0.5)*log(tmp) - tmp;
    for (j=0; j<6; j++){
        y += 1.0;
        ser += cof[j] / y;
    }
    return tmp + log(stp*ser/x);
}


double gcf(double a, double x){
    //Returns the incomplete gamma function Q(a, x) evaluated by its continued
    //fraction representation as gammcf.
    int ITMAX=100;      // maximum allowed number of iterations;
    double
        EPS=3.0e-7,     // relative accuracy;
        FPMIN=1.0e-30;  // ~smallest representable floating-point number.
    int i;
    double an, b, c, d, del, h;

    b = x + 1.0 - a;
    c = 1.0/FPMIN;
    d = 1.0/b;
    h = d;
    for (i=0; i<ITMAX; i++){
        an = -i*(i-a);
        b += 2.0;
        d = an*d + b;
        if (fabs(d) < FPMIN)
            d = FPMIN;
        c = b + an/c;
        if (fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0/d;
        del = d*c;
        h *= del;
        if (fabs(del-1.0) < EPS)
            return exp(-x + a*log(x) - gammln(a))*h;  // Put factors in front
    }
    printf("a too large, ITMAX too small in gcf");
    return 0.0;
}


double gser(double a, double x){
    // Returns the incomplete gamma function P(a,x) evaluated by its series
    // representation as gamser.
    int ITMAX=100;
    double EPS=3.0e-7;
    int n;
    double ap, del, sum;

    if (x <= 0.0){
        if (x < 0.0)
            printf("x < 0 in gser");
        return 0.0;
    }
    ap = a;
    sum = 1.0/a;
    del = sum;
    for (n=0; n<ITMAX; n++){
        ap += 1.0;
        del *= x/ap;
        sum += del;
        if (fabs(del) < fabs(sum)*EPS){
            return sum * exp(-x + a*log(x) - gammln(a));
        }
    }
    printf("a too large, ITMAX too small in gser");
    return 0.0;
}


double gammp(double a, double x){
    // Returns the incomplete gamma function P(a, x).
    if (x < 0.0 || a <= 0.0){
        printf("bad arguments in gammp");
        return 0.0;
    }
    // Use the series representation.
    if (x < a+1.0)
        return gser(a, x);

    //Use the continued fraction representation
    return 1.0 - gcf(a, x);  //and take its complement.
}


double gammq(double a, double x){
    // Returns the incomplete gamma function Q(a,x)=1-P(a,x).
    if (x < 0.0 || a <= 0.0){
        printf("bad arguments in gammq");
        return 0.0;
    }
    //Use the series representation
    if (x < a+1.0)
        return 1.0 - gser(a,x); //and take its complement.

    //Use the continued fraction representation.
    return gcf(a, x);
}


double erf(double x){
    // Returns the error function erf(x).
    if (x < 0.0)
        return -gammp(0.5, x*x);
    return gammp(0.5, x*x);
}


double st_normal_cdf(double x){
    if (x > 6.0)
        return 1.0;
    if (x < -6.0)
        return 0.0;
    return 0.5 * (1.0 + erf(x/1.41421356));
}


double log_sum_exp(double x, double y){
    // LogSumExp(x,y) = log(exp(x) + exp(y))
    if (x > y)
        return x + log(1.0 + exp(y-x));
    return y + log(1.0 + exp(x-y));
}


int bin_search(
    int *array, // array in descending order
    int npt,  // number of points in array
    int x){ // value whose position has to be found

    int start=0, end=npt; //starting, end & insertion points

    if (npt == 0)
        return 0;
 
    if (x >= array[0])
        return 0;

    if (x <= array[npt-1]){
        return npt;  // FINDME: out of bounds?
    }

    while (1){
        if (start == end)
            return end;

        if (x > array[(end+start)/2])
            end = (end+start)/2;
        else if (x < array[(end+start)/2])
            start = (end+start)/2 + 1;
        else
            return (end+start)/2;
    }
}


void diagonalize(
    double **mat,   // IN: matrix to diaginalize, OUT: eigenvectors (n,n)
    double *diag, // OUT: array of eigenvalues (diagonal) of the matrix (n)
    int n,  // Size of the matrix
    int init_eigen){  // Initialize eigen-analysis routine
    /* Find the eigenvectors and eigenvalues (diagonal) of a symmetric matrix */

    int lwork=1, liwork=1;

    //int id=0;
    int i, j, ierr, il=1, iu=n, m, lda=n, ldz=n;
    int isuppz[n]; //(2*n)
    double vl, vu,
        abstol=-1.0;
    double *a;
    double z[n*iu]; // (n,n)
    double *work;
    int *iwork;

    double wkopt;
    int iwkopt;

    for (i=0; i<n; i++)
        diag[i] = 0.0;

    // unpack the matrix into a 1D array:
    a = (double *)malloc(n*n * sizeof(double));
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            a[i+j*n] = mat[j][i];

    //$ id=omp_get_thread_num()

    if (setBlk == 0 || init_eigen == 1){
    //    if (id == 0){
            //find optimal block sizes
            lwork = -1;
            liwork = -1;

            //dsyevr(
            //    "Vectors", "Indices", "Upper", &n, a, &lda,
            //    &vl, &vu, &il, &iu, &abstol,
            //    &m,  // Total number of eigenvalues
            //    diag,  // [w] eigenvalues
            //    z,     // eigenvectors
            //    &ldz,  // ldz: leading dimension of z
            //    isuppz,  // support of eigenvectors in z (nonzero indices)
            //    &wkopt, // if info=0, required minimal size of lwork
            //    &lwork,  // dimension of work
            //    &iwkopt,  // workspace array
            //    &liwork,  // dimension of iwork
            //    &ierr);  // info: (0:OK, -i:illegal i-th par, i:internal error)

            lwork = (int)wkopt;
            liwork = iwkopt;

            setBlk = 1;
    //    }else
    //        while (1){
    //            //$OMP FLUSH(setBlk)
    //            if(setBlk == 1)
    //                break;
    //        }
    }

    work = (double *)malloc(lwork * sizeof(double));
    iwork = (int *)malloc(liwork * sizeof(int));

    /* Solve eigenproblem */
    //dsyevr(
    //    "Vectors", "Indices", "Upper", &n, a, &lda, &vl, &vu, &il, &iu,
    //    &abstol, &m, diag, z, &ldz, isuppz, work, &lwork, iwork, &liwork,
    //    &ierr);

    free(a);
    free(work);
    free(iwork);

    /* Check for convergence */
    if(ierr > 0) {
        printf("The algorithm failed to compute eigenvalues.\n");
        exit(1);
    }

    // a = z.T;
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            mat[j][i] = z[i+j*n];

    // Check for inf & nan
    //for (i=0; i<n; i++)
    //    if (diag[i] != diag[i] || diag[i] > huge(1.0)){
    //        diag = 1.0;
    //        a = 0.0;
    //        for (j=0; j<n; j++)
    //            a[j][j] = 1.0;
    //        break;
    //    }
}

void calc_covmat_wt(
    int npoints,  // Number of points
    int d,  // Dimensionality
    double **p,  // Points (d,npoints)
    double *weights,  // Probability weights (npoints)
    double *mean,  // Mean (d)
    double **covmat){  // Covariance matrix (d,d)

    int i, j, k;

    for (i=0; i<d; i++)
       for (j=0; j<d; j++)
           covmat[i][j] = 0.0;

    for (i=0; i<d; i++)
        for (j=i; j<d; j++){
            for (k=0; k<npoints; k++)
                covmat[i][j] +=
                    (p[i][k]-mean[i]) * (p[j][k]-mean[j]) * weights[k];

            if (j != i)
                covmat[j][i] = covmat[i][j];
        }
}


void inverse_covariance( // calc_invcovmat
    int d,  // Matrix dimension
    double **eigen_vecs,  // Eigen vectors (d,d)
    double *eigen_vals,  // Eigen values (d)
    double **invcov){  // OUT: Inverse covariance matrix (d,d)
    //inv_cov=(eigen_vecs).(inv_eigen_vals).Transpose(eigen_vecs)

    int i, j, k;

    for (i=0; i<d; i++)
        for (j=0; j<d; j++)
            invcov[i][j] = 0.0;

    for (i=0; i<d; i++)
        for (j=i; j<d; j++){
            for (k=0; k<d; k++)
                invcov[i][j] += eigen_vecs[i][k]*eigen_vecs[j][k] / eigen_vals[k];
            if (i != j)
                invcov[j][i] = invcov[i][j];
        }

    return;
}

