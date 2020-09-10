// C multinest

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "ind.h"
#include "random_ns.h"
#include "utils.h"
#include "kmeans.h"

struct random_ns rand_ns;


PyDoc_STRVAR(test__doc__, "import multinest");

//static PyObject *test_kmeans_anderson(PyObject *self, PyObject *args){
//    PyArrayObject *points, *clusters, *kmean;
//    double **pt, **means;
//    int i, j, k, ndim, npt;
//    int min_pt;
//    int anderson;
//    int *cluster;
//    //static struct random_ns rand_ns;
//    npy_intp
//        mean_size[2],
//        cluster_size[1];
//
//    // Load inputs:
//    if (!PyArg_ParseTuple(args, "Oi", &points, &min_pt))
//        return NULL;
//
//    k = 2;
//    n_dim = ndim = (int)PyArray_DIM(points, 0);
//    npt = (int)PyArray_DIM(points, 1);
//    mean_size[0] = k;
//    mean_size[1] = ndim;
//    kmean = (PyArrayObject *) PyArray_SimpleNew(2, mean_size, NPY_DOUBLE);
//    cluster_size[0] = npt;
//    clusters = (PyArrayObject *) PyArray_SimpleNew(1, cluster_size, NPY_INT);
//
//    pt = (double **)malloc(ndim * sizeof(double *));
//    pt[0] = (double *)malloc(ndim*npt * sizeof(double));
//    for (i=1; i<ndim; i++)
//        pt[i] = pt[0] + npt*i;
//
//    means = (double **)malloc(k * sizeof(double *));
//    means[0] = (double *)malloc(k*ndim * sizeof(double));
//    for (i=1; i<k; i++)
//        means[i] = means[0] + ndim*i;
//
//    cluster = (int *)malloc(npt * sizeof(int));
//
//    for (i=0; i<ndim; i++)
//        for (j=0; j<npt; j++)
//            pt[i][j] = IND2d(points,i,j);
//
//    init_random_ns(&rand_ns, 1, -2);
//    printf("FLAG 0: RANDOM\n");
//
//    kmeans3(k, pt, npt, ndim, means, cluster, min_pt);
//    printf("FLAG 1: KMEANS\n");
//
//    for (i=0; i<ndim; i++)
//        means[0][i] = means[0][i] - means[1][i];
//    anderson = anderson_darling(npt, n_dim, pt, means[0], 0.0001);
//
//    for (j=0; j<npt; j++)
//        INDi(clusters,j) = cluster[j];
//    for (i=0; i<k; i++)
//        for (j=0; j<ndim; j++)
//            IND2d(kmean,i,j) = means[i][j];
//
//    free(means[0]);
//    free(means);
//    free(pt[0]);
//    free(pt);
//    free(cluster);
//
//    return Py_BuildValue("[N,N]", clusters, kmean);
//}

static PyObject *test_diagonalize(PyObject *self, PyObject *args){
    PyArrayObject *matrix, *diagonal;
    double **mat, *diag;
    int i, j, n;
    npy_intp size[1];

    // Load inputs:
    if (!PyArg_ParseTuple(args, "O", &matrix))
        return NULL;

    size[0] = n = (int)PyArray_DIM(matrix, 0);
    diagonal = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

    diag = (double *)malloc(n * sizeof(double));
    mat = (double **)malloc(n * sizeof(double *));
    mat[0] = (double *)malloc(n*n * sizeof(double));
    for (i=1; i<n; i++)
        mat[i] = mat[0] + n*i;

    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            mat[i][j] = IND2d(matrix,i,j);

    diagonalize(mat, diag, n, 1);

    for (i=0; i<n; i++){
        INDd(diagonal,i) = diag[i];
        for (j=0; j<n; j++)
            IND2d(matrix,i,j) = mat[i][j];
    }
    free(mat[0]);
    free(mat);
    free(diag);

    return Py_BuildValue("N", diagonal);
}


static PyObject *test_kmeans(PyObject *self, PyObject *args){
    PyArrayObject *points, *clusters, *kmean;
    double **pt, **means;
    int i, j, k, ndim, npt, min_pt;
    int *cluster;
    npy_intp
        mean_size[2],
        cluster_size[1];

    // Load inputs:
    if (!PyArg_ParseTuple(args, "Oi", &points, &min_pt))
        return NULL;

    k = 2;
    ndim = (int)PyArray_DIM(points, 0);
    npt = (int)PyArray_DIM(points, 1);
    mean_size[0] = k;
    mean_size[1] = ndim;
    kmean = (PyArrayObject *) PyArray_SimpleNew(2, mean_size, NPY_DOUBLE);
    cluster_size[0] = npt;
    clusters = (PyArrayObject *) PyArray_SimpleNew(1, cluster_size, NPY_INT);

    pt = (double **)malloc(ndim * sizeof(double *));
    pt[0] = (double *)malloc(ndim*npt * sizeof(double));
    for (i=1; i<ndim; i++)
        pt[i] = pt[0] + npt*i;

    means = (double **)malloc(k * sizeof(double *));
    means[0] = (double *)malloc(k*ndim * sizeof(double));
    for (i=1; i<k; i++)
        means[i] = means[0] + ndim*i;

    cluster = (int *)malloc(npt * sizeof(int));

    for (i=0; i<ndim; i++)
        for (j=0; j<npt; j++)
            pt[i][j] = IND2d(points,i,j);

    init_random_ns(&rand_ns, 1, -2);

    kmeans3(k, pt, npt, ndim, means, cluster, min_pt);
    for (j=0; j<npt; j++)
        INDi(clusters,j) = cluster[j];
    for (i=0; i<k; i++)
        for (j=0; j<ndim; j++)
            IND2d(kmean,i,j) = means[i][j];
    kill_random_ns(&rand_ns);

    free(means[0]);
    free(means);
    free(pt[0]);
    free(pt);
    free(cluster);

    return Py_BuildValue("[N,N]", clusters, kmean);
}


static PyObject *test_random(PyObject *self, PyObject *args){
    PyArrayObject *rands;
    int i, seed, nrandom;
    static struct random_ns rand;
    npy_intp size[1];

    // Load inputs:
    if (!PyArg_ParseTuple(args, "ii", &nrandom, &seed))
        return NULL;

    init_random_ns(&rand, 1, seed);

    size[0] = nrandom;
    rands = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
    for (i=0; i<nrandom; i++){
        INDd(rands,i) = ranmarns(&rand, 0);
    }
    kill_random_ns(&rand);

    return Py_BuildValue("N", rands);
}


static PyObject *test_quicksort(PyObject *self, PyObject *args){
    PyArrayObject *array;
    double *arr;
    int i, size;

    // Load inputs:
    if (!PyArg_ParseTuple(args, "O", &array))
        return NULL;

    size = (int)PyArray_DIM(array, 0);
    arr = (double *)calloc(size, sizeof(double));
    for (i=0; i<size; i++)
        arr[i] = INDd(array,i);

    quicksort(arr, size);

    for (i=0; i<size; i++)
        INDd(array,i) = arr[i];

    free(arr);
    return Py_BuildValue("");
}


/* The module doc string */
PyDoc_STRVAR(multinest__doc__, "utility functions for C multinest.");

/* A list of all the methods defined by this module. */
static PyMethodDef multinest_methods[] = {
    {"test_diagonalize", test_diagonalize, METH_VARARGS, test__doc__},
    {"test_random", test_random, METH_VARARGS, test__doc__},
    {"test_quicksort", test_quicksort, METH_VARARGS, test__doc__},
    {"test_kmeans", test_kmeans, METH_VARARGS, test__doc__},
    {NULL, NULL, 0, NULL}    /* sentinel */
};

/* Module definition for Python 3. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "multinest",
    multinest__doc__,
    -1,
    multinest_methods
};

/* When Python 3 imports a C module named 'X' it loads the module */
/* then looks for a method named "PyInit_"+X and calls it.        */
PyObject *PyInit_multinest (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}

//PyMODINIT_FUNC PyInit_multinest(void) {
//    import_array();
//    return PyModule_Create(&moduledef);
//}

