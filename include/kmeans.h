// soft k-means clustering with k=2
// D.J.C. MacKay, Information Theory, Inference & Learning Algorithms, 2003, p.304
// Aug 2006


void kmeans3(
    int k, double **pt, int npt, int ndim, double **means, int *cluster,
    int min_pt);

double delF(
    int ndim, int n1, int n2, double mdis1, double mdis2, double detcov1,
    double detcov2, double kfac1, double kfac2);

