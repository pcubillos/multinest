// module RandomNS

struct random_ns{
    int nodes;
    double **U, *C, *CD, *CM, *gset;
    int *I97,  *J97, *iset;
};

void rmarinns(struct random_ns *random, int ij, int kl, int id);
double ranmarns(struct random_ns *random, int id);
void init_random_ns(struct random_ns *random, int nodes, int seed);
double gaussians1ns(struct random_ns *random, int id);
void kill_random_ns(struct random_ns *random);

