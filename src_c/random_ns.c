// module RandomNS

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "random_ns.h"

//struct random_ns{
//    int nodes;
//    double **U, *C, *CD, *CM, *gset;
//    int *I97,  *J97, *iset;
//};


void rmarinns(
    struct random_ns *random,
    int ij, int kl, int id){
    // This is the initialization routine for the randomNS number generator
    // ranmarns()
    // NOTE: The seed variables can have values between:  0 <= ij <= 31328
    //                                                    0 <= kl <= 30081
    // The randomNS number sequences created by these two seeds are
    // of sufficient length to complete an entire calculation with.
    // For example, if several different groups are working on
    // different parts of the same calculation, each group could be
    // assigned its own ij seed. This would leave each group with
    // 30000 choices for the second seed. That is to say, this
    // randomNS number generator can create 900 million different
    // subsequences -- with each subsequence having a length of
    // approximately 10^30.
    // Use ij = 1802 & kl = 9373 to test the randomNS number
    // generator. The subroutine ranmarns should be used to generate
    // 20000 randomNS numbers. Then display the next six randomNS
    // numbers generated multiplied by 4096*4096.  If the randomNS
    // number generator is working properly, the randomNS numbers
    // should be:
    //     6533892.0  14220222.0  7275067.0
    //     6172232.0  8354498.0   10633180.0
    int i, j, k, l, m, ii, jj;
    double s, t;

    if (ij < 0)
        ij += 31328;
    if (ij > 31328)
        ij = ij % 31328;

    if (kl < 0)
        kl += 30081;
    if (kl > 30081)
        kl = kl % 30081;

    if( ij < 0 || ij > 31328 || kl < 0 || kl > 30081){
        //printf('First number seed must have value between 0 and 31328'
        //printf('Second seed must have a value between 0 and 30081'
        return;
    }
    i = ((ij/177) % 177) + 2;
    j = (ij % 177) + 2;
    k = ((kl/169) % 178) + 1;
    l = kl % 169;

    for (ii=0; ii<97; ii++){
        s = 0.0;
        t = 0.5;
        for (jj=0; jj<24; jj++){
            m = (((i*j) % 179)*k) % 179;
            i = j;
            j = k;
            k = m;
            l = (53*l+1) % 169;
            if (((l*m) % 64) >= 32)
                s += t;
            t = 0.5 * t;
        }
        random->U[id][ii] = s;
    }
    random->C[id] = 362436.0 / 16777216.0;
    random->CD[id] = 7654321.0 / 16777216.0;
    random->CM[id] = 16777213.0 / 16777216.0;
    random->I97[id] = 96;
    random->J97[id] = 32;
}


double ranmarns(
    struct random_ns *random,
    int id){
    // This is the random number generator proposed by George Marsaglia in
    // Florida State University Report: FSU-SCRI-87-50
    // It was slightly modified by F. James to produce an array of
    // pseudo-random numbers.
    double uni = random->U[id][random->I97[id]]
               - random->U[id][random->J97[id]];

    if (uni < 0.0)
        uni += 1.0;

    random->U[id][random->I97[id]] = uni;
    random->I97[id] -= 1;
    if (random->I97[id] == -1)
        random->I97[id] = 96;

    random->J97[id] -= 1;
    if (random->J97[id] == -1)
        random->J97[id] = 96;

    random->C[id] -= random->CD[id];
    if (random->C[id] < 0.0)
        random->C[id] += random->CM[id];
    uni -= random->C[id];

    if (uni < 0.0)
        uni += 1.0; // bug?

    return uni;
}


void init_random_ns(
    struct random_ns *random,
    int nodes, //no. of nodes
    int seed){ // if >0, use seed, if 0 test case, else seed from clock

    int kl, ij, k;
    int rand_instNS=0;

    // Sanity  check:
    if (nodes <= 0){
        printf("you have asked for %i nodes\n", nodes);
        return;
    }

    random->nodes = nodes;
    random->C = (double *)malloc(nodes * sizeof(double));
    random->CD = (double *)malloc(nodes * sizeof(double));
    random->CM = (double *)malloc(nodes * sizeof(double));

    random->U = (double **)malloc(nodes* sizeof(double *));
    random->U[0] = (double *)malloc(nodes * 97 * sizeof(double));
    for (k=1; k<nodes; k++)
        random->U[k] = random->U[0] + 97*k;

    random->I97 = (int *)malloc(nodes * sizeof(int));
    random->J97 = (int *)malloc(nodes * sizeof(int));
    random->iset = (int *)malloc(nodes * sizeof(int));
    random->gset = (double *)malloc(nodes * sizeof(double));

    for (k=0; k<nodes; k++)
        random->iset[k] = 0;

    for (k=0; k<nodes; k++){
        if (seed > 0){  // input seed exists
            kl = 9373;
            ij = (seed + k) * 45;
        }else if (seed == 0){
            kl = 9373;
            ij = 1802;
        }else{
            /* Get current time:          */
            struct timeval tv;
            gettimeofday(&tv, NULL);
            ij = (int)(1e6*tv.tv_sec + tv.tv_usec);
            ij = ((ij + rand_instNS*100) % (31328)) + k*45;
            kl = (int)(tv.tv_usec) % 30081;

            /* convert to localtime */
            //time_t secs = time(0);
            //struct tm *local = localtime(&secs);
            //hms = (1000*(local->tm_hour)
            //      + 100*(local->tm_min)
            //          + (local->tm_sec));
            //kl = (int)(hms*1000) % 30081;

            //call system_clock(count=ij) // OUT:count = 
            //ij = mod(ij + rand_instNS*100, 31328)+(k-1)*45
            //call date_and_time(time=fred)  // OUT:time = hhmmss.sss
            //read (fred,'(e10.3)') hms
            //kl = (int)(hms*1000) % 30081;
        }

        rmarinns(random, ij, kl, k);
    }

    /* Testing */
    //k=0;
    //double urv = ranmarns(random, k);
    //printf("First random value: %f\n", urv);
    //urv = ranmarns(random, k);
    //printf("Second random value: %f\n", urv);
    //for (k=0; k<19998; k++)
    //    urv = ranmarns(random, 0);
    //for (k=0; k<6; k++){
    //    urv = ranmarns(random, 0);
    //    printf("Random [%i]: %10.1f\n", k, urv*4096*4096);
    //}

    // Use ij = 1802 & kl = 9373 to test the randomNS number
    // generator. The subroutine ranmarns should be used to generate
    // 20000 randomNS numbers. Then display the next six randomNS
    // numbers generated multiplied by 4096*4096.  If the randomNS
    // number generator is working properly, the randomNS numbers
    // should be:
    //     6533892.0
    //    14220222.0
    //     7275067.0
    //     6172232.0
    //     8354498.0
    //    10633180.0    
}


double gaussians1ns(
    struct random_ns *random,
    int id){

    double urv;
    double r, v1, v2, fac;

    if (random->iset[id] == 0){
        r = 2;
        while (r >= 1.0){
            urv = ranmarns(random, id);
            v1 = 2.0*urv - 1.0;
            urv = ranmarns(random, id);
            v2 = 2.0*urv - 1.0;
            r = v1*v1 + v2*v2;
        }
        fac = sqrt(-2.0*log(r)/r);
        random->gset[id] = v1*fac;
        random->iset[id] = 1;
        return v2*fac;
    }else{
        random->iset[id] = 0;
        return random->gset[id];
    }
}


void kill_random_ns(struct random_ns *random){

    free(random->C);
    free(random->CD);
    free(random->CM);
    free(random->U[0]);
    free(random->U);
    free(random->I97);
    free(random->J97);
    free(random->iset);
    free(random->gset);
}

