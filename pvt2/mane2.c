#define _POSIX_C_SOURCE 1
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define M_PI 3.14159265358979323846
double getrandx(unsigned int *seed)
{
    double result;
    do
    {
        result = (double)rand_r(seed) / RAND_MAX;
    } while (result < 0 || result > 1);

    return result;
}
double getrandy(unsigned int *seed, double x)
{
    double result;
    do
    {
        result = (double)rand_r(seed) / RAND_MAX * (1-x);
    } while (result < 0 || result > 1-x);
    return result;
}

double getrand(){ 
    return (double)rand() / RAND_MAX; 
}
double func(double x, double y)
{
    return exp(pow((x+y),2));
}
int main(int argc, char **argv)
{
    for(int l = 2; l <= 8; l+=2)
    {
    const int n = 100000000;
    printf("Numerical integration by Monte Carlo method: n = %d\n", n);
    int in = 0;
    double s = 0;
    omp_set_num_threads(l);
#pragma omp parallel
    {
        double s_loc = 0;
        int in_loc = 0;
        unsigned int seed =  omp_get_thread_num();
#pragma omp for nowait
        for (int i = 0; i < n; i++)
        {
            double x = getrandx(&seed); /* x in [0, 1] */
            double y = getrandy(&seed, x);        /* y in [2, 5] */
            if (y <= (1-x))
            {
                in_loc++;
                s_loc += func(x, y);
            }
        }
#pragma omp atomic
        s += s_loc;
#pragma omp atomic
        in += in_loc;
    }
    double v = M_PI * in / n;
    double res = v * s / in;
    printf("Result: %.12f, n %d\n", res, n);
    }
    return 0;
}