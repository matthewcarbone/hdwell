#include <iostream>
#include <cmath>
#include <math.h>
#include <cubature/cubature.h>
#include <fstream>

#include "geometry.h"
#include "rootfinder.h"

//const int N_delta = 100;
//const int N_dim = 80;
//const int N_beta = 100;

const int N_MAX_ITER = 1000;

/* Helpers! */
double exponential_prob(const double x, const double lambda)
{
    if (x < 0.0){return 0.0;}
    return exp(-x/lambda)/lambda;
}

double boltzmann_factor(const double x, const double x0, const double dim,
    const double beta, const double beta_critical)
{
    double xp = pow(x/x0, -beta*dim/beta_critical);
    if (xp > 1.0){xp = 1.0;}
    return xp;
}

double upper_limit(const double x0, const double x)
{
    if (x + x0 > 1.0){return x2star(x0, x);}
    else{return x0 + x;}
}

double lower_limit(const double x0, const double x)
{
    if (x < 2.0*x0){return xstar(x0, x);}
    else{return x0 - x;}
}

double rw(const double x, const double x0, const double z)
{
    return sqrt(x*x - x0*(x0 - 2.0*z));
}


/* Upwards probability */
int integrand_prob_up_interior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const double x = p[0];
    const double dim = p[1];
    const double x0 = p[2];
    const double beta = p[3];
    const double beta_critical = p[4];

    const double z = dummies[0];

    const double p_ = (dim - 3.0)/2.0;
    const double t1 = pow(x*x - (x0 - z)*(x0 - z), p_);
    fval[0] = t1*boltzmann_factor(rw(x, x0, z), x0, dim, beta, beta_critical);
    return 0;
}

double prob_up_interior(const double x0, const double dim, const double x,
    const double beta, const double beta_critical)
{
    double xmin[1] = {lower_limit(x0, x)};
    double xmax[1] = {upper_limit(x0, x)};
    double val, err;

    double p[5] = {x, dim, x0, beta, beta_critical};

    hcubature(1, integrand_prob_up_interior, &p, 1, xmin, xmax, N_MAX_ITER, 0,
        1.0e-5, ERROR_INDIVIDUAL, &val, &err);

    return val; //*solid_angle(dim);
}

int integrand_prob_up_exterior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const double dim = p[0];
    const double x0 = p[1];
    const double lambda = p[2];
    const double beta = p[3];
    const double beta_critical = p[4];

    const double x = dummies[0];

    const double t1 = x*exponential_prob(x, lambda)/surface_area(x, dim-1);
    fval[0] = t1*prob_up_interior(x0, dim, x, beta, beta_critical);
    //if (isinf(fval[0]) == true or isnan(fval[0]) == true){fval[0] = 0.0;}
    return 0;
}

double prob_up(const double x0, const double dim, const double lambda,
    const double beta, const double beta_critical)
{
    double xmin[1] = {0.0};
    double xmax[1] = {1.0 + x0};
    double val, err;

    double p[5] = {dim, x0, lambda, beta, beta_critical};

    hcubature(1, integrand_prob_up_exterior, &p, 1, xmin, xmax, N_MAX_ITER, 0,
        1.0e-8, ERROR_INDIVIDUAL, &val, &err);

    return val*solid_angle(dim-1);
}


/* Downwards probability */
int integrand_prob_down_interior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const double x = p[0];
    const double dim = p[1];
    const double x0 = p[2];

    const double z = dummies[0];

    fval[0] = pow(x*x - (x0 - z)*(x0 - z), (dim - 3.0)/2.0);
    return 0;
}

double prob_down_interior(const double x0, const double dim, const double x)
{
    double xmin[1] = {x0 - x};
    double xmax[1] = {xstar(x0, x)};
    double val, err;

    double p[3] = {x, dim, x0};

    hcubature(1, integrand_prob_down_interior, &p, 1, xmin, xmax, N_MAX_ITER, 0,
        1.0e-10, ERROR_INDIVIDUAL, &val, &err);

    return val; //*solid_angle(dim);
}

int integrand_prob_down_exterior(unsigned ndim, const double *dummies,
    void *fdata, unsigned fdim, double *fval)
{
    const double* p = (double *) fdata;
    const double dim = p[0];
    const double x0 = p[1];
    const double lambda = p[2];

    const double x = dummies[0];

    const double pexp = exponential_prob(x, lambda);
    const long double sa = surface_area(x, dim-1);
    const double t1 = x*pexp/sa;
    const double p_down = prob_down_interior(x0, dim, x);
    fval[0] = t1*p_down;
    //if (isinf(fval[0]) == true or isnan(fval[0]) == true){fval[0] = 0.0;}
    return 0;
}

double prob_down(const double x0, const double dim, const double lambda)
{
    double xmin[1] = {0.0};
    double xmax[1] = {2.0*x0};
    double val, err;

    double p[3] = {dim, x0, lambda};

    hcubature(1, integrand_prob_down_exterior, &p, 1, xmin, xmax, N_MAX_ITER, 0,
        1.0e-4, ERROR_INDIVIDUAL, &val, &err);

    return val*solid_angle(dim-1);
}


double minimizer(const double x0, const double dim, const double lambda,
    const double beta, const double beta_critical)
{
    const double pd = prob_down(x0, dim, lambda);
    const double pu = prob_up(x0, dim, lambda, beta, beta_critical);
    return pd - pu;
}


double r_th(const double dim, const double lambda, const double beta,
    const double beta_critical)
{
    std::random_device rd;
    static const double entropy = rd();
    //std::cout << dim << std::endl;
    const double r1 = bisector(minimizer, dim, lambda, beta,
        beta_critical, entropy);
    return r1;
}


/*
int main(int argc, char const *argv[])
{
    const double beta_critical = 1.0;

    const double d_min = 4.0, d_max = 500.0;
    const double d_d = (d_max - d_min)/((double) N_delta);

    const int N_min = 2, N_max = N_min + N_dim;

    const double beta_min = 0.5, beta_max = 5.0;
    const double d_beta = (beta_max - beta_min)/((double) N_beta);

    int N = N_min;
    double d = d_min, beta = beta_min, ans;

    std::ofstream fout("extrapolate.txt");

    for (int nn=0; nn<N_max; nn++)
    {
        N = N_min + nn;

        for (int dd=0; dd<N_delta; dd++)
        {
            d = d_min + dd*d_d;

            for (int bb=0; bb<N_beta; bb++)
            {
                beta = beta_min + bb*d_beta;
                ans = r_th(N, d, beta, beta_critical);
                std::cout << N << "/" << d << "/" << beta << "...r_th="
                    << ans << std::endl << std::flush;
                fout << N << " " << d << " " << beta << " " << ans << '\n';
            }
        }
    }
    
    return 0;
}
*/



int main(int argc, char const *argv[])
{
    const double beta_critical = 1.0;
    //const int N = 20;
    const double d = 1.;
    const double beta = 1.1;

    double ans;

    for (int ii=0; ii<70; ii++)
    {
        ans = r_th(3 + ii, d, beta, beta_critical);
        std::cout << 3+ii << " " << ans << std::endl;
    }
    


    return 0;
}