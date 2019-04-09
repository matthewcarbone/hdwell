#include <iostream>
#include <cmath>
#include <math.h>       /* tgamma */

double solid_angle(const int N)
{
    return N*pow(M_PI, N/2.0)/tgamma(N/2.0 + 1.0);
}

double surface_area(const double r, const int N)
{
    /* Surface area of an n-dimensional SURFACE not volume. e.g. N=2 yields the
       surface area of a 3D sphere: 4pi*r**2.*/

    return 2.0*pow(M_PI, ((N + 1.0)/2.0))*pow(r, N)/tgamma((N + 1.0)/2.0);
}

double xstar(const double x0, const double x)
{
    return (2.0*x0*x0 - x*x)/(2.0*x0);
}

double x2star(const double x0, const double x)
{
    return (1.0 - x*x + x0*x0)/(2.0*x0);
}