#include <iostream>
#include "bspline.h"
#include <iomanip>
using namespace std;

template<class T>
ostream& operator<<(ostream &c, vector<T> v)
{
    for(T e : v)
        c << e << " ";
    c << endl;
    return c;
}
template<class T>
ostream& operator<<(ostream &c, T* v)
{
    int n=5; // hard coded :(
    for(int i=0; i<n; ++i)
        c << setw(5) << v[i] << ' ';
    c << endl;
    return c;
}
// int main(){
//     BSpline spline(10,3);
//     int N=10;
//     for(int i=0; i<=N; ++i)
//         cout << spline.eval((double)i/N);
// }


extern "C" double * eval_point(int n_nodes, int degree, double u){
    BSpline spline(n_nodes, degree);
    double * p = spline.eval_p(u);
    // cout << p;
    return p;
    // return spline.eval_p(u)
}

extern "C" double * tangent_point(int n_nodes, int degree, double u){
    BSpline spline(n_nodes, degree);
    double * p = spline.tangent_p(u);
    cout << p;
    return p;
}

extern "C" double get_value_in_matrix(double * p, int idx){
    return p[idx];
}
