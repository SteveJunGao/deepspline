#include "bspline.h"
using namespace std;
namespace {
    double *to_array(const vector<double>& input)
    {
        double *output = new double[input.size()];
        copy(input.begin(), input.end(), output);
        return output;
    }
};

BSpline::BSpline(int n_nodes, int _order)
{
    Init(n_nodes, _order);
}

void BSpline::Init(int n_nodes, int _order)
{
    n_nodes_ = n_nodes;
    p_ = _order;//degree
    initialize();
}

int BSpline::FindSpan(double u)
{
    // n_ is the span index
    return FindSpan(u, n_, p_, &U_[0]);
}

// find the index id of knot which is right on the left of u
// that is, u lie in the knot inteval [U[id], U[id+1]]
// p  is degree,  n+ 1: # control points
// U is input , knot vector
int BSpline::FindSpan(double u, int n, int p, double *U)
{
    if (u == U[n+1]) return n;
    int low = p;
    int high = n+1;
    int mid = (low+high)/2;
    while (u < U[mid] || u >= U[mid+1])
    {
        if (u<U[mid]) high = mid;
        else low = mid;
        mid = (low+high)/2;
    }
    return mid;
}

// B spline basis function  which are nonzero at u
// i is given  by FindSpan
vector<double> BSpline::BasisFuns(double u, int i)
{
    return BasisFuns(u, i, p_, &U_[0]);
}


// B spline basis function  which are nonzero at u
vector<double> BSpline::BasisFuns(double u, int i, int p, double *U)
{
    vector<double> N(p+1, 0.), left(p+1, 0.), right(p+1, 0.);
    N[0] = 1.0;
    for(int j=1; j<=p; ++j)
    {
        left[j] = u-U[i+1-j];
        right[j] = U[i+j]-u;
        double saved = 0.0;
        for(int r=0; r<j; ++r)
        {
            double temp = N[r]/(right[r+1]+left[j-r]);
            N[r] = saved + right[r+1]*temp;
            saved = left[j-r]*temp;
        }
        N[j] = saved;
    }
    return N;
}


void BSpline::GetKnots(vector<double>& knots)
{
	knots = U_;
}

void BSpline::initialize()
{
    a_ = 0.;
    b_ = 1.;
    n_ = n_nodes_ - 1; // number of intervals instead of points
    m_ = p_ + n_ + 1;
    // set knot vector U_
    for (int i=0; i<p_+1; ++i) U_.push_back(a_);
    double d = (b_ - a_) / (m_ - 2*p_);
    for (int i=1.; i<(m_ - 2*p_); ++i) U_.push_back(a_ + d*i);
    for (int i=0; i<p_+1; ++i) U_.push_back(b_);
}

vector<double> BSpline::eval(double u) {
    vector<double> res(n_nodes_, 0);
    int span = FindSpan(u);
    vector<double> N = BasisFuns(u, span);
    for(int i=0; i<p_+1; ++i)
        res[span-p_+i] = N[i];
    return res;
}

vector<double> BSpline::BasisFunsPrev(double u, int i)
{
    vector<double> Uprev;
    Uprev.insert(Uprev.end(), U_.begin()+i+1-p_, U_.begin()+i+p_-1+1);
    Uprev.erase(Uprev.begin()+p_-1);
    return BasisFuns(u, p_-2, p_-1, &Uprev[0]);
}

vector<double> BSpline::BasisFunsNext(double u, int i)
{
    vector<double> Unext;
    Unext.insert(Unext.end(), U_.begin()+i+1-p_+1, U_.begin()+i+p_+1);
    Unext.erase(Unext.begin()+p_-1);
    return BasisFuns(u, p_-2, p_-1, &Unext[0]);
}

vector<double> BSpline::tangent(double u) {
    int span = FindSpan(u);
    vector<double> prev = BasisFunsPrev(u, span);
    vector<double> next = BasisFunsNext(u, span);
    vector<double> res(n_nodes_, 0);
    for(int i=0; i<p_; ++i)
    {
        res[span-p_+i] += prev[i];
        res[span-p_+i+1] -= next[i];
    }
    return res;
}

double* BSpline::eval_p(double u)
{
    return to_array(eval(u));
}

double* BSpline::tangent_p(double u)
{
    return to_array(tangent(u));
}
