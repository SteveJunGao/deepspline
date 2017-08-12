#ifndef BSPLINE_H
#define BSPLINE_H
#include <vector>

// Implementation Reference: The NURBS Book

class BSpline
{
public:
    BSpline(int n_nodes, int order = 3);
    double *eval_p(double u);
    double *tangent_p(double u);
    std::vector<double> eval(double u);
    std::vector<double> tangent(double u);

protected:
    void initialize();

private:
    void Init(int n_nodes, int order = 3);
    void GetKnots(std::vector<double>& knots);
    int FindSpan(double u);
    std::vector<double> BasisFuns(double u, int i);
    int p_; //order: for cubic / quadratic
    int n_nodes_; // number of control points
    int n_; // number of intervals ( #control_points - 1 )
    int m_; // ( #nodes - 1 )
    std::vector<double> U_;
    double a_;  //begin of parametrization
    double b_;  //end of parametrization
    std::vector<double> BasisFuns(double u, int i, int p, double *U);
    int FindSpan(double u, int n, int p, double *U);
    std::vector<double> BasisFunsPrev(double u, int i);
    std::vector<double> BasisFunsNext(double u, int i);
};
#endif // BSPLINE_H
