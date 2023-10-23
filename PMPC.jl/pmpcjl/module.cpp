#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C"
{
#include <stdlib.h>
#include "julia_init.h"

    void c_lqp_solve(double *X_out, double *U_out,
                     size_t xdim, size_t udim, size_t N, size_t M, long long Nc,
                     double *x0, double *f, double *fx, double *fu, double *X_prev,
                     double *U_prev, double *Q, double *R, double *X_ref, double *U_ref,
                     double *lx, double *ux, double *lu, double *uu, double reg_x,
                     double reg_u, double *slew_reg, double *slew_reg0, double *slew_um1,
                     long long verbose);

    void c_lcone_solve(double *X_out, double *U_out,
                       size_t xdim, size_t udim, size_t N, size_t M, long long Nc,
                       double *x0, double *f, double *fx, double *fu, double *X_prev,
                       double *U_prev, double *Q, double *R, double *X_ref, double *U_ref,
                       double *lx, double *ux, double *lu, double *uu, double reg_x,
                       double reg_u, double *slew_reg, double *slew_reg0, double *slew_um1,
                       long long verbose, double smooth_alpha);
}

namespace py = pybind11;

py::tuple lqp_solve(
    size_t xdim, size_t udim, size_t N, size_t M, long long Nc,
    py::array_t<double> x0, py::array_t<double> f, py::array_t<double> fx,
    py::array_t<double> fu, py::array_t<double> X_prev, py::array_t<double> U_prev,
    py::array_t<double> Q, py::array_t<double> R, py::array_t<double> X_ref,
    py::array_t<double> U_ref, py::array_t<double> lx, py::array_t<double> ux,
    py::array_t<double> lu, py::array_t<double> uu, double reg_x, double reg_u,
    py::array_t<double> slew_reg, py::array_t<double> slew_reg0,
    py::array_t<double> slew_um1, long long verbose)
{

    py::array_t<double> X_out(xdim * N * M);
    py::array_t<double> U_out(udim * N * M);

    c_lqp_solve(
        X_out.mutable_data(),     // X_out
        U_out.mutable_data(),     // U_out
        xdim,                     // xdim
        udim,                     // udim
        N,                        // N
        M,                        // M
        Nc,                       // Nc
        x0.mutable_data(),        // x0
        f.mutable_data(),         // f
        fx.mutable_data(),        // fx
        fu.mutable_data(),        // fu
        X_prev.mutable_data(),    // X_prev
        U_prev.mutable_data(),    // U_prev
        Q.mutable_data(),         // Q
        R.mutable_data(),         // R
        X_ref.mutable_data(),     // X_ref
        U_ref.mutable_data(),     // U_ref
        lx.mutable_data(),        // lx
        ux.mutable_data(),        // ux
        lu.mutable_data(),        // lu
        uu.mutable_data(),        // uu
        reg_x,                    // reg_x
        reg_u,                    // reg_u
        slew_reg.mutable_data(),  // slew_reg
        slew_reg0.mutable_data(), // slew_reg0
        slew_um1.mutable_data(),  // slew_um1
        verbose                   // verbose
    );
    return py::make_tuple(X_out, U_out);
}

py::tuple lcone_solve(
    size_t xdim, size_t udim, size_t N, size_t M, long long Nc,
    py::array_t<double> x0, py::array_t<double> f, py::array_t<double> fx,
    py::array_t<double> fu, py::array_t<double> X_prev, py::array_t<double> U_prev,
    py::array_t<double> Q, py::array_t<double> R, py::array_t<double> X_ref,
    py::array_t<double> U_ref, py::array_t<double> lx, py::array_t<double> ux,
    py::array_t<double> lu, py::array_t<double> uu, double reg_x, double reg_u,
    py::array_t<double> slew_reg, py::array_t<double> slew_reg0,
    py::array_t<double> slew_um1, long long verbose, double smooth_alpha)
{

    py::array_t<double> X_out(xdim * N * M);
    py::array_t<double> U_out(udim * N * M);

    c_lcone_solve(
        X_out.mutable_data(),     // X_out
        U_out.mutable_data(),     // U_out
        xdim,                     // xdim
        udim,                     // udim
        N,                        // N
        M,                        // M
        Nc,                       // Nc
        x0.mutable_data(),        // x0
        f.mutable_data(),         // f
        fx.mutable_data(),        // fx
        fu.mutable_data(),        // fu
        X_prev.mutable_data(),    // X_prev
        U_prev.mutable_data(),    // U_prev
        Q.mutable_data(),         // Q
        R.mutable_data(),         // R
        X_ref.mutable_data(),     // X_ref
        U_ref.mutable_data(),     // U_ref
        lx.mutable_data(),        // lx
        ux.mutable_data(),        // ux
        lu.mutable_data(),        // lu
        uu.mutable_data(),        // uu
        reg_x,                    // reg_x
        reg_u,                    // reg_u
        slew_reg.mutable_data(),  // slew_reg
        slew_reg0.mutable_data(), // slew_reg0
        slew_um1.mutable_data(),  // slew_um1
        verbose,                  // verbose
        smooth_alpha              // smooth_alpha
    );
    return py::make_tuple(X_out, U_out);
}

void initialize()  {
  char* argv[] = {"julia"};
  init_julia(1, argv);
}

PYBIND11_MODULE(pmpcjl, m)
{
    initialize();
    m.def("lqp_solve", &lqp_solve, "Julia lqp_solve");
    m.def("lcone_solve", &lcone_solve, "Julia lcone_solve");
}
