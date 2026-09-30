#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <functional>
#include <optional>
#include <vector>

#include "CDStratifiedSolution/CDStratifiedSolution.h"
#include "CDGraetzIsothermalSolution/CDGraetzIsothermalSolution.h"
#include "diagnostics/norm_residuals.h"
#include "diagnostics/layer_integral_diagnostic.h"
#include "diagnostics/fp_parseval_diagnostics.h"
#include "finite_peclet_roots.h"
#include "finite_peclet_radial.h"
#include "finite_peclet_norms.h"
#include "gram_method.h"
#include "rhs_method.h"
#include "projection_space.h"
#include "math_functions.h"
#include "comp_utils.h"
#include "nusselt_postprocessing.h"


namespace py = pybind11;

template <typename T>
void bind_class(py::module_ &m, const std::string &name) {
    py::class_<CDStratifiedSolution<T>>(m, name.c_str())
        .def(py::init<const std::vector<T>&, const std::vector<T>&, unsigned short>())
        .def(py::init<const std::vector<T>&, const std::vector<T>&>())
        .def("get_zi", &CDStratifiedSolution<T>::get_zi)
        .def("get_ui", &CDStratifiedSolution<T>::get_ui)
        .def("get_tol", &CDStratifiedSolution<T>::get_tol, py::arg("x_value"))
        .def("set_max_root", &CDStratifiedSolution<T>::set_max_root, py::arg("root_value"))
        .def("cutoff_modes", &CDStratifiedSolution<T>::cutoff_modes, py::arg("tol"),
             "Skip, when evaluating the series, modes whose amplitude |C_k exp(-Lam_k x)| * max_u |R_k(r_u)|\n"
             "is below tol times the largest amplitude of that plane. Evaluation-time only: neither the\n"
             "mode set, nor max_root, nor the coefficients change, so it can be toggled between calls.\n"
             "tol = 0 (default) evaluates every mode.")
        .def("get_cutoff_modes", &CDStratifiedSolution<T>::get_cutoff_modes)
        .def("get_max_root", &CDStratifiedSolution<T>::get_max_root)
        .def("get_number_of_gauss_points", &CDStratifiedSolution<T>::get_number_of_gauss_points)
        .def("set_number_of_gauss_points", &CDStratifiedSolution<T>::set_number_of_gauss_points)
        .def("get_coefficients", &CDStratifiedSolution<T>::get_coefficients)
        .def("get_bare_root_catalog", &CDStratifiedSolution<T>::get_bare_root_catalog)
        .def("get_norms", &CDStratifiedSolution<T>::get_norms)
        .def("get_roots", &CDStratifiedSolution<T>::get_roots)
        .def("get_solution", py::overload_cast<const std::array<std::vector<T>, 3> &>(&CDStratifiedSolution<T>::get_solution))
        .def("get_solution", py::overload_cast<const std::vector<T> &, const std::vector<T> &, const std::vector<T> &>(&CDStratifiedSolution<T>::get_solution))
        .def("get_solution_cloud_of_points", &CDStratifiedSolution<T>::get_solution_cloud_of_points)
        .def("get_solution_at_points", py::overload_cast<T, const std::vector<T>&, const std::vector<T>&>(&CDStratifiedSolution<T>::get_solution_at_points))
        .def("get_solution_at_planes", py::overload_cast<const std::vector<T>&, const std::vector<T>&, const std::vector<T>&>(&CDStratifiedSolution<T>::get_solution_at_planes))
        .def("get_max_K",                   &CDStratifiedSolution<T>::get_max_K)
        .def("get_number_of_coefficients", &CDStratifiedSolution<T>::get_number_of_coefficients)
        .def("setup_solution", py::overload_cast<T, T>(&CDStratifiedSolution<T>::setup_solution),
             py::arg("peclet"), py::arg("max_root"))
        .def("setup_solution", py::overload_cast<T>(&CDStratifiedSolution<T>::setup_solution),
             py::arg("peclet"))
        .def("setup_bare_solution", &CDStratifiedSolution<T>::setup_bare_solution)
        .def("setup_qep_solution", &CDStratifiedSolution<T>::setup_qep_solution, py::arg("peclet"))
        .def("setup_fp_solution", &CDStratifiedSolution<T>::setup_fp_solution, py::arg("peclet"))
        .def("get_solution_method", &CDStratifiedSolution<T>::get_solution_method)
        .def("get_QEP_solution_at_points", &CDStratifiedSolution<T>::get_QEP_solution_at_points,
             py::arg("peclet"), py::arg("x_points"), py::arg("r_points"), py::arg("phi_points"),
             "Exact finite-Peclet solution via the Quadratic Eigenvalue Problem in the bare "
             "eigenbasis (Neuhauser et al. 2025, Sec. 2.4). Independent reference for the "
             "modified-basis solution built by setup_solution().")
        .def("using_axial_diffusion", &CDStratifiedSolution<T>::using_axial_diffusion)
        .def("get_peclet", &CDStratifiedSolution<T>::get_peclet)
        .def("get_modified_data", &CDStratifiedSolution<T>::get_modified_data)
        .def("get_nusselt_number",
             py::overload_cast<const std::vector<T> &>(&CDStratifiedSolution<T>::get_nusselt_number, py::const_),
             py::arg("x_points"))
        .def("get_nusselt_number",
             py::overload_cast<T>(&CDStratifiedSolution<T>::get_nusselt_number, py::const_),
             py::arg("x"))
        .def("get_nusselt_number",
             [](const CDStratifiedSolution<T> &self, const std::vector<T> &x_points, NusseltEvaluation evaluation,
                const std::optional<std::vector<T>> &extra_rates) {
                 return self.get_nusselt_number(x_points, evaluation, extra_rates ? &*extra_rates : nullptr);
             },
             py::arg("x_points"), py::arg("evaluation"), py::arg("extra_rates") = py::none(),
             "Local Nusselt number with the Plain, SpectralSplit or SlugSplit evaluation; "
             "extra_rates (SpectralSplit only) reuses get_nusselt_tail_rates(x_min).")
        .def("get_nusselt_number",
             py::overload_cast<T, NusseltEvaluation>(&CDStratifiedSolution<T>::get_nusselt_number, py::const_),
             py::arg("x"), py::arg("evaluation"))
        .def("get_nusselt_tail_rates", &CDStratifiedSolution<T>::get_nusselt_tail_rates, py::arg("x_min"),
             "Finite-Peclet axisymmetric rates beyond the retained block, sufficient for x >= x_min.")
        .def("nusselt_converged_x_min", &CDStratifiedSolution<T>::nusselt_converged_x_min, py::arg("tol"),
             "ln(1/tol)/Lambda_max: exponential-truncation bound of the Plain Nusselt sum.")
        .def("get_fully_developed_nusselt_number", &CDStratifiedSolution<T>::get_fully_developed_nusselt_number)
        .def("get_nusselt_mode_data", &CDStratifiedSolution<T>::get_nusselt_mode_data)
        .def("get_blurriness", py::overload_cast<T, BlurrinessTail>(&CDStratifiedSolution<T>::get_blurriness, py::const_),
             py::arg("x"), py::arg("tail") = BlurrinessTail::Asymptotic,
             "Blurriness B(x) with the given treatment of the omitted modes (BlurrinessTail.Dropped, "
             ".Constant or .Asymptotic; see the enum docstring).")
        .def("get_blurriness", py::overload_cast<const std::vector<T> &, BlurrinessTail>(&CDStratifiedSolution<T>::get_blurriness, py::const_),
             py::arg("x_points"), py::arg("tail") = BlurrinessTail::Asymptotic, "Vectorised get_blurriness with a BlurrinessTail.")
        .def("get_blurriness", py::overload_cast<T, bool>(&CDStratifiedSolution<T>::get_blurriness, py::const_),
             py::arg("x"), py::arg("include_truncation_tail"),
             "Blurriness B(x) of the section at the dimensionless axial coordinate x = x_phys / (Pe R) (bool tail form; the BlurrinessTail form, default Asymptotic, is preferred):\n"
             "  B(x)^2 = || psi(x) - f ||^2_omega / || psi_inf - f ||^2_omega ,\n"
             "evaluated from the series coefficients through the L2_omega Gram matrix of the retained "
             "modes (the Parseval identity, eq. gram_parseval): numerator a^T W a with a_k = c_k (e^{-Lam_k x} - 1), "
             "denominator the same form at x = inf. With include_truncation_tail (default) the Parseval "
             "defect D_K = ||f||^2 - c^T W c is added to both, which is the exact blurriness of the truncated "
             "field for x >> 1/Lam_K and equals a quadrature of the series over the disk; it requires the "
             "L2_omega projection (bare or Weighted). Without it the ratio is 0 at x = 0 and 1 at x = inf "
             "in every projection space. Requires an active bare or finite-Peclet setup (not QEP).")
        .def("get_blurriness", py::overload_cast<const std::vector<T> &, bool>(&CDStratifiedSolution<T>::get_blurriness, py::const_),
             py::arg("x_points"), py::arg("include_truncation_tail"),
             "Vectorised get_blurriness.")
        .def("get_blurriness_truncation_floor", &CDStratifiedSolution<T>::get_blurriness_truncation_floor,
             "B_K(0) = sqrt(D_K / ||psi_inf - f||^2): the resolution floor of the truncated series at the inlet.")
        .def("get_far_field_value", &CDStratifiedSolution<T>::get_far_field_value,
             "Far-field constant psi_inf in the user's units (zero-eigenvalue mode; wall value for a Dirichlet basis).")
        .def("get_inlet_square_norm", &CDStratifiedSolution<T>::get_inlet_square_norm,
             "||f||^2_omega of the user's inlet, exact (no truncation).")
        .def("get_inlet_far_field_square_norm", &CDStratifiedSolution<T>::get_inlet_far_field_square_norm,
             "||f - psi_inf||^2_omega of the user's inlet: the denominator of the blurriness.")
        .def("get_inlet_projection_square_norm", &CDStratifiedSolution<T>::get_inlet_projection_square_norm,
            "Squared norm of the user-scaled inlet projection. Bare and QEP setups return "
            "the L2_omega norm; finite-Peclet setups return the norm of their selected "
            "projection space (L2_omega or L2_r). The stored Galerkin contraction is "
            "c^T b in that same space. Requires an active solution.")
        .def("set_fp_quad_panels", &CDStratifiedSolution<T>::set_fp_quad_panels)
        .def("set_fp_cap_quad_margin", &CDStratifiedSolution<T>::set_fp_cap_quad_margin, py::arg("margin"))
        .def("get_fp_cap_quad_margin", &CDStratifiedSolution<T>::get_fp_cap_quad_margin)
        .def("set_nusselt_asymptotic_terms", &CDStratifiedSolution<T>::set_nusselt_asymptotic_terms, py::arg("terms"))
        .def("get_nusselt_asymptotic_terms", &CDStratifiedSolution<T>::get_nusselt_asymptotic_terms)
        .def("set_gram_method", &CDStratifiedSolution<T>::set_gram_method, py::arg("method"))
        .def("get_gram_method", &CDStratifiedSolution<T>::get_gram_method)
        .def("set_gram_oversampling", &CDStratifiedSolution<T>::set_gram_oversampling, py::arg("factor"), py::arg("margin"))
        .def("set_gram_order_check", &CDStratifiedSolution<T>::set_gram_order_check, py::arg("enabled"))
        .def("set_gram_minimum_factor", &CDStratifiedSolution<T>::set_gram_minimum_factor, py::arg("factor"))
        .def("get_gram_minimum_factor", &CDStratifiedSolution<T>::get_gram_minimum_factor)
        .def("set_rhs_method", &CDStratifiedSolution<T>::set_rhs_method, py::arg("method"))
        .def("get_rhs_method", &CDStratifiedSolution<T>::get_rhs_method)
        .def("set_rhs_representer_margin", &CDStratifiedSolution<T>::set_rhs_representer_margin, py::arg("margin"))
        .def("get_rhs_representer_margin", &CDStratifiedSolution<T>::get_rhs_representer_margin)
        .def("set_projection_space", py::overload_cast<ProjectionSpace>(&CDStratifiedSolution<T>::set_projection_space),
             py::arg("space"),
             "Inner product used for the finite-Peclet inlet projection: ProjectionSpace.Weighted "
             "(L2_omega, weight (1-r^2) r dr -- the default) or ProjectionSpace.Radial (L2_r, weight "
             "r dr). Radial requires RhsMethod.DirectQuadrature and GramMethod.GaussJacobiQR; other "
             "combinations are rejected by setup_solution rather than silently falling back.")
        .def("set_projection_space", py::overload_cast<const std::string &>(&CDStratifiedSolution<T>::set_projection_space),
             py::arg("space"),
             "String selector for set_projection_space: \"w\"/\"omega\"/\"weighted\" or \"r\"/\"radial\" "
             "(case-insensitive). Unknown strings raise ValueError.")
        .def("get_projection_space", &CDStratifiedSolution<T>::get_projection_space)
        .def("set_fp_tolerance", &CDStratifiedSolution<T>::set_fp_tolerance,
             py::arg("tolerance"),
             "Set the finite-Peclet root relative tolerance and the requested absolute "
             "tolerance for the wall-centered positive-cap series; the implementation "
             "also applies fixed-precision stability checks.")
        .def("set_fp_max_iter", &CDStratifiedSolution<T>::set_fp_max_iter,
             py::arg("max_iterations"),
             "Set both the maximum root-solver iterations and the maximum number of "
             "wall-centered positive-cap series summands.")
        .def("compute_and_fix_coefficients", &CDStratifiedSolution<T>::compute_and_fix_coefficients)
        .def("fix_coefficients", &CDStratifiedSolution<T>::fix_coefficients)
        .def("free_coefficients", &CDStratifiedSolution<T>::free_coefficients)
        .def("save_coefficients", &CDStratifiedSolution<T>::save_coefficients)
        .def("print_info", &CDStratifiedSolution<T>::print_info)
        .def("set_max_warnings", &CDStratifiedSolution<T>::set_max_warnings)
        .def("set_verbose_level", &CDStratifiedSolution<T>::set_verbose_level)
        .def("get_verbose_level", &CDStratifiedSolution<T>::get_verbose_level)
        ;
}

template <typename T>
void bind_graetz_class(py::module_ &m, const std::string &name) {
    py::class_<CDGraetzIsothermalSolution<T>>(m, name.c_str())
        .def(py::init<T, T, unsigned short>())
        .def(py::init([](T T0, T T_wall) { return new CDGraetzIsothermalSolution<T>(T0, T_wall, (unsigned short)1); }))
        .def("get_tol", &CDGraetzIsothermalSolution<T>::get_tol, py::arg("x_value"))
        .def("set_max_root", &CDGraetzIsothermalSolution<T>::set_max_root, py::arg("root_value"))
        .def("cutoff_modes", &CDGraetzIsothermalSolution<T>::cutoff_modes, py::arg("tol"),
             "Skip, when evaluating the series, modes whose amplitude |C_k exp(-Lam_k x)| * max_u |R_k(r_u)|\n"
             "is below tol times the largest amplitude of that plane. Evaluation-time only: neither the\n"
             "mode set, nor max_root, nor the coefficients change, so it can be toggled between calls.\n"
             "tol = 0 (default) evaluates every mode.")
        .def("get_cutoff_modes", &CDGraetzIsothermalSolution<T>::get_cutoff_modes)
        .def("get_max_root", &CDGraetzIsothermalSolution<T>::get_max_root)
        .def("get_number_of_gauss_points", &CDGraetzIsothermalSolution<T>::get_number_of_gauss_points)
        .def("set_number_of_gauss_points", &CDGraetzIsothermalSolution<T>::set_number_of_gauss_points)
        .def("get_coefficients", &CDGraetzIsothermalSolution<T>::get_coefficients)
        .def("get_bare_root_catalog", &CDGraetzIsothermalSolution<T>::get_bare_root_catalog)
        .def("get_norms", &CDGraetzIsothermalSolution<T>::get_norms)
        .def("get_roots", &CDGraetzIsothermalSolution<T>::get_roots)
        .def("get_solution", py::overload_cast<const std::array<std::vector<T>, 3> &>(&CDGraetzIsothermalSolution<T>::get_solution))
        .def("get_solution", py::overload_cast<const std::vector<T> &, const std::vector<T> &, const std::vector<T> &>(&CDGraetzIsothermalSolution<T>::get_solution))
        .def("get_solution_cloud_of_points", &CDGraetzIsothermalSolution<T>::get_solution_cloud_of_points)
        .def("get_solution_at_points", py::overload_cast<T, const std::vector<T>&, const std::vector<T>&>(&CDGraetzIsothermalSolution<T>::get_solution_at_points))
        .def("get_solution_at_planes", py::overload_cast<const std::vector<T>&, const std::vector<T>&, const std::vector<T>&>(&CDGraetzIsothermalSolution<T>::get_solution_at_planes))
        .def("get_max_K",                   &CDGraetzIsothermalSolution<T>::get_max_K)
        .def("get_number_of_coefficients", &CDGraetzIsothermalSolution<T>::get_number_of_coefficients)
        .def("setup_solution", py::overload_cast<T, T>(&CDGraetzIsothermalSolution<T>::setup_solution),
             py::arg("peclet"), py::arg("max_root"))
        .def("setup_solution", py::overload_cast<T>(&CDGraetzIsothermalSolution<T>::setup_solution),
             py::arg("peclet"))
        .def("setup_bare_solution", &CDGraetzIsothermalSolution<T>::setup_bare_solution)
        .def("setup_qep_solution", &CDGraetzIsothermalSolution<T>::setup_qep_solution, py::arg("peclet"))
        .def("setup_fp_solution", &CDGraetzIsothermalSolution<T>::setup_fp_solution, py::arg("peclet"))
        .def("get_solution_method", &CDGraetzIsothermalSolution<T>::get_solution_method)
        .def("get_QEP_solution_at_points", &CDGraetzIsothermalSolution<T>::get_QEP_solution_at_points,
             py::arg("peclet"), py::arg("x_points"), py::arg("r_points"), py::arg("phi_points"),
             "Exact finite-Peclet solution via the Quadratic Eigenvalue Problem in the bare "
             "eigenbasis (Neuhauser et al. 2025, Sec. 2.4). Independent reference for the "
             "modified-basis solution built by setup_solution().")
        .def("using_axial_diffusion", &CDGraetzIsothermalSolution<T>::using_axial_diffusion)
        .def("get_peclet", &CDGraetzIsothermalSolution<T>::get_peclet)
        .def("get_modified_data", &CDGraetzIsothermalSolution<T>::get_modified_data)
        .def("get_nusselt_number",
             py::overload_cast<const std::vector<T> &>(&CDGraetzIsothermalSolution<T>::get_nusselt_number, py::const_),
             py::arg("x_points"), "Diameter-based local Nusselt number for an isothermal wall.")
        .def("get_nusselt_number",
             py::overload_cast<T>(&CDGraetzIsothermalSolution<T>::get_nusselt_number, py::const_),
             py::arg("x"))
        .def("get_nusselt_number",
             [](const CDGraetzIsothermalSolution<T> &self, const std::vector<T> &x_points, NusseltEvaluation evaluation,
                const std::optional<std::vector<T>> &extra_rates) {
                 return self.get_nusselt_number(x_points, evaluation, extra_rates ? &*extra_rates : nullptr);
             },
             py::arg("x_points"), py::arg("evaluation"), py::arg("extra_rates") = py::none(),
             "Local Nusselt number with the Plain, SpectralSplit or SlugSplit evaluation; "
             "extra_rates (SpectralSplit only) reuses get_nusselt_tail_rates(x_min).")
        .def("get_nusselt_number",
             py::overload_cast<T, NusseltEvaluation>(&CDGraetzIsothermalSolution<T>::get_nusselt_number, py::const_),
             py::arg("x"), py::arg("evaluation"))
        .def("get_nusselt_tail_rates", &CDGraetzIsothermalSolution<T>::get_nusselt_tail_rates, py::arg("x_min"),
             "Finite-Peclet axisymmetric rates beyond the retained block, sufficient for x >= x_min.")
        .def("nusselt_converged_x_min", &CDGraetzIsothermalSolution<T>::nusselt_converged_x_min, py::arg("tol"),
             "ln(1/tol)/Lambda_max: exponential-truncation bound of the Plain Nusselt sum.")
        .def("get_fully_developed_nusselt_number", &CDGraetzIsothermalSolution<T>::get_fully_developed_nusselt_number)
        .def("get_nusselt_mode_data", &CDGraetzIsothermalSolution<T>::get_nusselt_mode_data,
             "Per axisymmetric mode: (m, rate, coefficient, wall slope, flow-weighted moment).")
        .def("get_blurriness", py::overload_cast<T, BlurrinessTail>(&CDGraetzIsothermalSolution<T>::get_blurriness, py::const_),
             py::arg("x"), py::arg("tail") = BlurrinessTail::Asymptotic,
             "Blurriness B(x) with the given treatment of the omitted modes (BlurrinessTail.Dropped, "
             ".Constant or .Asymptotic; see the enum docstring).")
        .def("get_blurriness", py::overload_cast<const std::vector<T> &, BlurrinessTail>(&CDGraetzIsothermalSolution<T>::get_blurriness, py::const_),
             py::arg("x_points"), py::arg("tail") = BlurrinessTail::Asymptotic, "Vectorised get_blurriness with a BlurrinessTail.")
        .def("get_blurriness", py::overload_cast<T, bool>(&CDGraetzIsothermalSolution<T>::get_blurriness, py::const_),
             py::arg("x"), py::arg("include_truncation_tail"),
             "Blurriness B(x) of the section at the dimensionless axial coordinate x = x_phys / (Pe R) (bool tail form; the BlurrinessTail form, default Asymptotic, is preferred):\n"
             "  B(x)^2 = || psi(x) - f ||^2_omega / || psi_inf - f ||^2_omega ,\n"
             "evaluated from the series coefficients through the L2_omega Gram matrix of the retained "
             "modes (the Parseval identity, eq. gram_parseval): numerator a^T W a with a_k = c_k (e^{-Lam_k x} - 1), "
             "denominator the same form at x = inf. With include_truncation_tail (default) the Parseval "
             "defect D_K = ||f||^2 - c^T W c is added to both, which is the exact blurriness of the truncated "
             "field for x >> 1/Lam_K and equals a quadrature of the series over the disk; it requires the "
             "L2_omega projection (bare or Weighted). Without it the ratio is 0 at x = 0 and 1 at x = inf "
             "in every projection space. Requires an active bare or finite-Peclet setup (not QEP).")
        .def("get_blurriness", py::overload_cast<const std::vector<T> &, bool>(&CDGraetzIsothermalSolution<T>::get_blurriness, py::const_),
             py::arg("x_points"), py::arg("include_truncation_tail"),
             "Vectorised get_blurriness.")
        .def("get_blurriness_truncation_floor", &CDGraetzIsothermalSolution<T>::get_blurriness_truncation_floor,
             "B_K(0) = sqrt(D_K / ||psi_inf - f||^2): the resolution floor of the truncated series at the inlet.")
        .def("get_far_field_value", &CDGraetzIsothermalSolution<T>::get_far_field_value,
             "Far-field constant psi_inf in the user's units (zero-eigenvalue mode; wall value for a Dirichlet basis).")
        .def("get_inlet_square_norm", &CDGraetzIsothermalSolution<T>::get_inlet_square_norm,
             "||f||^2_omega of the user's inlet, exact (no truncation).")
        .def("get_inlet_far_field_square_norm", &CDGraetzIsothermalSolution<T>::get_inlet_far_field_square_norm,
             "||f - psi_inf||^2_omega of the user's inlet: the denominator of the blurriness.")
        .def("get_inlet_projection_square_norm", &CDGraetzIsothermalSolution<T>::get_inlet_projection_square_norm,
            "Squared norm of the user-scaled inlet projection. Bare and QEP setups return "
            "the L2_omega norm; finite-Peclet setups return the norm of their selected "
            "projection space (L2_omega or L2_r). The stored Galerkin contraction is "
            "c^T b in that same space. Requires an active solution.")
        .def("set_fp_quad_panels", &CDGraetzIsothermalSolution<T>::set_fp_quad_panels)
        .def("set_fp_cap_quad_margin", &CDGraetzIsothermalSolution<T>::set_fp_cap_quad_margin, py::arg("margin"),
             "Margin for the interface-cap/full-disk Gauss-Jacobi rule's node count. Each "
             "retained mode gets its own quadrature order, sized from its own bare root: "
             "order = ceil(bare_root) + margin, rounded up to a bucket boundary. Graetz has "
             "no interfaces to cap-project, but its uniform-inlet load vector is still a "
             "per-mode quadrature of this order. Kept for API symmetry with CDStratifiedSolution.")
        .def("get_fp_cap_quad_margin", &CDGraetzIsothermalSolution<T>::get_fp_cap_quad_margin)
        .def("set_nusselt_asymptotic_terms", &CDGraetzIsothermalSolution<T>::set_nusselt_asymptotic_terms,
             py::arg("terms"),
             "Number of terms M of the sum in G_inf (eq. G_inf_split_asymptotic) used by "
             "NusseltEvaluation.AsymptoticSplit (default 200). Does not invalidate the solution.")
        .def("get_nusselt_asymptotic_terms", &CDGraetzIsothermalSolution<T>::get_nusselt_asymptotic_terms)
        .def("set_gram_method", &CDGraetzIsothermalSolution<T>::set_gram_method, py::arg("method"))
        .def("get_gram_method", &CDGraetzIsothermalSolution<T>::get_gram_method)
        .def("set_gram_oversampling", &CDGraetzIsothermalSolution<T>::set_gram_oversampling, py::arg("factor"), py::arg("margin"))
        .def("set_gram_order_check", &CDGraetzIsothermalSolution<T>::set_gram_order_check, py::arg("enabled"))
        .def("set_gram_minimum_factor", &CDGraetzIsothermalSolution<T>::set_gram_minimum_factor, py::arg("factor"))
        .def("get_gram_minimum_factor", &CDGraetzIsothermalSolution<T>::get_gram_minimum_factor)
        .def("set_rhs_method", &CDGraetzIsothermalSolution<T>::set_rhs_method, py::arg("method"))
        .def("get_rhs_method", &CDGraetzIsothermalSolution<T>::get_rhs_method)
        .def("set_rhs_representer_margin", &CDGraetzIsothermalSolution<T>::set_rhs_representer_margin, py::arg("margin"))
        .def("get_rhs_representer_margin", &CDGraetzIsothermalSolution<T>::get_rhs_representer_margin)
        .def("set_projection_space", py::overload_cast<ProjectionSpace>(&CDGraetzIsothermalSolution<T>::set_projection_space),
             py::arg("space"),
             "Inner product used for the finite-Peclet inlet projection: ProjectionSpace.Weighted "
             "(L2_omega, weight (1-r^2) r dr -- the default) or ProjectionSpace.Radial (L2_r, weight "
             "r dr). Radial requires RhsMethod.DirectQuadrature and GramMethod.GaussJacobiQR; other "
             "combinations are rejected by setup_solution rather than silently falling back.")
        .def("set_projection_space", py::overload_cast<const std::string &>(&CDGraetzIsothermalSolution<T>::set_projection_space),
             py::arg("space"),
             "String selector for set_projection_space: \"w\"/\"omega\"/\"weighted\" or \"r\"/\"radial\" "
             "(case-insensitive). Unknown strings raise ValueError.")
        .def("get_projection_space", &CDGraetzIsothermalSolution<T>::get_projection_space)
        .def("set_fp_tolerance", &CDGraetzIsothermalSolution<T>::set_fp_tolerance,
             py::arg("tolerance"),
             "Set the finite-Peclet root relative tolerance and the requested absolute "
             "tolerance for the wall-centered positive-cap series; the implementation "
             "also applies fixed-precision stability checks.")
        .def("set_fp_max_iter", &CDGraetzIsothermalSolution<T>::set_fp_max_iter,
             py::arg("max_iterations"),
             "Set both the maximum root-solver iterations and the maximum number of "
             "wall-centered positive-cap series summands.")
        .def("compute_and_fix_coefficients", &CDGraetzIsothermalSolution<T>::compute_and_fix_coefficients)
        .def("fix_coefficients", &CDGraetzIsothermalSolution<T>::fix_coefficients)
        .def("free_coefficients", &CDGraetzIsothermalSolution<T>::free_coefficients)
        .def("save_coefficients", &CDGraetzIsothermalSolution<T>::save_coefficients)
        .def("print_info", &CDGraetzIsothermalSolution<T>::print_info)
        .def("set_max_warnings", &CDGraetzIsothermalSolution<T>::set_max_warnings)
        .def("set_verbose_level", &CDGraetzIsothermalSolution<T>::set_verbose_level)
        .def("get_verbose_level", &CDGraetzIsothermalSolution<T>::get_verbose_level)
        ;
}

// Standalone radial/angular/axial building blocks from math_functions.h, exposed
// under MassDiffusion3Dcpp.definitions so they can be evaluated from Python without
// constructing a full solver object. `suffix` is "" for the double-precision variant
// and "_ld" for the long-double variant (mirrors the diagnostics submodule's
// compute_layer_integral / compute_layer_integral_ld pair) — Python floats only carry
// double precision either way, but the "_ld" variant still computes internally in
// long double before rounding down, which can matter for the higher modes.
template <typename T>
void bind_definitions(py::module_ &defs, const std::string &suffix) {
    defs.def(("psinm_r" + suffix).c_str(),
        [](unsigned n, T b, T r) { return psinm_r<T>(n, b, r); },
        py::arg("n"), py::arg("b"), py::arg("r"),
        "Radial eigenfunction R_n(b, r).");

    defs.def(("dpsinm_r" + suffix).c_str(),
        [](unsigned order, unsigned n, T b, T r) { return dpsinm_r<T>(order, n, b, r); },
        py::arg("order"), py::arg("n"), py::arg("b"), py::arg("r"),
        "order-th (1 or 2) radial derivative of psinm_r with respect to r.");

    defs.def(("gnm_x" + suffix).c_str(),
        [](T b, T x) { return gnm_x<T>(b, x); },
        py::arg("b"), py::arg("x"),
        "Axial decay function exp(-b^2 * x).");

    defs.def(("sn_phi" + suffix).c_str(),
        [](unsigned n, T phi) { return sn_phi<T>(n, phi); },
        py::arg("n"), py::arg("phi"),
        "Angular eigenfunction S_n(phi).");

    defs.def(("dsn_phi" + suffix).c_str(),
        [](unsigned order, unsigned n, T phi) { return dsn_phi<T>(order, n, phi); },
        py::arg("order"), py::arg("n"), py::arg("phi"),
        "order-th angular derivative of sn_phi with respect to phi.");

    defs.def(("get_gaussian_integration" + suffix).c_str(),
        [](const std::function<T(T)> &f, const std::vector<T> &weights, const std::vector<T> &nodes) {
            T result;
            gaussian_integration(
                [&f](T x, T &out) { out = f(x); },
                weights, nodes, result);
            return result;
        },
        py::arg("f"), py::arg("weights"), py::arg("nodes"),
        "Weighted quadrature sum: sum_i weights[i] * f(nodes[i]).\n"
        "f is a Python callable taking one float and returning one float. weights/nodes "
        "are typically the (weight, node) columns read from one of the solver's own "
        "gaussian_weights_eigenvalues_hypergeometric_n_*.txt tables (src/CDBaseSolution/data/), "
        "so this reproduces the exact quadrature the C++ solver uses for coefficient "
        "projection and norm computation.");

    // ---- finite-Peclet primitives (finite_peclet_radial.h / finite_peclet_norms.h) ----
    // Everything the flow equation of Sec. 5.4 needs: the modified radial mode and the
    // two quadratures B + 2 kappa Lam C = N^2_fp (eq. norm_fp) and C = U_mm (eq. U_nm_def).

    defs.def(("btilde_from_b" + suffix).c_str(),
        [](T b, T kappa) { return btilde_from_b<T>(b, kappa); },
        py::arg("b"), py::arg("kappa"),
        "Shifted Kummer carrier bt = b*(1 + kappa*b^2) = b*(1 + kappa*Lam) "
        "(eq. b_and_btilde_def).");

    defs.def(("psinm_r_fp" + suffix).c_str(),
        [](unsigned n, T b, T bt, T r) { return psinm_r_fp<T>(n, b, bt, r); },
        py::arg("n"), py::arg("b"), py::arg("bt"), py::arg("r"),
        "Finite-Peclet radial eigenfunction R_nm(r) (eq. radial_solution_fp). "
        "Reduces to psinm_r(n, b, r) when bt == b.");

    defs.def(("psi_at_1_fp" + suffix).c_str(),
        [](unsigned n, T b, T bt) { return psi_at_1_fp<T>(n, b, bt); },
        py::arg("n"), py::arg("b"), py::arg("bt"),
        "Dirichlet characteristic value R_nm(1) (eq. char_dirichlet_fp).");

    defs.def(("dpsi_dr_at_1_fp" + suffix).c_str(),
        [](unsigned n, T b, T bt) { return dpsi_dr_at_1_fp<T>(n, b, bt); },
        py::arg("n"), py::arg("b"), py::arg("bt"),
        "Neumann characteristic value (n+1)*R_nm'(1) (eq. char_neumann_fp).");

    defs.def(("fp_char" + suffix).c_str(),
        [](unsigned n, T Lam, T kappa, WallCondition wall) {
            return fp_char<T>(n, Lam, kappa, wall);
        },
        py::arg("n"), py::arg("Lam"), py::arg("kappa"), py::arg("wall"),
        "Characteristic function F^{D/N}_n(Lam; kappa); its positive zeros are the "
        "exact decay rates (eqs. char_dirichlet_fp / char_neumann_fp).");

    defs.def(("fp_seed" + suffix).c_str(),
        [](unsigned n, unsigned m, T base_beta2, T kappa, T alpha, WallCondition wall) {
            return fp_seed<T>(n, m, base_beta2, kappa, alpha, wall);
        },
        py::arg("n"), py::arg("m"), py::arg("base_beta2"), py::arg("kappa"),
        py::arg("alpha"), py::arg("wall"),
        "Closed-form seed Lam = 2 beta^2 / (1 + sqrt(1 + 4 kappa alpha beta^2)) "
        "(eq. seed_root).");

    defs.def(("fp_generalized_norm" + suffix).c_str(),
        [](unsigned n, T b, T bt, T kappa, T Lam, unsigned panels) {
            return fp_generalized_norm<T>(n, b, bt, kappa, Lam, panels);
        },
        py::arg("n"), py::arg("b"), py::arg("bt"), py::arg("kappa"), py::arg("Lam"),
        py::arg("panels") = 64u,
        "Generalised norm N^2_fp = int_0^1 [(1-r^2) + 2 kappa Lam] R_nm^2 r dr "
        "(eq. norm_fp). Equals B + 2 kappa Lam C in the notation of "
        "eq. rate_energy_quadratic, i.e. exactly the denominator of the flow "
        "equation (eq. flow_equation) times C.");

    defs.def(("fp_unweighted_overlap" + suffix).c_str(),
        [](unsigned n, T bi, T bti, T bj, T btj, unsigned panels) {
            return fp_unweighted_overlap<T>(n, bi, bti, bj, btj, panels);
        },
        py::arg("n"), py::arg("bi"), py::arg("bti"), py::arg("bj"), py::arg("btj"),
        py::arg("panels") = 64u,
        "Unweighted radial overlap U^n_km = int_0^1 r R_nm R_nk dr (eq. U_nm_def). "
        "Pass identical (b, bt) pairs for the diagonal C = U_mm of "
        "eq. rate_energy_quadratic.");

    defs.def(("fp_solve_ladder" + suffix).c_str(),
        [](unsigned n, const std::vector<T> &base_beta2, T kappa, WallCondition wall,
           bool skip_zero_mode, T rel_tol, unsigned max_iter,
           const std::vector<T> &alpha_by_m) {
            std::vector<unsigned> iters;
            std::vector<T> roots = fp_solve_ladder<T>(n, base_beta2, kappa, wall, skip_zero_mode, rel_tol, max_iter, iters, alpha_by_m);
            return py::make_tuple(roots, iters);
        },
        py::arg("n"), py::arg("base_beta2"), py::arg("kappa"), py::arg("wall"),
        py::arg("skip_zero_mode"), py::arg("rel_tol"), py::arg("max_iter"),
        py::arg("alpha_by_m") = std::vector<T>(),
        "Solve the ordered set of rates for angular index n. alpha_by_m are the exact "
        "seed ratios alpha_{nm} = U^n_mm / N_nm^2 (eq. seed_root), aligned with "
        "base_beta2; omit (or pass []) to fall back to a representative constant. "
        "Returns a tuple (roots, final_iters).");

    // ---- diagnostics on the retained mode set (diagnostics/fp_parseval_diagnostics.h) ----
    // Off every solution path: these exist so an external check of the Parseval
    // identity (eq. gram_parseval) can reuse the solver's own quadrature and
    // integral kernels instead of re-implementing them in the calling language.

    defs.def(("cap_weight" + suffix).c_str(),
        [](T z) {
            return static_cast<T>(DIAG_SQRT_HALF_PI) * layer_integral_diag::closed_form_n0<T>(z);
        },
        py::arg("z"),
        "omega-weighted area of the disk above the chord at height z, "
        "int_{z' > z} (1 - r^2) dA. Equals pi/2 at z = -1 and 0 at z = +1, so the "
        "layer between z1 < z2 carries cap_weight(z1) - cap_weight(z2). This is the "
        "sqrt(pi/2) rescaling of layer_integral_diag::closed_form_n0 -- the same closed "
        "form CDBaseSolution::compute_partial_flux evaluates, reached without "
        "duplicating it.");

    defs.def(("fp_assemble_spectral_data" + suffix).c_str(),
        [](const std::vector<unsigned> &n, const std::vector<unsigned> &m,
           const std::vector<T> &root_fp, const std::vector<T> &btilde_fp,
           const std::vector<T> &rate_fp, T kappa, unsigned panels) {
            FPSpectralAssembly<T> a = fp_assemble_spectral_data<T>(
                n, m, root_fp, btilde_fp, rate_fp, kappa, panels);
            return py::make_tuple(std::move(a.generalized_norm),
                                  std::move(a.block_angular_index),
                                  std::move(a.block_modes),
                                  std::move(a.block_overlap));
        },
        py::arg("angular_index"), py::arg("radial_index"), py::arg("root_fp"),
        py::arg("btilde_fp"), py::arg("rate_fp"), py::arg("kappa"),
        py::arg("panels") = 64u,
        "Generalised norms Nfp^2 (eq. norm_fp) and per-angular-block normalised "
        "overlaps Uhat_mk = U_mk / (Nfp_m Nfp_k) (eq. U_nm_def) for a retained mode "
        "set, from ONE pass of the solver's own radial-table builder.\n"
        "Input arrays are the SeriesTermData fields (n, m, root_fp, btilde_fp, rate_fp) "
        "of the retained modes, grouped by n and strictly increasing in m -- exactly the "
        "order get_modified_data() returns.\n"
        "Returns (generalized_norm, block_angular_index, block_modes, block_overlap): "
        "one norm per mode in input order; then per block its angular index, the input "
        "indices of its modes, and its row-major K_n x K_n normalised overlap. The "
        "overlap is normalised because Nfp^2 spans ~160 decades at beta_max = 600; "
        "contract it against c_m * Nfp_m.");

    defs.def(("fp_solve_bracketed" + suffix).c_str(),
        [](unsigned n, T kappa, WallCondition wall, T lo, T hi, T rel_tol, unsigned max_iter) {
            unsigned iters = 0;
            T root = fp_solve_bracketed<T>(n, kappa, wall, lo, hi, rel_tol, max_iter, iters);
            return py::make_tuple(root, iters);
        },
        py::arg("n"), py::arg("kappa"), py::arg("wall"), py::arg("lo"), py::arg("hi"),
        py::arg("rel_tol"), py::arg("max_iter"),
        "Solve for ONE rate inside a verified bracket [lo, hi]. Returns a tuple (root, final_iters).");
}

// Change the name of solution_class
PYBIND11_MODULE(MassDiffusion3Dcpp, m) {
    py::enum_<SolutionMethod>(m, "SolutionMethod")
        .value("Uninitialized", SolutionMethod::Uninitialized)
        .value("Bare", SolutionMethod::Bare)
        .value("QEP", SolutionMethod::QEP)
        .value("ModifiedRoots", SolutionMethod::ModifiedRoots)
        .export_values();
    py::enum_<WallCondition>(m, "WallCondition")
        .value("Neumann", WallCondition::Neumann)
        .value("Dirichlet", WallCondition::Dirichlet)
        .export_values();
    py::enum_<GramMethod>(m, "GramMethod")
        .value("GaussJacobiQR", GramMethod::GaussJacobiQR)
        .value("Ultraspherical", GramMethod::Ultraspherical)
        .export_values();
    py::enum_<RhsMethod>(m, "RhsMethod")
        .value("DirectQuadrature", RhsMethod::DirectQuadrature)
        .value("Representer", RhsMethod::Representer)
        .export_values();
    py::enum_<ProjectionSpace>(m, "ProjectionSpace")
        .value("Weighted", ProjectionSpace::Weighted)
        .value("Radial", ProjectionSpace::Radial)
        .export_values();

    py::enum_<BlurrinessTail>(m, "BlurrinessTail",
        "Treatment of the modes beyond the truncation in get_blurriness: Dropped (retained modes only; "
        "0 at the inlet, misses the energy D_K of the omitted modes downstream), Constant (D_K added "
        "as fully decayed: the blurriness of the truncated field, floor sqrt(D_K/||f-psi_inf||^2) at "
        "the inlet), Asymptotic (the omitted modes decay following the spectral energy density "
        "beta^-(p+1) of the projection error, D_K ~ beta_K^-p, normalised to the exact D_K: starts at "
        "0 with the physical small-x law and coincides with Constant downstream of the truncation scale).")
        .value("Dropped", BlurrinessTail::Dropped)
        .value("Constant", BlurrinessTail::Constant)
        .value("Asymptotic", BlurrinessTail::Asymptotic);

    py::enum_<NusseltEvaluation>(m, "NusseltEvaluation")
        .value("Plain", NusseltEvaluation::Plain)
        .value("SpectralSplit", NusseltEvaluation::SpectralSplit)
        .value("SlugSplit", NusseltEvaluation::SlugSplit)
        .value("AsymptoticSplit", NusseltEvaluation::AsymptoticSplit);

    bind_class<double>(m, "CDStratifiedSolution");
    bind_class<long double>(m, "CDStratifiedSolutionLD");


    bind_graetz_class<double>(m, "CDGraetzIsothermalSolution");
    bind_graetz_class<long double>(m, "CDGraetzIsothermalSolutionLD");

    py::module_ defs = m.def_submodule(
        "definitions",
        "Standalone radial/angular/axial eigenfunctions (psinm_r, gnm_x, sn_phi, ...) "
        "usable without constructing a solver object.");
    bind_definitions<double>(defs, "");
    bind_definitions<long double>(defs, "_ld");

    py::module_ diag = m.def_submodule("diagnostics");
    diag.def("get_residues_gaussian", &get_residues_gaussian);

    // Expose LayerIntegralComponents as a named type so Python can inspect fields
    py::class_<LayerIntegralComponents>(diag, "LayerIntegralComponents")
        .def_readonly("gaussian_part",        &LayerIntegralComponents::gaussian_part)
        .def_readonly("hypergeometric_part",  &LayerIntegralComponents::hypergeometric_part)
        .def_readonly("closed_form_ref",      &LayerIntegralComponents::closed_form_ref)
        .def_readonly("total",                &LayerIntegralComponents::total)
        .def_readonly("n_nodes",              &LayerIntegralComponents::n_nodes)
        .def("__repr__", [](const LayerIntegralComponents &c) {
            return "LayerIntegralComponents(gaussian=" + std::to_string(c.gaussian_part)
                 + ", hypergeometric=" + std::to_string(c.hypergeometric_part)
                 + ", closed_form_ref=" + std::to_string(c.closed_form_ref)
                 + ", total=" + std::to_string(c.total)
                 + ", n_nodes=" + std::to_string(c.n_nodes) + ")";
        });

    // double precision variant
    diag.def("compute_layer_integral",
        [](unsigned n, unsigned m, double z_i, double root,
           const std::vector<double> &weights, const std::vector<double> &nodes) {
            return compute_layer_integral_components<double>(n, m, z_i, root, weights, nodes);
        },
        py::arg("n"), py::arg("m"), py::arg("z_i"), py::arg("root"),
        py::arg("gauss_weights"), py::arg("gauss_nodes"),
        "Compute all components of the single-layer integral I(n,m,z_i,root).\n"
        "Returns LayerIntegralComponents with gaussian_part, hypergeometric_part,\n"
        "closed_form_ref (n==0 only, else NaN), total, and n_nodes.");

    // long double precision variant
    diag.def("compute_layer_integral_ld",
        [](unsigned n, unsigned m, double z_i, double root,
           const std::vector<double> &weights, const std::vector<double> &nodes) {
            return compute_layer_integral_components<long double>(n, m, z_i, root, weights, nodes);
        },
        py::arg("n"), py::arg("m"), py::arg("z_i"), py::arg("root"),
        py::arg("gauss_weights"), py::arg("gauss_nodes"),
        "Same as compute_layer_integral but uses long double internally.");
}
