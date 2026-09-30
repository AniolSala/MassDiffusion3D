#if !defined(DATA_STRUCT_H)
#define DATA_STRUCT_H

// Flat per-mode record that lets the evaluation loops iterate a single vector
// instead of a nested (n, m) structure.
template <typename Ttype>
struct SeriesTermData
{
    unsigned n;
    unsigned m;
    Ttype root;
    Ttype norm;
    Ttype coeff;
    // Rate used in exp(-exp_rate * x); equals root² for the base solution.
    Ttype exp_rate;

    // --- Exact finite-Péclet (axial-diffusion) eigen-data. ------------------
    // Populated by the setup_solution(peclet, ...) orchestrator via the
    // finite_peclet_* kernels; theory/analytical_solution.tex Sec. 5.2–5.3.
    // All default to the bare (κ→0) values so an object that never ran the
    // finite-Péclet setup behaves byte-for-byte as the base solution.
    //   root_fp   = b   = sqrt(Λ)                (unshifted radial parameter)
    //   btilde_fp = b̃  = b(1 + κΛ)              (shifted Kummer-parameter carrier)
    //   rate_fp   = Λ   (axial decay rate, used in exp(-Λ x))
    //   norm_fp   = 𝒩²_fp = ∫₀¹ (1−r²+2κΛ) R_fp² r dr   (eq. norm_fp)
    //   coeff_fp  = Ĉ  (modified inlet coefficient, eq. coefficients_fp)
    Ttype root_fp   = static_cast<Ttype>(0);
    Ttype btilde_fp = static_cast<Ttype>(0);
    Ttype rate_fp   = static_cast<Ttype>(0);
    Ttype norm_fp   = static_cast<Ttype>(0);
    Ttype coeff_fp  = static_cast<Ttype>(0);
};

#endif // DATA_STRUCT_H
