#ifndef CD_BASE_SOLUTION_H
#define CD_BASE_SOLUTION_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <filesystem>

#include <omp.h>

#include "../series_term_struct.h"
#include "../solution_method.h"
#include "../gram_method.h"
#include "../rhs_method.h"
#include "../projection_space.h"
#include "../finite_peclet_roots.h"   // WallCondition + fp_modify_roots
#include "../math_functions.h"        // psinm_r / sn_phi (bare-path evaluation)
#include "../finite_peclet_radial.h"  // psinm_r_fp (finite-Péclet evaluation)
#include "../finite_peclet_radial_block.h"
#include "../qep_computations.h"      // QEPBlock + qep_build_and_solve / qep_evaluate_at_points
#include "../nusselt_postprocessing.h" // NusseltEvaluation + tail rates of the split evaluations
#include "../finite_peclet_gram_gauss_jacobi.h" // L2_omega Gram blocks of the finite-Peclet modes (blurriness)

// How get_blurriness accounts for the modes beyond the truncation (see the
// protected "Blurriness" block of CDBaseSolution).
//   Dropped     : the retained modes alone; 0 at x = 0 and 1 at x = inf in every
//                 projection space, but the energy D_K of the omitted modes is
//                 missing at every x.
//   Constant    : D_K added to numerator and denominator, i.e. the omitted modes
//                 are taken as fully decayed. Exact for the TRUNCATED field (what
//                 a quadrature of the series over the disk returns) and exact for
//                 the true field once e^{-Lam_K x} is negligible; at the inlet it
//                 sits on the floor B_K(0) = sqrt(D_K / ||f - psi_inf||^2).
//   Asymptotic  : the omitted modes decay with x according to the spectral
//                 energy density that the theory establishes for the inlet
//                 projection error (eq. truncation_error_fK: D_K ~ beta_K^-p,
//                 hence a density ~ beta^-(p+1) beyond beta_K), normalised to
//                 the exact D_K: T_K(x) = D_K int_0^1 p s^(p-1) (1 - e^{-Lam(beta_K/s) x})^2 ds,
//                 with Lam(beta) the finite-Peclet rate of the bare root beta.
//                 T_K(0) = 0 and T_K(inf) = D_K, so the curve starts at 0 and
//                 coincides with Constant downstream of the truncation scale.
enum class BlurrinessTail
{
    Dropped,
    Constant,
    Asymptotic
};

template <typename Ttype>
class CDBaseSolution
{
protected:
    std::vector<Ttype> m_zi; // Positions of the interfaces
    std::vector<Ttype> m_ui; // Values of the density values
    std::vector<Ttype> m_gauss_points;
    std::vector<Ttype> m_gauss_weights;
    std::vector<unsigned> m_max_n_by_m;

    std::vector<std::vector<std::vector<Ttype>>> m_pseudo_products_matrix;

    using SeriesData = SeriesTermData<Ttype>;
    std::vector<SeriesData> m_series_data; // Avoid nested loops
    // Offsets of the angular blocks in m_series_data: block n occupies
    // [m_block_start[n], m_block_start[n+1]). set_series_data pushes modes
    // n-major with m ascending, so every block is one contiguous range and the
    // decay rate increases along it. Size = number of blocks + 1; empty until
    // set_series_data runs.
    std::vector<std::size_t> m_block_start;
    // Relative amplitude below which a mode is skipped when the series is
    // EVALUATED (see cutoff_modes). 0 disables the cut, which is the default and
    // reproduces the untruncated sum bit for bit. Never affects m_max_root, the
    // mode set, or the coefficients.
    Ttype m_eval_cutoff_tol = static_cast<Ttype>(0);

    unsigned m_n_layers; // Number of layers
    unsigned m_number_of_gauss_points;
    unsigned m_max_K;

    Ttype m_max_ui; // Maximum value of the density
    Ttype m_min_ui; // Minimum value of the density
    Ttype m_constant_term;
    Ttype m_max_root;

    unsigned m_max_warnings;
    unsigned m_current_warning;
    unsigned short m_verbose_level;

    bool mCoefficientsAreFixed;
    bool mCoefficientsAreComputed;
    SolutionMethod m_solution_method = SolutionMethod::Uninitialized;

    // --- Finite-Péclet (axial-diffusion) state. -----------------------------
    // The exact axial-diffusion solution is the DEFAULT (m_use_axial_diffusion =
    // true). The bare (axial-diffusion-free) solution is the special case
    // κ → 0, selected by setup_solution(peclet = +inf). See
    // theory/analytical_solution.tex Sec. 5.
    bool  m_use_axial_diffusion; // retained internally for numerical-kernel compatibility
    Ttype m_peclet;              // Péclet number used at the last setup (inf ⇒ bare)
    Ttype m_kappa;               // κ = Pe⁻² of the last finite-Péclet setup
    Ttype m_fp_rel_tol;          // relative tolerance for the rate root-finding
    unsigned m_fp_max_iter;      // max Newton/bisection iterations per rate
    unsigned m_fp_quad_panels;   // composite-GL panels used by the QEP overlap assembly
    // Margin for the order of the (alpha=0, beta=1/2) Gauss-Jacobi cap rule
    // (Appendix B) that projects stratified-inlet interface jumps onto the
    // radial modes -- the load vector for the finite-Peclet Gram solve. Each
    // retained mode gets its OWN order, sized from its own bare (kappa-
    // independent) root: fp_rhs_required_order(root, m_fp_cap_quad_margin) =
    // ceil(root) + margin, rounded up to a bucket boundary (see
    // finite_peclet_rhs.h). This is deliberately per-mode, not per-block: the
    // cap/full-disk integrals are independent 1-D integrals (unlike the
    // Gram matrix, which couples a whole block's K modes into one K-by-K
    // system), so sizing them off the whole block's mode count would
    // over-resolve every mode less oscillatory than the block's worst one.
    // Measured sufficient (margin=100 converges the cap quadrature to
    // ~1e-14 relative even for the most oscillatory mode, root~67, in this
    // repo's max_root=600 benchmark configuration).
    unsigned m_fp_cap_quad_margin = 100u;
    // Terms M of the sum in G_inf (NusseltEvaluation::AsymptoticSplit).
    unsigned m_nusselt_asymptotic_terms = 200u;
    GramMethod m_gram_method = GramMethod::GaussJacobiQR;
    unsigned m_gram_oversampling_factor = 2u;
    unsigned m_gram_oversampling_margin = 20u;
    bool m_gram_order_check = false;
    unsigned m_gram_minimum_factor = 2u;
    // RHS load-vector assembly backend; see rhs_method.h. Representer is the
    // production default (plans/RHS_REPRESENTER_PLAN.md section 11): it
    // reduces the RHS pass to zero confluent-hypergeometric evaluations by
    // reusing the Gram assembly's own retained samples, and requires
    // GramMethod::GaussJacobiQR (checked at setup time). There is no
    // fallback between backends, mirroring m_gram_method.
    RhsMethod m_rhs_method = RhsMethod::Representer;
    unsigned m_rhs_representer_margin = 40u;
    // Inner product the finite-Peclet inlet coefficients are projected in; see
    // projection_space.h and finite_peclet_coefficients_radial.h. Weighted
    // (L2_omega) is the historical and default behaviour, and a configuration
    // that never touches the setter below is bit-for-bit unchanged by the
    // existence of the alternative.
    ProjectionSpace m_projection_space = ProjectionSpace::Weighted;

    // || f_K ||^2 of the truncated inlet reconstruction, in the inner product
    // the active setup projected in, and for the INTERNALLY SCALED inlet that
    // set_ui built. Every setup_* path fills it; NaN means none has, which is
    // what get_inlet_projection_square_norm reports instead of a stale value.
    // On the finite-Peclet path it is the contraction of the coefficients with
    // the load vector, c^T b, which equals c^T W c because the coefficients
    // solve G c = b; on the bare and QEP paths the Gram matrix is diagonal and
    // the same quantity is sum_k N_k^2 c_k^2. The public getter converts it
    // back to the user's scaling.
    Ttype m_inlet_projection_square_norm;

    // QEP spectral data, cached so that repeated evaluations at one Péclet do not
    // repeat the eigen-decomposition. Rebuilt whenever the requested κ changes.
    std::vector<QEPBlock<Ttype>> m_qep_blocks;
    Ttype m_qep_kappa;

    std::filesystem::path m_data_dir;

    // Setters
    void set_zi(std::vector<Ttype> zi);
    void set_ui(std::vector<Ttype> ui);
    virtual void set_norms(std::vector<std::vector<Ttype>> &) const = 0;
    virtual void set_pseudo_products_matrix(std::vector<std::vector<std::vector<Ttype>>> &) = 0;
    void set_series_data();
    void set_gaussian_weights_and_points();
    void set_constant_term();
    void invalidate_solution();
    void require_solution_ready(const char *caller) const;

    // Getters
    Ttype get_unscaled_solution_value(const Ttype &);

    // Solution computations
    void prepare_to_compute_solution(const std::vector<Ttype> &, const std::vector<Ttype> &);
    void compute_solution(const std::vector<Ttype> &, const std::vector<Ttype> &, const std::vector<Ttype> &, std::vector<Ttype> &);
    void compute_solution(Ttype, const std::vector<Ttype> &, const std::vector<Ttype> &, std::vector<Ttype> &);

    // Others
    Ttype compute_partial_flux(const Ttype) const;
    void modify_n_layers(unsigned new_n_layers);
    void warn(std::string);

    // Returns true for modes whose contribution is the constant (zero-eigenvalue)
    // mode. The default covers the stratified problem's β_{00}=0 mode; Graetz
    // overrides this to return false (all its eigenvalues are non-trivial).
    virtual bool is_zero_eigenvalue_mode(unsigned n, unsigned m) const
    {
        return (n == 0 && m == 0);
    }

    // --- Inlet-projection norm: constant-function terms. --------------------
    // Unscaling || f_K ||^2 from the internal profile f~ to the user's f needs
    // two quantities beyond the stored contraction, neither of which involves a
    // new integral against the inlet (see the getter): the squared norm of the
    // projected constant, || P1 ||^2, and its pairing with the inlet,
    // <P1, f~>. Both default to the closed form valid when the constant
    // function IS a retained mode (Neumann wall, Lambda = 0, R == 1), where
    // P1 = 1 exactly: the weighted disk area and that area times the
    // flow-weighted mean already held in m_constant_term. A basis without the
    // constant mode must override both -- the default refuses rather than
    // returning the wrong closed form.
    virtual Ttype constant_projection_square_norm() const;
    virtual Ttype constant_projection_inlet_cross() const;
    bool has_retained_constant_mode() const;
    // sum_k N_k^2 c_k^2 over the retained modes: the same || f_K ||^2 on the bare
    // and QEP paths, where the Gram matrix is diagonal in the bare norms.
    Ttype diagonal_inlet_projection_square_norm() const;

    // Wall boundary condition of the radial eigenproblem. Stratified profiles use
    // zero-flux (Neumann) walls; the isothermal Graetz problem overrides this to
    // Dirichlet. Consumed by the finite-Péclet rate solver.
    virtual WallCondition wall_condition() const { return WallCondition::Neumann; }

    // The homogeneous series gives a wall heat-transfer coefficient only for
    // an isothermal wall. Derived mass-transport classes may reject it explicitly.
    virtual void require_nusselt_number_defined(const char *caller) const;

    // --- Active-field accessors (unified evaluation path). ------------------
    // On the finite-Péclet path they return the *_fp fields; on the bare path the
    // base fields. Because psinm_r_fp(n, b, b, r) == psinm_r(n, b, r), the modified
    // radial mode with b̃ = b reproduces the bare mode exactly, so a single loop
    // serves both solutions.
    Ttype active_coeff(const SeriesData &t) const { return t.coeff; }
    Ttype active_rate(const SeriesData &t)  const { return t.root * t.root; }
    // --- Evaluation-time mode cut (cutoff_modes). ---------------------------
    // max_u |R_k(r_u)| for every mode, read off the radial table that the
    // evaluation kernels have already built (rows = unique radii, K per row).
    void radial_row_maxima(const std::vector<Ttype> &radial_table, std::size_t mode_count,
                           std::vector<Ttype> &max_radial) const;
    // Half-open k-ranges to sum at one plane. With the cut disabled this is the
    // single range [0, mode_count), so the evaluation loop is unchanged.
    void build_eval_ranges(const std::vector<Ttype> &coeff_gnm,
                           const std::vector<Ttype> &max_radial, std::size_t mode_count,
                           std::vector<std::pair<std::size_t, std::size_t>> &ranges) const;
    // Chunk size for the radial-table build, which is parallelised over the
    // flattened (unique radius, mode) index. One radial evaluation costs
    // microseconds, so a chunk of a few tens of items already makes the OpenMP
    // hand-out free; the cap keeps at least ~4 chunks per thread so that a small
    // table (a coarse radial grid, or a low max_root) still balances.
    static int radial_table_chunk(std::size_t total_items)
    {
        const std::size_t threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
        const std::size_t target = total_items / (4 * threads);
        return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(64, target)));
    }
    Ttype active_radial(const SeriesData &t, const Ttype &r) const
    {
        return m_solution_method == SolutionMethod::ModifiedRoots
            ? psinm_r_fp(t.n, t.root, t.root * (static_cast<Ttype>(1) + m_kappa * t.root * t.root), r)
            : psinm_r(t.n, t.root, r);
    }
    Ttype active_radial_derivative_at_wall(const SeriesData &t) const
    {
        return radial_derivative_at_1_fp(t.n, t.root,
            t.root * (static_cast<Ttype>(1) + m_kappa * t.root * t.root));
    }
    // Separately-added constant. The bare path carries the flow-weighted mean in
    // m_constant_term (the zero mode is skipped there); on the finite-Péclet path
    // the zero mode is an ordinary coefficient (reviewer decision), so no extra
    // constant is added.
    Ttype active_constant() const { return static_cast<Ttype>(0); }

    // --- Blurriness (thesis, chapter "Applications", Sec. "Blurriness"). ------
    // The blurriness of the section at x is
    //     B(x)^2 = || psi(x) - f ||^2_omega / || psi_inf - f ||^2_omega ,
    // with f the inlet datum, psi_inf the far field and omega = 1 - r^2. Writing
    // the truncated series as psi_K(x) = sum_k c_k e^{-Lam_k x} psi_k and the
    // inlet projection as f_K = sum_k c_k psi_k,
    //     psi_K(x) - f = sum_k a_k(x) psi_k - (f - f_K),   a_k(x) = c_k (e^{-Lam_k x} - 1).
    // In the L2_omega Galerkin projection (ProjectionSpace::Weighted, the bare
    // solution included) f - f_K is L2_omega-orthogonal to the retained span, so
    //     || psi_K(x) - f ||^2_omega = a(x)^T W a(x) + D_K ,
    // where W is the block-diagonal L2_omega Gram matrix of the retained modes
    // (eq. gram_matrix_weighted; N_nm^2 delta_mk at kappa = 0) and
    // D_K = || f ||^2_omega - c^T W c is the Parseval defect of the inlet
    // projection (eq. truncation_error_fK). D_K is the energy of the modes
    // beyond the truncation, which decay at least as fast as e^{-Lam_K x}, so
    // treating them as fully decayed (that is what the term D_K does) makes the
    // formula exact up to a relative error bounded by (D_K / Q_inf) e^{-Lam_K x}.
    // The denominator Q_inf is the same quadratic form at x -> inf, where a_k =
    // -c_k for every decaying mode and 0 for the zero-eigenvalue (constant) mode.
    // Everything is evaluated in the internal scaling of set_ui; the ratio is
    // invariant under the affine map back to the user's units.
    //
    // The three scalars of the formula are taken from the same places as the
    // Parseval analysis of the inlet projection, so that the blurriness at the
    // inlet IS that analysis in different units:
    //   D_K            = || f ||^2_omega - || f_K ||^2_omega, with || f ||^2 in closed
    //                    form (inlet_square_norm_scaled) and || f_K ||^2 the stored
    //                    Galerkin contraction c^T b (m_inlet_projection_square_norm);
    //   psi_inf        = the term of the series with zero rate (the constant mode);
    //   || f - psi_inf ||^2_omega in closed form from psi_inf and the inlet moments.
    // The Gram blocks are needed only for the x-dependent retained part a^T W a.
    //
    // The cache below is a pure function of the active setup: built lazily by
    // prepare_blurriness(), dropped by invalidate_solution(). It is mutable so
    // that the getters are const like every other post-setup getter.
    struct BlurrinessBlock
    {
        std::vector<unsigned> modes; // indices into m_series_data, one angular block
        std::vector<Ttype> gram;     // row-major K x K, W^n_mk = <R_nm, R_nk>_omega
    };
    mutable std::vector<BlurrinessBlock> m_blurriness_blocks;
    mutable Ttype m_blurriness_tail = std::numeric_limits<Ttype>::quiet_NaN();            // D_K (NaN: not available)
    mutable Ttype m_blurriness_far_field = std::numeric_limits<Ttype>::quiet_NaN();       // psi_inf, internal scaling
    mutable Ttype m_blurriness_denominator = std::numeric_limits<Ttype>::quiet_NaN();     // || f~ - psi~_inf ||^2_omega
    mutable Ttype m_blurriness_far_field_form = std::numeric_limits<Ttype>::quiet_NaN();  // a_inf^T W a_inf (retained variance)
    mutable bool m_blurriness_ready = false;
    void prepare_blurriness() const;
    // a(x)^T W a(x) summed over the angular blocks; x = +inf gives the far-field form.
    Ttype blurriness_quadratic_form(Ttype x) const;
    // || f~ ||^2_omega of the INTERNALLY SCALED inlet. Default: the layered
    // (piecewise-constant in z) profile of m_zi / m_ui through the weighted cap
    // areas. A subclass whose inlet is not that profile must override it.
    virtual Ttype inlet_square_norm_scaled() const;
    // Flow-weighted mean <f~, 1>_omega / <1, 1>_omega of the internally scaled
    // inlet. Default: m_constant_term. Overridden where m_constant_term carries
    // a different meaning (the Graetz far field).
    virtual Ttype inlet_flow_weighted_mean_scaled() const { return m_constant_term; }
    // Decay exponent p of the inlet projection error, D_K ~ beta_K^-p, used by
    // BlurrinessTail::Asymptotic. The interior jumps of a stratified inlet give
    // p = 1 for both the bare and the modified modes (theory, Sec.
    // pencil_orthogonality); the isothermal Graetz inlet overrides it.
    virtual Ttype truncation_defect_exponent() const { return static_cast<Ttype>(1); }
    // Flux-weighted jump moment of the inlet, S_omega = sum_j int_{Gamma_j} [[f]]_j^2
    // omega dl, over the curves across which the inlet datum jumps (internal scaling
    // of set_ui, so that it is homogeneous with D_K). It is the exact amplitude of the
    // harmonic entrance law N(x) -> (2 ln2 / pi) S_omega Pe x for x << kappa, where
    // the residual is confined to bands of width Pe x around the jump curves.
    // Default: the layered profile of m_zi / m_ui, whose jump curves are the chords
    // z = z_i, with omega = a_i^2 - y^2 and a_i^2 = 1 - z_i^2, so that
    //     S_omega = (4/3) sum_i (u_{i+1} - u_i)^2 (1 - z_i^2)^{3/2}.
    // A subclass whose inlet is not that profile must override it; an inlet with no
    // interior jump (or one that sits on the wall, where omega = 0) returns 0 and the
    // calibration below falls back.
    virtual Ttype inlet_jump_flux_moment() const;
    // c^2, with c = S_omega / (pi beta_K D_K): the square of the ratio between the
    // wavenumber the truncated series resolves uniformly and the truncation root
    // beta_K. Both inputs are exact, so nothing here is fitted. Calibrating on it
    // makes tail_rate reproduce the harmonic entrance law exactly at every Peclet
    // number, and D_K carries the interpolation between the two limits it must have:
    // c -> S_omega / S_sqrt(omega) when beta_K << Pe (the classical edge) and
    // c -> 1 / <varrho>_omega when beta_K >> Pe (the Bessel edge). Falls back to the
    // previous mode-averaged 2/3 when S_omega or D_K is unavailable.
    Ttype tail_edge_ratio_squared() const;
    // Finite-Peclet rate of a bare root beta beyond the truncation: the decaying root
    // of kappa Lam^2 + c^2 Lam = c^2 beta^2 (beta^2 itself on the bare path, where the
    // constant never enters).
    Ttype tail_rate(Ttype beta) const;
    // T_K(x) of BlurrinessTail::Asymptotic (internal scaling).
    Ttype blurriness_asymptotic_tail(Ttype x) const;

    // Pure virtual: must be implemented by derived classes
    virtual void set_roots(std::vector<std::vector<Ttype>> &) const = 0;
    virtual void compute_coefficients() = 0;
    // Finite-Péclet coefficient hook (writes coeff_fp). Called by setup_solution
    // on the axial-diffusion path; delegates to the finite_peclet_coefficients
    // free functions. Default throws so a subclass that has not opted in fails loudly.
    virtual void compute_coefficients_fp(const FPRadialTable<Ttype> &);
    virtual void draw_step_solution_terminal(std::ostringstream &) = 0;

private:
    // (m, rate, coefficient, wall slope, flow-weighted radial moment).
    std::vector<std::array<Ttype, 5>> nusselt_axisymmetric_mode_data(const char *caller) const;
    // Ascending rates beyond the retained axisymmetric block whose tail
    // sum_m exp(-Lam_m x) is resolved for every x >= x_min: exact finite-Peclet
    // rates (SpectralSplit) or slug-flow rates (SlugSplit).
    std::vector<Ttype> nusselt_tail_rates(const std::vector<std::array<Ttype, 5>> &modes, Ttype x_min,
                                          NusseltEvaluation evaluation, const char *caller) const;

public:
    // Constructor
    CDBaseSolution(const std::vector<Ttype> &zi, const std::vector<Ttype> &ui, unsigned short verbose_level);
    CDBaseSolution(const std::vector<Ttype> &zi, const std::vector<Ttype> &ui) : CDBaseSolution(zi, ui, 1) {}

    // Virtual destructor for safe polymorphism
    virtual ~CDBaseSolution() = default;

    // Setters
    void set_number_of_gauss_points(unsigned);
    // Skip, when EVALUATING the series, the modes whose contribution at the
    // requested plane is below `tol` relative to the largest contribution there.
    //
    // The quantity compared is the amplitude actually summed by the evaluation
    // loop, |C_k e^{-Lam_k x}| * max_u |R_k(r_u)|, NOT the coefficient and NOT
    // the exponential alone: the coefficients live in the unnormalised radial
    // basis, where their size is set by the Gram column scaling D = sqrt(W_kk)
    // that the evaluator never sees (|C| reaches 1e72 while max|R| reaches
    // 1e-76), so neither factor means anything on its own. Lam_k is the ACTIVE
    // rate -- the modified finite-Peclet rate on that path, beta^2 on the bare
    // one -- so the cut follows the decay the solution really has.
    //
    // Because the rate increases with m inside an angular block, the skipped
    // modes are a tail of each block and the retained ones stay contiguous; the
    // arithmetic of the retained terms, and their summation order, are exactly
    // those of the full sum.
    //
    // This is an evaluation-time filter only: it changes neither m_max_root nor
    // the mode set nor the coefficients, so it does not invalidate the solution
    // and can be toggled between calls. tol = 0 (the default) disables it.
    // Applies where the radial table is in use -- the only place max_u |R_k| is
    // available for free; an evaluation that falls back to per-point radial
    // evaluation keeps every mode, which is never less accurate. The
    // zero-eigenvalue (constant) mode is never dropped.
    void cutoff_modes(Ttype tol);
    Ttype get_cutoff_modes() const { return m_eval_cutoff_tol; }
    void set_max_warnings(unsigned);
    void set_verbose_level(unsigned short verbose_level) { m_verbose_level = verbose_level; }
    void set_max_root(Ttype root_value);

    // Getters
    std::vector<Ttype> get_zi() const { return m_zi; }
    std::vector<Ttype> get_ui() const { return m_ui; }
    Ttype get_tol(Ttype x_value) const;
    Ttype get_max_root() const { return m_max_root; }
    unsigned get_number_of_gauss_points() const { return m_number_of_gauss_points; }
    std::vector<std::vector<Ttype>> get_coefficients() const;
    std::vector<std::vector<Ttype>> get_roots() const;
    std::vector<std::vector<Ttype>> get_bare_root_catalog() const;
    std::vector<std::vector<Ttype>> get_norms() const;
    unsigned get_number_of_coefficients() const;
    short unsigned get_verbose_level() const { return m_verbose_level; }
    unsigned get_max_K() const { return m_max_K; }

    std::vector<Ttype> get_solution(const std::vector<Ttype> &, const std::vector<Ttype> &, const std::vector<Ttype> &);
    std::vector<Ttype> get_solution(const std::array<std::vector<Ttype>, 3> &);

    std::vector<Ttype> get_solution_cloud_of_points(const std::vector<Ttype> &, const std::vector<Ttype> &, const std::vector<Ttype> &);
    std::vector<std::vector<Ttype>> get_solution_at_planes(const std::vector<Ttype> &, const std::vector<Ttype> &, const std::vector<Ttype> &);
    std::vector<std::vector<Ttype>> get_solution_at_planes(const std::vector<Ttype> &, const std::vector<std::vector<Ttype>> &, const std::vector<std::vector<Ttype>> &);
    std::vector<Ttype> get_solution_at_points(Ttype, const std::vector<Ttype> &, const std::vector<Ttype> &);
    std::vector<Ttype> get_solution_at_points(const std::vector<Ttype> &, const std::vector<Ttype> &, const std::vector<Ttype> &);
    std::vector<Ttype> get_solution_at_points(Ttype, Ttype, Ttype);

    void group_points_in_planes(const std::vector<Ttype> &, const std::vector<Ttype> &, const std::vector<Ttype> &, std::vector<Ttype> &, std::vector<std::vector<Ttype>> &, std::vector<std::vector<Ttype>> &, std::vector<std::pair<size_t, size_t>> &, Ttype = 1e-8);
    void get_solution_at_point(const Ttype &, const Ttype &, const Ttype &, Ttype &);

    // --- Setup (initialization), separate from evaluation. ------------------
    // setup_solution builds the retained mode set at the given truncation and
    // prepares the coefficients for the requested Péclet number. It is the single
    // entry point the user calls before any get_solution*; the heavy work is
    // delegated to the finite_peclet_* free functions (root finding, norms,
    // coefficients), not implemented in this class.
    //   peclet finite > 0 : exact axial-diffusion solution (DEFAULT behaviour).
    //   peclet = +inf      : bare (axial-diffusion-free) solution.
    void setup_solution(Ttype peclet, Ttype max_root);
    void setup_solution(Ttype peclet); // reuse the current max_root (or keep-all default)
    void setup_bare_solution();
    void setup_qep_solution(Ttype peclet);
    void setup_fp_solution(Ttype peclet);

    // Finite-Péclet configuration (optional; sensible defaults set in the ctor).
    void set_fp_tolerance(Ttype rel_tol) { m_fp_rel_tol = rel_tol; invalidate_solution(); }
    void set_fp_max_iter(unsigned n)     { m_fp_max_iter = n; invalidate_solution(); }
    void set_fp_quad_panels(unsigned n)
    {
        if (n == 0) throw std::invalid_argument("set_fp_quad_panels: panels must be positive");
        m_fp_quad_panels = n;
        invalidate_solution();
    }
    // Margin of the per-mode interface-cap/full-disk Gauss-Jacobi rule's node
    // count (see m_fp_cap_quad_margin).
    void set_fp_cap_quad_margin(unsigned margin)
    {
        m_fp_cap_quad_margin = margin;
        invalidate_solution();
    }
    unsigned get_fp_cap_quad_margin() const { return m_fp_cap_quad_margin; }
    // Number of terms M of the sum in G_inf (eq. G_inf_split_asymptotic) used by
    // NusseltEvaluation::AsymptoticSplit. Read at evaluation time only, so the
    // solution stays valid.
    void set_nusselt_asymptotic_terms(unsigned terms)
    {
        if (terms == 0u) throw std::invalid_argument("set_nusselt_asymptotic_terms: at least one term is required");
        m_nusselt_asymptotic_terms = terms;
    }
    unsigned get_nusselt_asymptotic_terms() const { return m_nusselt_asymptotic_terms; }
    // The Gram backend is deliberately explicit: failures are never retried by
    // another implementation.
    void set_gram_method(GramMethod method) { m_gram_method = method; invalidate_solution(); }
    GramMethod get_gram_method() const { return m_gram_method; }
    void set_gram_oversampling(unsigned factor, unsigned margin)
    {
        if (factor == 0u) throw std::invalid_argument("set_gram_oversampling: factor must be positive");
        m_gram_oversampling_factor = factor;
        m_gram_oversampling_margin = margin;
        invalidate_solution();
    }
    void set_gram_order_check(bool enabled) { m_gram_order_check = enabled; invalidate_solution(); }
    // Hard safety floor on the Gram node count (N >= minimum_factor * K); see
    // FPGramGaussJacobiOptions::minimum_factor. Zero disables the floor and is
    // only valid in a deliberate study -- production must leave it at 2.
    void set_gram_minimum_factor(unsigned factor)
    {
        if (factor == 0u)
            throw std::invalid_argument("set_gram_minimum_factor: must be positive (0 disables the "
                                        "singularity floor and is only valid in a deliberate study)");
        m_gram_minimum_factor = factor;
        invalidate_solution();
    }
    unsigned get_gram_minimum_factor() const { return m_gram_minimum_factor; }

    // RHS load-vector backend is deliberately explicit, mirroring the Gram
    // backend selector: failures are never retried by another implementation.
    // RhsMethod::Representer requires GramMethod::GaussJacobiQR (see
    // rhs_method.h / finite_peclet_coefficients_gauss_jacobi.cpp); the
    // mismatch is checked at setup time, not here.
    void set_rhs_method(RhsMethod method) { m_rhs_method = method; invalidate_solution(); }
    RhsMethod get_rhs_method() const { return m_rhs_method; }
    // Margin added to the exact (n >= 1) / calibrated (n == 0) representer
    // quadrature order; see finite_peclet_rhs_representer.h's
    // FPRepresenterOptions::representer_margin.
    void set_rhs_representer_margin(unsigned margin) { m_rhs_representer_margin = margin; invalidate_solution(); }
    unsigned get_rhs_representer_margin() const { return m_rhs_representer_margin; }

    // Inner product used for the finite-Peclet inlet projection; deliberately
    // explicit, like the two selectors above, and with no fallback between the
    // two spaces. Weighted (L2_omega, the default) is the norm in which the
    // BARE modes are orthogonal; Radial (L2_r) is the one whose Gram
    // conditioning does not degrade with truncation, because the MODIFIED
    // modes are a Riesz basis of L2_r and not of L2_omega (omega vanishes at
    // the wall). See finite_peclet_coefficients_radial.h.
    //
    // ProjectionSpace::Radial supports both RHS backends but has no
    // ultraspherical Gram backend (that factor is built for the (1,n)
    // measure); that combination is rejected at setup time, not here.
    void set_projection_space(ProjectionSpace space) { m_projection_space = space; invalidate_solution(); }
    ProjectionSpace get_projection_space() const { return m_projection_space; }
    // String selector: "w" / "omega" / "weighted", or "r" / "radial"
    // (case-insensitive). Unknown strings throw; they never silently default.
    void set_projection_space(const std::string &space)
    {
        std::string key;
        key.reserve(space.size());
        for (char c : space)
            key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (key == "w" || key == "omega" || key == "weighted")
            set_projection_space(ProjectionSpace::Weighted);
        else if (key == "r" || key == "radial")
            set_projection_space(ProjectionSpace::Radial);
        else
            throw std::invalid_argument("set_projection_space: unknown projection space \"" + space
                                        + "\"; expected \"w\"/\"omega\"/\"weighted\" or \"r\"/\"radial\"");
    }

    // --- Exact finite-Péclet solution via the Quadratic Eigenvalue Problem. --
    // Independent reference solution: instead of deforming the radial basis
    // (setup_solution), it keeps the bare Pe-independent eigenbasis and solves the
    // coupled modal ODE system exactly, following Neuhauser et al. (2025), Sec. 2.4.
    // Every computation is delegated to qep_computations.{h,cpp}; this method only
    // sequences the calls and rescales the result.
    //
    // Self-contained: it prepares the bare mode set and inlet coefficients itself,
    // and does not disturb the state built by setup_solution.
    //
    // @param peclet   Péclet number (finite, strictly positive).
    // @param x_points axial coordinates.
    // @param r_points radial coordinates.
    // @param phi_points angular coordinates.
    // @return solution values, one per input point.
    std::vector<Ttype> get_QEP_solution_at_points(Ttype peclet,
                                                  const std::vector<Ttype> &x_points,
                                                  const std::vector<Ttype> &r_points,
                                                  const std::vector<Ttype> &phi_points);

    // Introspection.
    bool  using_axial_diffusion() const { return m_solution_method == SolutionMethod::QEP || m_solution_method == SolutionMethod::ModifiedRoots; }
    SolutionMethod get_solution_method() const { return m_solution_method; }
    Ttype get_peclet() const { return m_peclet; }
    // (Λ_nm, b_nm, b̃_nm, 𝒩²_fp, Ĉ_nm) per retained mode, in m_series_data order.
    std::vector<std::array<Ttype, 5>> get_modified_data() const;
    // Squared norm of the inlet projection in the space used by the active
    // setup, converted to the user's inlet scaling. Bare and QEP setups use
    // L2_omega; finite-Peclet setups use L2_omega or L2_r according to
    // m_projection_space. In either finite-Peclet space this is c^T b, with
    // the Gram matrix and load vector assembled in that same space.
    Ttype get_inlet_projection_square_norm() const;

    // Diameter-based local and fully developed Nusselt numbers for an
    // isothermal wall. Only retained axisymmetric modes contribute.
    std::vector<Ttype> get_nusselt_number(const std::vector<Ttype> &x_points) const;
    Ttype get_nusselt_number(Ttype x) const;
    Ttype get_fully_developed_nusselt_number() const;
    std::vector<std::array<Ttype, 5>> get_nusselt_mode_data() const;
    // Plain truncates the wall-gradient series at the retained block, whose sum
    // then converges only through exp(-Lam x) near the inlet. The split
    // evaluations (finite-Peclet path, x > 0) add the modes beyond the block with
    // their Bessel-limit amplitude, c R'(1) -> -2 sqrt(2 pi), which assumes the
    // unit internal inlet of the isothermal Graetz problem:
    //   Nu = -1/2 [sum_k c_k sigma_k e^{-Lam_k x} - 2 sqrt(2 pi) sum_{m>K} e^{-Lam_m x}] / bulk.
    // This is the plan's N = -2 sqrt(2 pi) G + sum_k delta_k e^{-Lam_k x} with the
    // retained part of G cancelled. The remainder is sum_k delta_k over the block,
    // so the split is only as good as the retained c_k sigma_k: with the Weighted
    // projection the last part of the block departs from the limit (the weight
    // vanishes at the wall), with the Radial projection it does not.
    // AsymptoticSplit evaluates N = -2 sqrt(2 pi) G + sum_k delta_k e^{-Lam_k x} with
    // G = G_0 + G_inf: G_0 of eq. G_split over the retained block and G_inf of
    // eq. G_inf_split_asymptotic with its sum truncated at M terms
    // (set_nusselt_asymptotic_terms). Where the omitted modes are below round-off,
    // x >= ln(1/eps)/(Lam_max - Lam_min), it returns the Plain value, which it
    // equals there to round-off.
    // extra_rates (SpectralSplit only): precomputed tail rates, e.g. from
    // get_nusselt_tail_rates, validated against the smallest requested x.
    std::vector<Ttype> get_nusselt_number(const std::vector<Ttype> &x_points, NusseltEvaluation evaluation,
                                          const std::vector<Ttype> *extra_rates = nullptr) const;
    Ttype get_nusselt_number(Ttype x, NusseltEvaluation evaluation) const;
    // SpectralSplit tail rates sufficient for every x >= x_min.
    std::vector<Ttype> get_nusselt_tail_rates(Ttype x_min) const;
    // ln(1/tol)/Lam_max over the retained axisymmetric modes: below this x the
    // first omitted exponential exceeds tol. It bounds the exponential truncation
    // of the Plain sum only, not the convergence of the coefficients themselves.
    Ttype nusselt_converged_x_min(Ttype tol) const;

    // --- Blurriness of the section at x (see the protected block above). -----
    // x is the dimensionless axial coordinate of the series (x / (Pe R)); x = +inf
    // is allowed and returns 1 (with the tail) exactly. include_truncation_tail
    // adds the Parseval defect D_K to numerator and denominator, which is the
    // exact blurriness of the truncated field and the value a quadrature of the
    // series over the disk returns; it requires the L2_omega projection (bare
    // solution or ProjectionSpace::Weighted). Without the tail the ratio is the
    // blurriness of the truncated basis alone: 0 at x = 0 and 1 at x = inf in
    // every projection space. Not available for the QEP setup, whose modes do
    // not carry a single decay rate each.
    Ttype get_blurriness(Ttype x, BlurrinessTail tail = BlurrinessTail::Asymptotic) const;
    std::vector<Ttype> get_blurriness(const std::vector<Ttype> &x_points,
                                      BlurrinessTail tail = BlurrinessTail::Asymptotic) const;
    // bool form kept for the first callers: true -> Constant, false -> Dropped.
    Ttype get_blurriness(Ttype x, bool include_truncation_tail) const
    {
        return get_blurriness(x, include_truncation_tail ? BlurrinessTail::Constant : BlurrinessTail::Dropped);
    }
    std::vector<Ttype> get_blurriness(const std::vector<Ttype> &x_points, bool include_truncation_tail) const
    {
        return get_blurriness(x_points, include_truncation_tail ? BlurrinessTail::Constant : BlurrinessTail::Dropped);
    }
    // B_K(0) = sqrt(D_K / Q_inf): the value the truncated field gives at the
    // inlet, i.e. the resolution floor of the truncation.
    Ttype get_blurriness_truncation_floor() const;
    // Far-field constant psi_inf in the user's units (the zero-eigenvalue mode;
    // the wall value for a Dirichlet basis).
    Ttype get_far_field_value() const;
    // || f ||^2_omega and || f - psi_inf ||^2_omega of the user's inlet (exact,
    // no truncation): the latter is the denominator of the blurriness.
    Ttype get_inlet_square_norm() const;
    Ttype get_inlet_far_field_square_norm() const;

    // Others
    void compute_and_fix_coefficients();
    void fix_coefficients();
    void free_coefficients();
    void save_coefficients(const std::string &);

    // Pure virtual: must be implemented by derived classes
    virtual void print_info() = 0;
};

#ifndef CD_BASE_SOLUTION_TPP
#include "CDBaseSolution.tpp"
#endif

#endif
