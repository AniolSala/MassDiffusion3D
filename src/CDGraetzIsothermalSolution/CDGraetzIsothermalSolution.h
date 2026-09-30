#ifndef CD_GRAETZ_ISOTHERMAL_SOLUTION_H
#define CD_GRAETZ_ISOTHERMAL_SOLUTION_H

#include <sstream>
#include <vector>

#include "../CDBaseSolution/CDBaseSolution.h"

template <typename Ttype>
class CDGraetzIsothermalSolution : public CDBaseSolution<Ttype>
{
public:
    CDGraetzIsothermalSolution(Ttype T0, Ttype T_wall,
                                unsigned short verbose_level = 1);

    void print_info() override;

protected:
    void set_roots(std::vector<std::vector<Ttype>> &roots) const override;
    void set_norms(std::vector<std::vector<Ttype>> &norms) const override;
    void set_pseudo_products_matrix(std::vector<std::vector<std::vector<Ttype>>> &) override;
    void compute_coefficients() override;
    void compute_coefficients_fp(const FPRadialTable<Ttype> &) override;
    void draw_step_solution_terminal(std::ostringstream &os) override;

    bool is_zero_eigenvalue_mode(unsigned, unsigned) const override { return false; }

    // Isothermal Graetz has fixed-scalar (Dirichlet) walls.
    WallCondition wall_condition() const override { return WallCondition::Dirichlet; }

    // --- Constant-function terms of the inlet-projection norm. ---------------
    // The base-class closed forms assume the constant is a retained mode, which
    // a Dirichlet wall denies. Here they are available for a different reason:
    // this problem's INTERNAL inlet datum is identically one (theta = 1 at the
    // inlet, by the T_wall / T0 normalisation set in the constructor), so the
    // constant's projection IS the inlet's projection, P1 = P f~, and both
    // quantities collapse onto the stored contraction. The unscaling in
    // CDBaseSolution::get_inlet_projection_square_norm then reduces to the
    // single factor (alpha + beta)^2 = m_min_ui^2 = T0^2, which is right because
    // the user's inlet datum is the constant T0.
    Ttype constant_projection_square_norm() const override
    {
        return this->m_inlet_projection_square_norm;
    }
    Ttype constant_projection_inlet_cross() const override
    {
        return this->m_inlet_projection_square_norm;
    }

    // --- Blurriness inlet data. ---------------------------------------------
    // The INTERNAL inlet datum is identically one (theta = 1), not the layered
    // m_zi / m_ui dummy the base constructor received, so || f~ ||^2_omega is
    // the weighted disk area and the flow-weighted mean is one. m_constant_term
    // holds the far field (zero) here, not the inlet mean, hence the override.
    Ttype inlet_square_norm_scaled() const override
    {
        return static_cast<Ttype>(1.57079632679489661923132169163975144L);
    }
    Ttype inlet_flow_weighted_mean_scaled() const override { return static_cast<Ttype>(1); }
    // Wall mismatch of the isothermal inlet: D_K ~ beta_K^-4/3 for the bare modes
    // (Airy turning point at the wall), beta_K^-2 for the modified ones (theory,
    // Sec. pencil_orthogonality and fig. parseval_defect).
    Ttype truncation_defect_exponent() const override
    {
        return this->m_solution_method == SolutionMethod::ModifiedRoots
                   ? static_cast<Ttype>(2) : static_cast<Ttype>(4.0L / 3.0L);
    }

private:
    Ttype m_T0;
    Ttype m_T_wall;
};

#ifndef CD_GRAETZ_ISOTHERMAL_SOLUTION_TPP
#include "CDGraetzIsothermalSolution.tpp"
#endif

#endif
