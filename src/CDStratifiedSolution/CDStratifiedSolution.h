#ifndef CD_STRATIFIED_SOLUTION_H
#define CD_STRATIFIED_SOLUTION_H

#include <sstream>

#include "../CDBaseSolution/CDBaseSolution.h"

template <typename Ttype>
class CDStratifiedSolution : public CDBaseSolution<Ttype>
{
protected:
    void set_roots(std::vector<std::vector<Ttype>> &) const override;
    void set_norms(std::vector<std::vector<Ttype>> &) const override;
    void set_pseudo_products_matrix(std::vector<std::vector<std::vector<Ttype>>> &) override;
    void compute_coefficients() override;
    void compute_coefficients_fp(const FPRadialTable<Ttype> &) override;
    void draw_step_solution_terminal(std::ostringstream &) override;
    void require_nusselt_number_defined(const char *caller) const override
    {
        throw std::logic_error(std::string(caller)
            + ": CDStratifiedSolution models mass transport through zero-flux walls; "
              "the Nusselt number is not defined for this configuration.");
    }
    // Stratified profiles use zero-flux (Neumann) walls — the CDBaseSolution
    // default — so wall_condition() is not overridden here.

public:
    // Constructors
    CDStratifiedSolution(const std::vector<Ttype> &zi, const std::vector<Ttype> &ui, unsigned short verbose_level);
    CDStratifiedSolution(const std::vector<Ttype> &zi, const std::vector<Ttype> &ui) : CDStratifiedSolution(zi, ui, 1) {}

    // Override pure-virtual from CDBaseSolution
    void print_info() override;
};

#include "CDStratifiedSolution.tpp"

#endif
