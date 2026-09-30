#ifndef GRAM_DIAGNOSTICS_H
#define GRAM_DIAGNOSTICS_H

template <typename Ttype>
struct GramBlockDiagnostics
{
    unsigned angular_index = 0;
    unsigned mode_count = 0;
    unsigned resolution = 0;
    Ttype min_abs_diagonal_R = 0;
    Ttype max_abs_diagonal_R = 0;
    Ttype condition_estimate = 0;
    Ttype order_consistency = 0; // order-difference estimator, never a bound
    bool order_check_ran = false;
};

#endif
