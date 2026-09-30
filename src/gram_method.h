#ifndef GRAM_METHOD_H
#define GRAM_METHOD_H

// The finite-Peclet Gram assembly is deliberately selectable.  There is no
// fallback between implementations: a failing selected method must be visible
// to its caller.
enum class GramMethod
{
    GaussJacobiQR = 0,
    Ultraspherical = 1
};

#endif
