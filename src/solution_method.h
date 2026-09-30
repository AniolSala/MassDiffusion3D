#ifndef SOLUTION_METHOD_H
#define SOLUTION_METHOD_H

// The active representation of a solution object.  Uninitialized is the only
// state in which evaluation and active-data access are invalid.
enum class SolutionMethod
{
    Uninitialized,
    Bare,
    QEP,
    ModifiedRoots
};

#endif
