#include <functional>

void gaussian_integration(
    const std::function<void(double, double &)> &f,
    const std::vector<double> weights,
    const std::vector<double> nodes,
    double &result)
{
    unsigned MAX_DIM = (weights.size() < nodes.size()) ? weights.size() : nodes.size();

    result = 0.0;

    // double new_term = 0.0;
    double f_node;
    for (unsigned i = 0; i < MAX_DIM; i++)
    {
        // new_term = weights[i] * f(nodes[i]);
        // result += new_term;
        f(nodes[i], f_node);
        result += weights[i] * f_node;
        // std::cout << "i = " << i << ", w = " << weights[i] << ", node = " << nodes[i] << ", f(node) = " << f(nodes[i]) << std::endl;
    }
}

void gaussian_integration(
    const std::function<void(long double, long double &)> &f,
    const std::vector<long double> weights,
    const std::vector<long double> nodes,
    long double &result)
{
    unsigned MAX_DIM = (weights.size() < nodes.size()) ? weights.size() : nodes.size();

    result = 0.0;

    // long double new_term = 0.0;
    long double f_node;
    for (unsigned i = 0; i < MAX_DIM; i++)
    {
        // new_term = weights[i] * f(nodes[i]);
        // result += new_term;
        f(nodes[i], f_node);
        result += weights[i] * f_node;
        // std::cout << "i = " << i << ", w = " << weights[i] << ", node = " << nodes[i] << ", f(node) = " << f(nodes[i]) << std::endl;
    }
}
