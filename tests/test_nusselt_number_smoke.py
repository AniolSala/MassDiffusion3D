"""Python binding smoke tests for the isothermal Nusselt API.

Run from the repository root with ``python3 tests/test_nusselt_number_smoke.py``.
"""

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import MassDiffusion3Dcpp as md


class NusseltSmoke(unittest.TestCase):
    def test_stratified_rejected(self):
        solution = md.CDStratifiedSolution([0.0], [1.0, 0.0], 0)
        with self.assertRaisesRegex(RuntimeError, "mass transport"):
            solution.get_nusselt_number(0.1)

    def test_classical_asymptote_and_bulk_identity(self):
        solution = md.CDGraetzIsothermalSolution(1.0, 0.0, 0)
        solution.set_max_root(25.0)
        solution.setup_bare_solution()
        beta = solution.get_bare_root_catalog()[0][0]
        expected = beta * beta / 2.0
        self.assertAlmostEqual(solution.get_fully_developed_nusselt_number(), expected, places=9)
        self.assertAlmostEqual(solution.get_nusselt_number(math.inf), expected, places=9)
        self.assertEqual(solution.get_nusselt_number([0.1]), [solution.get_nusselt_number(0.1)])
        projection = sum(row[2] * math.sqrt(2.0 * math.pi) * row[4]
                         for row in solution.get_nusselt_mode_data())
        self.assertAlmostEqual(projection, solution.get_inlet_projection_square_norm(), places=8)
        with self.assertRaisesRegex(RuntimeError, "finite-Peclet"):
            solution.get_nusselt_number(0.1, md.NusseltEvaluation.SpectralSplit)

    def test_split_evaluations(self):
        solution = md.CDGraetzIsothermalSolution(1.0, 0.0, 0)
        solution.set_max_root(25.0)
        solution.setup_fp_solution(5.0)
        xs = [0.05, 1.0]
        plain = solution.get_nusselt_number(xs)
        rates = solution.get_nusselt_tail_rates(min(xs))
        spectral = solution.get_nusselt_number(xs, md.NusseltEvaluation.SpectralSplit, rates)
        self.assertEqual(spectral, solution.get_nusselt_number(xs, md.NusseltEvaluation.SpectralSplit))
        slug = solution.get_nusselt_number(xs, md.NusseltEvaluation.SlugSplit)
        self.assertAlmostEqual(spectral[1], plain[1], places=10)
        self.assertLess(abs(slug[0] / spectral[0] - 1.0), 1e-4)
        with self.assertRaises(ValueError):
            solution.get_nusselt_number(0.0, md.NusseltEvaluation.SlugSplit)


if __name__ == "__main__":
    unittest.main()
