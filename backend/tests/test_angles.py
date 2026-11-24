import numpy as np
import pytest

from app.biomech.angles import (angular_velocity, joint_angle, line_angle, line_to_line_angle,
                                segment_angle_vs_vertical)


def test_right_angle():
    assert joint_angle([1, 0], [0, 0], [0, 1]) == pytest.approx(90.0)


def test_straight_and_folded():
    assert joint_angle([0, 1], [0, 0], [0, -1]) == pytest.approx(180.0)
    assert joint_angle([1, 0], [0, 0], [2, 0]) == pytest.approx(0.0)


def test_known_angles():
    assert joint_angle([1, 0], [0, 0], [1, 1]) == pytest.approx(45.0)
    assert joint_angle([1, 0], [0, 0], [-1, np.sqrt(3)]) == pytest.approx(120.0)


def test_vectorised_and_degenerate():
    a = np.array([[1, 0], [0, 1]])
    b = np.zeros((2, 2))
    c = np.array([[0, 1], [0, -1]])
    assert np.allclose(joint_angle(a, b, c), [90.0, 180.0])
    assert np.isnan(joint_angle([0, 0], [0, 0], [1, 0]))


def test_segment_vs_vertical():
    assert segment_angle_vs_vertical([0, 0], [0, 1]) == pytest.approx(0.0)
    assert segment_angle_vs_vertical([0, 0], [1, 1]) == pytest.approx(45.0)
    assert segment_angle_vs_vertical([0, 0], [-1, 1]) == pytest.approx(-45.0)
    assert segment_angle_vs_vertical([0, 0], [-1, 1], signed=False) == pytest.approx(45.0)


def test_lines():
    assert line_angle([0, 0], [1, 1]) == pytest.approx(45.0)
    assert line_to_line_angle([0, 0], [1, 0], [0, 0], [1, 1]) == pytest.approx(45.0)
    assert line_to_line_angle([0, 0], [1, 0], [0, 0], [-1, 0.0]) == pytest.approx(0.0)


def test_angular_velocity():
    ang = np.array([0.0, 10.0, 20.0, 30.0])
    assert np.allclose(angular_velocity(ang, 100.0), [1000.0] * 4)
    wrapped = np.array([170.0, -170.0])  # +20° across the ±180 seam
    assert angular_velocity(wrapped, 10.0)[0] == pytest.approx(200.0)
