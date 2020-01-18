"""Triangular Wing Tip Angle Checker

Checks the difference in sweep between the leading and trailing edges.

Last updated: 12/03/2019
"""
__author__ = "Tim Jim"
__version__ = '0.0.1'

import argparse
import numpy as np


def sweep_diff(le_sweep, root_chord, semi_area=0.00165529, disp=False):
    """Calculate the tip angle of a triangular wing.

    Inputs can be numpy arrays.

    Args:
        le_sweep (float/np.array): Leading edge sweep.
        root_chord (float/np.array): Root chord.
        semi_area (float/np.array): Area of he half wing.

    Returns:
        (float/np.array): the difference between the LE and TE sweep (tip angle).
    """
    if disp is True:
        print(f'Leading edge sweep: {le_sweep} degrees')
        print(f'Root chord: {root_chord} m')
        print(f'Semi-area: {semi_area} m^2')
    else:
        pass

    le_length = 2 * semi_area / (root_chord * np.cos(np.radians(le_sweep)))
    te_length = np.sqrt(root_chord ** 2 + le_length ** 2
                        - 2 * root_chord * le_length * np.sin(np.radians(le_sweep)))

    tip_angle = np.degrees(np.arcsin(root_chord * np.cos(np.radians(le_sweep))
                                     / te_length))

    if disp is True:
        print(f'Tip angle: {tip_angle} degrees')
    else:
        pass

    return tip_angle


def min_angle_violated(angle, min_angle = 7.0,disp=False):
    """Check if minimum angle is violated.

    Inputs cans be numpy arrays.

    Args:
        min_angle (float/np.array): The minimum allowed angle.
        angle (float/np.array): Angle to check.

    Returns:
        (bool/np.array): True if angle is less than minimum angle.
    """
    not_violated = angle > min_angle
    violated = ~ not_violated
    if np.any(violated) and disp is True:
        print(f'Minimum angle of {min_angle} degrees is violated.')
    return not_violated


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=__doc__,
    #                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('le_sweep', type=float,
    #                     help="Leading edge sweep in degrees.")
    # parser.add_argument('root_chord', type=float,
    #                     help="Root chord in m.")
    # parser.add_argument('--semi_area', type=float, default=0.00165529,
    #                     help="Area of the half wing in m^2. Defaults to 0.00165529 m^2.")
    # parser.add_argument('--min_angle', type=float, default=7.0,
    #                     help="Minimum allowed tip angle in degrees. Defaults to 7.")
    #
    # args = parser.parse_args()
    #
    # angle = sweep_diff(args.le_sweep, args.root_chord, args.semi_area)
    #
    # violated = min_angle_violated(args.min_angle, angle)
    dat = np.loadtxt('../../innout/Timnext5.csv', skiprows=1, delimiter=',')
    tip_angle = sweep_diff(dat[:, 2], dat[:, 4], 0.00165529)
    stat = min_angle_violated(tip_angle,7)
    print(stat)