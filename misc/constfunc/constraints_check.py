"""Two-section Wing Area and Tip Angle Checker

Calculate the areas and tip angle of a two section wing based on the
11 variable model.

Last updated: 07/12/2019
"""
__author__ = "Tim Jim"
__version__ = '1.0.0'


import argparse
import numpy as np


def triangular_tip_angle(le_sweep, root_chord, section_area):
    """Calculate the tip angle of a triangular wing section.

    i.e. a section defined with a single chord, area, and LE sweep.

    Inputs can be numpy arrays.

    Args:
        le_sweep (float/np.array): Leading edge sweep of section.
        root_chord (float/np.array): Chord of section.
        section_area (float/np.array): Section area of he half wing.

    Returns:
        (float/np.array): the difference between the LE and TE sweep (tip angle).
    """
    le_length = 2 * section_area / (root_chord * np.cos(np.radians(le_sweep)))
    te_length = np.sqrt(root_chord**2 + le_length**2
                        - 2 * root_chord * le_length * np.sin(np.radians(le_sweep)))

    tip_angle = np.degrees(np.arcsin(root_chord * np.cos(np.radians(le_sweep))
                                     / te_length))

    return tip_angle


def calc_areas(proj_span_1, chord_1, dihedral_1, chord_2, dihedral_2,
               total_proj_area):
    """Calculate the projected and actual areas for a 2-section wing.

    Assumes section 1 is a trapezium and section two is triangular.
    The total projected area S = proj_area_1 + proj_area_2.
    The actual areas are calculated as a projection onto the dihedral plane.

    Inputs cans be numpy arrays.

    Args:
        proj_span_1 (float/np.array): Section 1 projected semispan
            proj_span_1 in m."
        chord_1 (float/np.array): Section 1 chord chord_1 in m.
        dihedral_1 (float/np.array): Section 1 dihedral dihedral_1 in degrees.
        chord_2 (float/np.array): Section 2 chord chord_2 in m.
        dihedral_2 (float/np.array): Section 2 dihedral dihedral_2 in degrees.
        total_proj_area (float/np.array): Allowable total projected
            semi-area, S, of the half wing in m^2.

    Returns:
        proj_area_1 (float/np.array): the projected area of section 1.
        area_1 (float/np.array): the actual area of section 1.
        proj_area_2 (float/np.array): the projected area of section 2.
        area_2 (float/np.array): the actual area of section 2.
    """

    # Assuming 0 alpha, calculate projected areas
    # Section 1 projected area modelled as a trapezium, section 2 as a triangle
    proj_area_1 = (chord_1 + chord_2) / 2 * proj_span_1
    proj_area_2 = total_proj_area - proj_area_1
    # proj_span_2 = 2 * proj_area_2 / wing.XSec_1.tip_chord  # for calculation check

    # Area projected as trapezium/triangle from X-Y plane onto dihedral planes
    area_1 = proj_area_1 / np.cos(np.radians(dihedral_1))
    area_2 = proj_area_2 / np.cos(np.radians(dihedral_2))

    return proj_area_1, area_1, proj_area_2, area_2


def min_max_satisfied(vals, min_val=None, max_val=None, disp=True):
    """Check if values are inside range.

    If only a max or min value check is required, the other can be set
    as None.
    Inputs cans be numpy arrays.

    Args:
        vals (float/np.array): Values to check.
        min_val (float, optional): The minimum allowed value. Default is
            None.
        max_val (float, optional): The maximum allowed value. Default is
            None.

    Returns:
        (bool/np.array): True if vals inside range.
    """
    if min_val is not None and max_val is not None:
        min_satisfied = vals >= min_val
        max_satisfied = vals <= max_val
    elif min_val is not None:
        min_satisfied = vals >= min_val
        max_satisfied = True
    elif max_val is not None:
        min_satisfied = True
        max_satisfied = vals <= max_val
    else:
        msg = 'Both min_val and max_val cannot be None. Set a constraint.'
        raise ValueError(msg)

    # Only True where both min and max satisfied.
    constraints_satisfied = min_satisfied & max_satisfied
    if min_val is not None and not np.all(min_satisfied) and disp==True:
        print(f'Minimum constraint {min_val} is violated.')
    if max_val is not None and not np.all(max_satisfied) and disp==True:
        print(f'Maximum constraint {max_val} is violated.')

    return constraints_satisfied


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------
if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description=__doc__,
    #                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('proj_span_1', type=float,
    #                     help="Section 1 projected semispan proj_span_1 in m.")
    # parser.add_argument('chord_1', type=float, help="chord_1 in m.")
    # parser.add_argument('dihedral_1', type=float,
    #                     help="Section 1 dihedral dihedral_1 in degrees.")
    # parser.add_argument('chord_2', type=float, help="chord_2 in m.")
    # parser.add_argument('dihedral_2', type=float,
    #                     help="Section 2 dihedral dihedral_2 in degrees.")
    # parser.add_argument('le_sweep_2', type=float,
    #                     help="Section 2 leading edge sweep in degrees.")
    # parser.add_argument('--total_area', type=float, default=0.00165529,
    #                     help="Allowable total projected semi-area, S, "
    #                          "of the half wing in m^2, where S = S_1 + S_2."
    #                          "Defaults to 0.00165529 m^2.")
    # parser.add_argument('--min_angle', type=float, default=7.0,
    #                     help="Minimum allowed tip angle in degrees. Defaults to 7.")
    #
    # args = parser.parse_args()

    dat = np.loadtxt('../../innout/tim/nextpoints.csv', skiprows=1, delimiter=',')

    proj_area_1, area_1, proj_area_2, area_2 = calc_areas(dat[:,6],dat[:,4],dat[:,3],dat[:,7],dat[:,9],
                                                                                total_proj_area=0.00165529)

    print(f'proj_area_1: {proj_area_1}  m^2')
    print(f'area_1: {area_1}  m^2')
    print(f'proj_area_2 :{proj_area_2}  m^2')
    print(f'area_2: {area_2} m^2')

    s1_min = 0.3 * 0.00165529
    s1_max = 0.9 * 0.00165529
    s1_satisfied = min_max_satisfied(proj_area_1, min_val=s1_min, max_val=s1_max)

    # print(f'Leading edge sweep: {args.le_sweep_2} degrees')
    # print(f'Chord: {args.chord_2} m')
    tip_angle = triangular_tip_angle(dat[:,8], dat[:,7], area_2)
    print(f'Tip angle: {tip_angle} degrees')
    tip_satisfied = min_max_satisfied(tip_angle, min_val=7)
    print(tip_angle)