import numpy as np
from anastruct import SystemElements


def trussbridge(Ediag, Adiag, Ebot, Abot, Etop, Atop, p, w_tri = 4, h_tri=2,num_tri=6, disp=False):
    """
    Calculate displacement of middle point in bridge truss

    Args:
         Ediag (list): list of Young's modulus for each pair of diagonal trusses (Pa)
         Adiag (list): list of cross-sectional area for each pair of diagonal trusses (m2)
         Ebot (list): list of Young's modulus for each bottom truss (Pa)
         Abot (list): list of cross-sectional area for each bottom truss (m2)
         Etop (list): list of Young's modulus for each top truss (Pa)
         Atop (list): list of cross-sectional area for each top truss (m2)t
         p (list): list of force applied on the top nodes (N)
         num_tri (int): number of triangles
         disp (bool): display image or not

    """
    Ediag = np.array(Ediag)
    Adiag = np.array(Adiag)
    Ebot = np.array(Ebot)
    Abot = np.array(Abot)
    Etop = np.array(Etop)
    Atop = np.array(Atop)
    EAdiag = Ediag*Adiag
    EAbot = Ebot * Abot
    EAtop = Etop * Atop

    ss = SystemElements()

    # Triangle coord
    x_base = np.arange(0,num_tri+1) * w_tri
    x_top = np.arange(0,num_tri) * w_tri + h_tri
    y = np.ones(num_tri) * h_tri

    # Create 6 triangles
    for i in range(num_tri):
        p1 = [x_base[i],0]
        p2 = [x_top[i],y[i]]
        p3 = [x_base[i+1],0]
        ss.add_truss_element(location=[p1, p2], EA=EAdiag[i])
        ss.add_truss_element(location=[p2, p3], EA=EAdiag[i])
        ss.add_truss_element(location=[p1, p3], EA=EAbot[i])

    # Create 5 horizontal trusses
    for i in range(num_tri-1):
        ss.add_truss_element(location=[[x_top[i],y[i]], [x_top[i+1],y[i+1]]], EA=EAtop[i])

    # Create support
    ss.add_support_hinged(node_id=1)
    ss.add_support_roll(node_id=13, direction=2)

    # Create Load
    loadnode = [2,4,6,8,12]
    for index, point in enumerate(loadnode):
        ss.point_load(node_id=point,Fy=p[index])
        ss.point_load(node_id=point,Fy=p[index])
        ss.point_load(node_id=point,Fy=p[index])
        ss.point_load(node_id=point,Fy=p[index])

    ss.solve()
    disp7 = ss.get_node_displacements(node_id=7)

    if disp is True:
        ss.show_axial_force()
        ss.show_displacement(factor=10)

    return disp7


if __name__ == "__main__":
    num_tri = 6
    Ediag = np.ones(shape=num_tri) * 2e11
    Adiag = np.ones(shape=num_tri) * 3e-4
    Ebot = np.ones(shape=num_tri) * 2.2e11
    Abot = np.ones(shape=num_tri) * 3.5e-4
    Etop = np.ones(shape=num_tri-1) * 2.2e11
    Atop = np.ones(shape=num_tri-1) * 3.5e-4
    p = np.ones(shape=num_tri) * -1e4

    mid_disp = trussbridge(Ediag,Adiag,Ebot,Abot,Etop,Atop,p,disp=True)