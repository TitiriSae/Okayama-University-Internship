#aled

#CONVERGENCE_EPS = 1e-8

#from dpca_test import algebraic_connectivity_plot
import matplotlib.pyplot as plt

#fy

#from report 7

ac_y_7 = [
    (8, 1),
    (2.585786437626905, 4),
    (0.585786437626905, 13),
    (1, 21),
    (0.15224093497742655, 46)
]

#for report 8, seed = 22

ac_y_8_22 = [
    (0.2508824522534783, 28),
    (0.3506335021057974, 8),
    (0.9402549083341685, 0),
    (1.4659298032772674, 9),
    (1.6277186767309857, 11),
    (2.835752061539789, 0),
    (4.0, 0),
    (6, 0)
]

#for report 8, seed = 23

ac_y_8_23 = [
    (0.21368226527104744, 8),
    (0.24669938645338058, 2),
    (0.8062884009828438, 0),
    (1.2078389787751613, 0),
    (2.6360404847412333, 0),
    (2.7639320225002093, 0),
    (4.585786437626906, 1),
    (6.0, 0)
]

#for report 8, seed = 24

ac_y_8_24 = [
    (0.2434017461399327, 2),
    (0.3186693563950218, 18),
    (0.8725324834354761, 1),
    (1.4384471871911708, 0),
    (2.07609066504979, 0),
    (3.438447187191169, 0),
    (4.0, 0),
    (6.0, 0)
]



#fixed seed 22_X

ac_y_8_22_X = [
    (0.16671700823775712, 1),
    (0.3186693563950216, 1),
    (0.7574727579393633, 0),
    (1.4435126843986152, 0),
    (1.8926644174974205, 0),
    (2.7381977547400305, 0),
    (4.0, 0),
    (6.0, 0)
]

#fixed seed 22_1

ac_y_8_22_1 = [
    (0.15224093497742663, 1),
    (0.37565089916577926, 2),
    (0.35424868893541, 1),
    (0.8793434264453559, 1),
    (2.1442274933640135, 0),
    (3.186393497351668, 1),
    (5.0, 0),
    (6.0, 0)
]

#fixed seed 22_13

ac_y_8_22_13 = [
    (0.22428714426378596, 1),
    (0.4149280933658772, 1),
    (0.7981805099035858, 1),
    (1.3679746201566327, 0),
    (2.413094015707478, 0),
    (3.267949192431122, 0),
    (4.0, 0),
    (6.0, 0)
]










#from report 7

ac_z_7 = [
    (8, 1),
    (2.585786437626905, 13),
    (0.585786437626905, 42),
    (1, 73),
    (0.15224093497742655, 159)
]

#for report 8, seed = 22

ac_z_8_22 = [
    (0.2508824522534783, 134),
    (0.3506335021057974, 112),
    (0.9402549083341685, 45),
    (1.4659298032772674, 41),
    (1.6277186767309857, 32),
    (2.835752061539789, 18),
    (4.0, 34),
    (6,  26)
]

#for report 8, seed = 23

ac_z_8_23 = [
    (0.21368226527104744, 150),
    (0.24669938645338058, 118),
    (0.8062884009828438, 51),
    (1.2078389787751613, 38),
    (2.6360404847412333, 17),
    (2.7639320225002093, 18),
    (4.585786437626906, 10),
    (6.0, 7)
]

#for report 8, seed = 24

ac_z_8_24 = [
    (0.2434017461399327, 117),
    (0.3186693563950218, 117),
    (0.8725324834354761, 64),
    (1.4384471871911708, 30),
    (2.07609066504979, 26),
    (3.438447187191169, 16),
    (4.0, 12),
    (6.0, 7)
]



#fixed seed 22_X

ac_z_8_22_X = [
    (0.16671700823775712, 159),
    (0.3186693563950216, 118),
    (0.7574727579393633, 56),
    (1.4435126843986152, 32),
    (1.8926644174974205, 27),
    (2.7381977547400305, 18),
    (4.0, 13),
    (6.0, 6)
]

#fixed seed 22_1

ac_z_8_22_1 = [
    (0.15224093497742663, 167),
    (0.37565089916577926, 104),
    (0.35424868893541, 97),
    (0.8793434264453559, 55),
    (2.1442274933640135, 23),
    (3.186393497351668, 17),
    (5.0, 8),
    (6.0, 6)
]

#fixed seed 22_13

ac_z_8_22_13 = [
    (0.22428714426378596, 129),
    (0.4149280933658772, 98),
    (0.7981805099035858, 52),
    (1.3679746201566327, 29),
    (2.413094015707478, 23),
    (3.267949192431122, 15),
    (4.0, 13),
    (6.0, 6)
]


def algebraic_connectivity_plot(scatter_plot, ax_y_label, option, add_title=""):
    """
    Display a scatter plot of the points set in parameters.
    
    return:
        None
    """

    x, y = [], []
    for point in scatter_plot:
        x.append(point[0])
        y.append(point[1])

    if option == 1:
        plt.scatter(x, y, s=40, color="red")
    elif option == 0:
        plt.scatter(x, y, s=20, color="blue")

    plt.xlabel("Algebraic connectivity")
    plt.ylabel(ax_y_label)
    plt.title(f"Scatter plot of the {ax_y_label} as a function of Algebraic connectivity {add_title}")

    plt.grid(True)
    #plt.show()


if __name__ == "__main__":

    scatter_plot_y_not_fixed = ac_y_7 + ac_y_8_22 + ac_y_8_23 + ac_y_8_24
    scatter_plot_z_not_fixed = ac_z_7 + ac_z_8_22 + ac_z_8_23 + ac_z_8_24
    scatter_plot_y_fixed = ac_y_8_22_X + ac_y_8_22_1 + ac_y_8_22_13
    scatter_plot_z_fixed = ac_z_8_22_X + ac_z_8_22_1 + ac_z_8_22_13
    scatter_plot_y = scatter_plot_y_not_fixed + scatter_plot_y_fixed
    scatter_plot_z = scatter_plot_z_not_fixed + scatter_plot_z_fixed


    algebraic_connectivity_plot(scatter_plot_y_not_fixed, "Best value of T_Y", 0, "| random u_p(0)")
    plt.show()
    algebraic_connectivity_plot(scatter_plot_z_not_fixed, "Best value of T_Z", 0, "| random u_p(0)")
    plt.show()

    algebraic_connectivity_plot(scatter_plot_y_fixed, "Best value of T_Y", 1, "| fixed u_p(0)")
    plt.show()
    algebraic_connectivity_plot(scatter_plot_z_fixed, "Best value of T_Z", 1, "| fixed u_p(0)")
    plt.show()

    algebraic_connectivity_plot(scatter_plot_y_not_fixed, "Best value of T_Y", 0)
    algebraic_connectivity_plot(scatter_plot_y_fixed, "Best value of T_Y", 1)
    plt.show()

    algebraic_connectivity_plot(scatter_plot_z_not_fixed, "Best value of T_Z", 0)
    algebraic_connectivity_plot(scatter_plot_z_fixed, "Best value of T_Z", 1)
    plt.show()





