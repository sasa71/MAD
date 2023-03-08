import matplotlib.pyplot as plt
import networkx

def show_results(G):
    # Print the first five results
    print("Graph")
    for n1,n2 in list(G.edges())[:]:
        print("Ollivier-Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2]["ricciCurvature"]))
def r_file(G):
    a=[]
    for n1,n2 in list(G.edges())[:]:
        a.append(G[n1][n2]["ricciCurvature"])
    return a
def histogram(G):
    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = networkx.get_edge_attributes(G, "ricciCurvature").values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures")
    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = networkx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights ")

    plt.tight_layout()
    plt.show()
