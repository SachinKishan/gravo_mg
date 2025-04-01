

#include <iostream>
#include <unordered_set>

#include<gravomg/multigrid_solver.h>

#include<igl/readOBJ.h>

#include <unsupported/Eigen/SparseExtra>


void savePointCloudToOBJ(const Eigen::MatrixXd& points, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Loop through each point and write it as a vertex in the .obj format
    for (int i = 0; i < points.rows(); ++i) {
        outFile << "v " << points(i, 0) << " " << points(i, 1) << " " << points(i, 2) << "\n";
    }

    outFile.close();
    std::cout << "Point cloud saved to " << filename << std::endl;
}

Eigen::MatrixXi createNeighborMatrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    // Number of vertices
    int numVertices = V.rows();

    // Create a vector of sets to store neighbors
    std::vector<std::set<int>> neighbors(numVertices);

    // Populate the neighbor sets
    for (int f = 0; f < F.rows(); ++f) {
        // Each face has three vertices
        for (int i = 0; i < 3; ++i) {
            int v1 = F(f, i);
            for (int j = 0; j < 3; ++j) {
                if (i != j) { // Avoid adding self as neighbor
                    int v2 = F(f, j);
                    neighbors[v1].insert(v2); // Add v2 as neighbor of v1
                }
            }
        }
    }

    // Create the neighbor matrix
    int maxNeighbors = 0;
    for (const auto& neighborSet : neighbors) {
        maxNeighbors = std::max(maxNeighbors, static_cast<int>(neighborSet.size()));
    }

    Eigen::MatrixXi neigh(numVertices, maxNeighbors);

    // Fill the neighbor matrix
    for (int i = 0; i < numVertices; ++i) {
        int j = 0;
        for (int neighbor : neighbors[i]) {
            neigh(i, j++) = neighbor;
        }
        // Fill remaining entries with -1 if fewer neighbors than max
        for (; j < maxNeighbors; ++j) {
            neigh(i, j) = -1; // or some sentinel value
        }
    }

    return neigh;
}

Eigen::SparseMatrix<double> constructMassMatrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    int numVertices = V.rows();
    Eigen::SparseMatrix<double> M(numVertices, numVertices);

    // Loop over each face (triangle)
    for (int f = 0; f < F.rows(); ++f) {
        // Get the indices of the vertices of the triangle
        int v1 = F(f, 0);
        int v2 = F(f, 1);
        int v3 = F(f, 2);

        // Calculate the area of the triangle
        Eigen::Vector3d p1 = V.row(v1);
        Eigen::Vector3d p2 = V.row(v2);
        Eigen::Vector3d p3 = V.row(v3);
        double area = 0.5 * ((p2 - p1).cross(p3 - p1)).norm();

        // Update the mass matrix entries
        M.coeffRef(v1, v1) += area / 3.0; // Vertex v1 shares area with the triangle
        M.coeffRef(v2, v2) += area / 3.0; // Vertex v2 shares area with the triangle
        M.coeffRef(v3, v3) += area / 3.0; // Vertex v3 shares area with the triangle

        M.coeffRef(v1, v2) += area / 6.0; // Off-diagonal entries for interactions
        M.coeffRef(v1, v3) += area / 6.0;
        M.coeffRef(v2, v1) += area / 6.0;
        M.coeffRef(v2, v3) += area / 6.0;
        M.coeffRef(v3, v1) += area / 6.0;
        M.coeffRef(v3, v2) += area / 6.0;
    }

    M.makeCompressed(); // Optional: compress the sparse matrix for efficiency
    return M;
}

Eigen::SparseMatrix<double> constructUniformLaplacian(int numVertices, const Eigen::MatrixXi& F) {
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    Eigen::VectorXd degree(numVertices);
    degree.setZero();

    // Loop over each face (triangle)
    for (int f = 0; f < F.rows(); ++f) {
        int v1 = F(f, 0);
        int v2 = F(f, 1);
        int v3 = F(f, 2);

        // Each pair of vertices in a triangle defines an edge
        degree(v1) += 1;
        degree(v2) += 1;
        degree(v3) += 1;

        triplets.emplace_back(v1, v2, -1);
        triplets.emplace_back(v2, v1, -1);
        triplets.emplace_back(v1, v3, -1);
        triplets.emplace_back(v3, v1, -1);
        triplets.emplace_back(v2, v3, -1);
        triplets.emplace_back(v3, v2, -1);
    }

    // Set diagonal entries
    for (int i = 0; i < numVertices; ++i) {
        triplets.emplace_back(i, i, degree(i));
    }

    Eigen::SparseMatrix<double> L(numVertices, numVertices);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

template<typename T>
void save_sparse_mat(const Eigen::SparseMatrix<T>& mat, const std::string& filename) {
    std::ofstream file(filename);

    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
            file << it.row() << " " << it.col() << " " << it.value() << "\n";
        }
    }

    file.close();
}


#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <sstream>

Eigen::MatrixXd loadMatrixMarketArray(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Skip comments/header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') break;  // First non-comment line
    }

    std::istringstream iss(line);
    int rows, cols;
    if (!(iss >> rows >> cols)) {
        throw std::runtime_error("Invalid header in MatrixMarket file.");
    }

    Eigen::MatrixXd matrix(rows, cols);

    // Read matrix values row-wise
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value;
            if (!(file >> value)) {
                throw std::runtime_error("Unexpected end of file.");
            }
            matrix(i, j) = value;
        }
    }

    return matrix;
}


int main() {
	Eigen::MatrixXd V;  // Vertex coordinates
	Eigen::MatrixXi F;  // Face indices

    // Load the mesh
    std::string filename = "bumpy-cube.obj";
    //std::string filename = "sphere3.obj";
    //std::string filename = "stanford-bunny.obj";
    if (!igl::readOBJ(filename, V, F)) {
        std::cerr << "Error loading the mesh." << std::endl;
        return -1;
    }

    // Create the neighbor matrix
    Eigen::MatrixXi neigh = createNeighborMatrix(V, F);

    // Output the neighbor matrix
    //std::cout << "Neighbor matrix:" << std::endl;
    //std::cout << neigh << std::endl;

    Eigen::SparseMatrix<double> M;// = constructMassMatrix(V, F);
    Eigen::SparseMatrix<double> laplacian=constructUniformLaplacian(V.rows(),F);// = constructMassMatrix(V, F);





    MGBS::MultigridSolver mg(V, neigh,laplacian);

	mg.buildHierarchy();
    save_sparse_mat(laplacian, "laplacian0.txt");
    Eigen::SparseMatrix<double> Abar = mg.U[0].transpose() * laplacian * mg.U[0];

	for(int i=1;i<mg.U.size();i++)
    {
        save_sparse_mat(Abar, "laplacian"+std::to_string(i)+".txt");
        Abar = mg.U[i].transpose() * Abar * mg.U[i];
    }

    /*
    std::string output_dir = "C:\\Users\\sachi\\OneDrive\\Documents\\GitHub\\RXMesh-fork\\Outputs";

    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> B;

    // Load a .mtx file
    if (Eigen::loadMarket(A, output_dir+"/A.mtx")) {
        std::cout << "\nSuccessfully loaded the matrix A." << std::endl;
    }
    else {
        std::cerr << "\nFailed to load the matrix A." << std::endl;
        return 1;
    }
    if (Eigen::loadMarket(B, output_dir + "/A.mtx")) {
        std::cout << "\nSuccessfully loaded the matrix B." << std::endl;
    }
    else {
        std::cerr << "\nFailed to load the matrix B." << std::endl;
        return 1;
    }


    std::cout << "\nConverted B to a dense matrix. Size: " << B.rows() << " x " << B.cols() << std::endl;
    std::cout << "\n A. Size: " << A.rows() << " x " << A.cols() << std::endl;

    Eigen::MatrixXd B_d = Eigen::MatrixXd(B);

    */
    //mg.solve(laplacian,);

   // mg.constructProlongation();

    

    //std::cout << "\nNumber of columns :" << mg.Abar[0].cols();

    //std::cout << std::endl<<mg.pointsAtLevel;




     //mg.U[0];

    /*
	Eigen::SparseMatrix<double> Abar;
    Abar[1] = U[0].transpose() * LHS * U[0];
    mg.U[k - 1].transpose()* V* mg.U[k - 1];
     */



    return 0;
}

