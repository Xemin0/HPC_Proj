#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Function to map 3D indices (1-based) to 1D index for column-major order
double getElement(const std::vector<double>& matrixData, int rows, int cols, int depth, int i, int j, int k) {
    // Convert 1-based indices to 0-based indices
    int zeroBasedI = i - 1;
    int zeroBasedJ = j - 1;
    int zeroBasedK = k - 1;

    // Calculate 0-based index for the 1D array and return the value
    // Note: In column-major order, the first index changes fastest.
    return matrixData[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
}


int main() {
    // Reading dimensions from the text file
    std::ifstream dimFile("finger1_dimensions.txt");
    if (!dimFile.is_open()) {
        std::cerr << "Error opening dimensions file." << std::endl;
        return 1;
    }

    int rows, cols, depth;
    dimFile >> rows >> cols >> depth;
    dimFile.close();

    // Reading the binary data
    std::ifstream binFile("finger1_data.bin", std::ios::binary);
    if (!binFile.is_open()) {
        std::cerr << "Error opening binary data file." << std::endl;
        return 1;
    }

    std::vector<double> matrixData(rows * cols * depth);
    binFile.read(reinterpret_cast<char*>(matrixData.data()), matrixData.size() * sizeof(double));
    binFile.close();

    // Example of accessing an element (i, j, k) in the matrix
    // In MATLAB, the indices would be (1, 1, 1)
    // In C++, they should be adjusted to (0, 0, 0) for the first element
    int i = 3, j = 3, k = 3; // MATLAB-style 1-based indices
    if (i > 0 && i <= rows && j > 0 && j <= cols && k > 0 && k <= depth) {
        double value = getElement(matrixData, rows, cols, depth, i, j, k);
        std::cout << "Value at (" << i << ", " << j << ", " << k << "): " << value << std::endl;
    } else {
        std::cerr << "Indices are out of bounds." << std::endl;
    }

    return 0;
}
