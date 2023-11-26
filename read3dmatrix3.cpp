#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

class Dataset {
private:
    std::vector<double> data;
    int rows, cols, depth;

public:
    // Constructor that loads the data
    Dataset(const std::string& dataFilename, const std::string& dimFilename) {
        // Read dimensions
        std::ifstream dimFile(dimFilename);
        if (!dimFile.is_open()) {
            throw std::runtime_error("Cannot open dimension file.");
        }
        dimFile >> rows >> cols >> depth;
        dimFile.close();

        // Reserve space for the data
        data.resize(rows * cols * depth);

        // Read binary data
        std::ifstream dataFile(dataFilename, std::ios::binary);
        if (!dataFile.is_open()) {
            throw std::runtime_error("Cannot open data file.");
        }
        dataFile.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(double));
        dataFile.close();
    }

    // Function to get an element from the dataset
    double getElement(int i, int j, int k) const {
        if (i < 1 || i > rows || j < 1 || j > cols || k < 1 || k > depth) {
            throw std::out_of_range("Index out of range.");
        }
        // Convert 1-based indices to 0-based indices for internal storage
        int zeroBasedI = i - 1;
        int zeroBasedJ = j - 1;
        int zeroBasedK = k - 1;
        // Calculate index for column-major order
        return data[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
    }
};

int main() {
    try {
        // Initialize dataset
        Dataset finger1("finger1_data.bin", "finger1_dimensions.txt");

        // Access some element
        std::cout << "Value at (1, 1, 1): " << finger1.getElement(1, 1, 1) << std::endl;
        std::cout << "Value at (2, 2, 2): " << finger1.getElement(2, 2, 2) << std::endl;
        
        // You can now use finger1.getElement(i, j, k) wherever you need to access the dataset
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
