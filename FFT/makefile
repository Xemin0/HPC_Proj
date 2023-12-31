## Make File to Compile the projet

#### !!Replace the main driver and the `FFTW` library path as needed!! ####

# Compiler and flags ### g++ or g++-13
CXX = g++
#CXXFLAGS = -Wall -std=c++11 -I/opt/homebrew/Cellar/fftw/3.3.10_1/include
#LDFLAGS = -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3

CXXFLAGS = -Wall -std=c++11 -I/usr/local/Cellar/fftw/3.3.10_1/include
LDFLAGS = -L/usr/local/Cellar/fftw/3.3.10_1/lib -lfftw3




# Directories ## Add more dir for modules
SRC_DIR = .
LIB_DIR = lib
CT_DIR = CooleyTukeyFFT
DL_DIR = data_loader
UTILS_DIR = utils
EVAL_DIR = eval

# Source files ## Add more modules
SOURCES = $(SRC_DIR)/main.cpp \
          $(DL_DIR)/loader.cpp \
          $(UTILS_DIR)/timer.cpp \
		  $(UTILS_DIR)/vec_utils.cpp \
		  $(EVAL_DIR)/eval_correctness.cpp \
		  $(EVAL_DIR)/eval_performance.cpp \
          $(CT_DIR)/iterative_CT.cpp \
          $(CT_DIR)/recursive_CT.cpp \
          $(CT_DIR)/fftw_wrapper.cpp \
		  $(CT_DIR)/fft2d.cpp  #### Add NewFiles Below this line: Presumably under $(CT_DIR)

# Object files ## No Need to Change
OBJECTS = $(SOURCES:.cpp=.o)

# Executable name ## No Need to Change unless renaming is necessary
EXEC = It_CT.out

# Default target ## No Need to Change
all: $(EXEC)
	@echo "Build Successful!"
	rm -f $(OBJECTS) 		# Remove intermediate .o files

# // Linking 
#### Append Compiling Flags as Needed, e.g. OpenMP, MPI, CUDA
#### Change the Compiler as Needed, e.g. MPI, CUDA
$(EXEC): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Generic rule for building object files  #### No Need to Change
# // Compiling For intemerdiate binary files with `-c`
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

# Clean up #### No Need to Change
clean:
	rm -f $(OBJECTS) $(EXEC)


# Dependencies 
# Miscellaneous Section #### No Need to Change
$(DL_DIR)/loader.o: $(LIB_DIR)/loader.h 			# Data loader
$(UTILS_DIR)/timer.o: $(LIB_DIR)/timer.h 			# CPU/GPU timer
$(UTILS_DIR)/vec_utils.o: $(LIB_DIR)/vec_utils.h	# Vector manipulations
$(EVAL_DIR)/eval_correctness.o: $(LIB_DIR)/eval_correctness.h	# Validation of Correctness
$(EVAL_DIR)/eval_performance.o: $(LIB_DIR)/eval_performance.h	# Validation of Performance

# FFT Method Section
#### Relate the Intermediate Binary Files with its Direct Declartion (Header)
$(CT_DIR)/iterative_CT.o: $(LIB_DIR)/iterative_CT.h # Iterative method
$(CT_DIR)/recursive_CT.o: $(LIB_DIR)/recursive_CT.h # Recursive method
$(CT_DIR)/fftw_wrapper.o: $(LIB_DIR)/fftw_wrapper.h # FFTW wrapper
$(CT_DIR)/fft2d.o: $(LIB_DIR)/fft2d.h 				# 2D FFT

# Main Driver #### No Need to Change
#				   $(LIB_DIR)/timer.h $(LIB_DIR)/vec_utils.h \
$(SRC_DIR)/main.o: $(LIB_DIR)/loader.h \
				   $(LIB_DIR)/eval_correctness.h $(LIB_DIR)/eval_performance.h \
				   $(LIB_DIR)/iterative_CT.h $(LIB_DIR)/recursive_CT.h \
				   $(LIB_DIR)/fftw_wrapper.h $(LIB_DIR)/fft2d.h  #### Add New Lines of dependencies needed in main.cpp

