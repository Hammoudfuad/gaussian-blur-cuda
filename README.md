# Gaussian Blur Filter with CUDA and NPP

Welcome to the Gaussian Blur Filter project. This project uses CUDA and NPP to apply a Gaussian blur filter to images. 
## Requirements

Before you start, make sure you have the following:

- **CUDA Toolkit**: Version 10.2 or later.
- **NVIDIA GPU**: With compute capability 3.5 or higher.
- **g++ Compiler**: For compiling the C++ code.
- **FreeImage Library**: For image loading and saving.

## Getting Started

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gaussian-blur-cuda.git
    cd gaussian-blur-cuda
    ```

### Building the Project

1. Navigate to the project directory:
    ```sh
    cd gaussian-blur-cuda
    ```

2. Build the project using the Makefile:
    ```sh
    make
    ```

3. Run the project:
    ```sh
    make run
    ```

## Project Structure

- `gaussianBlurNPP.cpp`: The main source code file where the Gaussian blur is implemented.
- `Makefile`: The file used to build the project.
- `PGM_Images/`: The directory where the processed images will be saved.

## How to Use

1. **Input Images**: Place your `.pgm` images in the `input/` folder.

2. **Run the Program**:
    ```sh
    ./gaussianBlurNPP -input=/path/to/input_folder -output=/path/to/output_folder
    ```

### Example

1. Place images: Put your images in `/PGM_Images`.

2. Run the program:
    ```sh
    ./gaussianBlurNPP -input=/home/coder/project/boxFilterNPP/PGM_Images -output=/home/coder/project/boxFilterNPP/PGM_Images/processed
    ```

3. Check the output: Processed images will be saved in the `PGM_Images/processed/` folder.


## Logs

The build_and_run.txt shows the logs of running the process.