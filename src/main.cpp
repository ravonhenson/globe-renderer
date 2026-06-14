#include "GlobeApp.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const std::filesystem::path executablePath = (argc > 0 && argv[0] != nullptr)
            ? std::filesystem::path(argv[0])
            : std::filesystem::current_path();
        GlobeApp app(executablePath);
        app.run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
