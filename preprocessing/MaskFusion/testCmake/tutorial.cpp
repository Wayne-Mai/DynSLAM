#include "TutorialConfig.h"
#include <iostream>

int main(){

if (argc < 2) {
    // report version
    std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
              << Tutorial_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;

    const double inputValue = std::stod(argv[1]);
    return 1;
  }
}