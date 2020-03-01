cmake_minimum_required (VERSION 2.6)
project (Tutorial)
# 版本号.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
 
# 配置一个头文件来传递一些CMake设置到源代码
configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  )
 
# 添加TutorialConfig.h的路径到头文件的搜索路径
include_directories("${PROJECT_BINARY_DIR}")

 
# # 添加目标可执行文件
# add_executable(Tutorial tutorial.cxx)


#需要用自定义的数学函数么？
option (USE_MYMATH
            "Use tutorial provided math implementation" ON)

# add the MathFunctions library?
if (USE_MYMATH)
  include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
  add_subdirectory (MathFunctions)
  SET (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif (USE_MYMATH)

# add executable
add_executable (Tutorial tutorial.cxx)
target_link_libraries (Tutorial ${EXTRA_LIBS})
