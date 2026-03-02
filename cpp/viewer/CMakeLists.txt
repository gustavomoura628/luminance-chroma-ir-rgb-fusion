cmake_minimum_required(VERSION 3.22)
project(ir_rgb_fusion_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
pkg_check_modules(GL REQUIRED IMPORTED_TARGET gl)
pkg_check_modules(GLEW REQUIRED IMPORTED_TARGET glew)

# --- Dear ImGui via FetchContent ---
include(FetchContent)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.92.6
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC
  PkgConfig::SDL2
  PkgConfig::GL
  PkgConfig::GLEW
)

# --- Viewer executable ---
add_executable(viewer_node src/viewer_node.cpp)
target_link_libraries(viewer_node
  imgui_lib
  PkgConfig::SDL2
  PkgConfig::GL
  PkgConfig::GLEW
)
ament_target_dependencies(viewer_node rclcpp sensor_msgs)

install(TARGETS viewer_node DESTINATION lib/${PROJECT_NAME})

ament_package()
