cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(fishing_bot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add executable
add_executable(fishing_bot
  main.cpp
  GameScreen.h
  GameScreen.cpp
  DebugDrawer.h
  DebugDrawer.cpp
  Tools.h
  Tools.cpp
  tinyfsm.hpp
  json.hpp
  FishingBot.h
  FishingBot.cpp)

# Link libraries
target_link_libraries(fishing_bot ${TORCH_LIBRARIES})
target_link_libraries(fishing_bot ${OpenCV_LIBRARIES})

# Include directories
target_include_directories(fishing_bot PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set properties for LibTorch
set_property(TARGET fishing_bot PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET fishing_bot
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:fishing_bot>)
endif (MSVC)