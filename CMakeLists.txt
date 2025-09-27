cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(fishing_bot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SOURCE_FILES "*.cpp" "*.h" "*.hpp")
file(GLOB TEMPLATE_FILES "${CMAKE_SOURCE_DIR}/templates/*")
add_executable(fishing_bot ${SOURCE_FILES} ${TEMPLATE_FILES} config.json)

source_group("templates" FILES ${TEMPLATE_FILES})
source_group("config" FILES config.json)
source_group("sources" FILES ${SOURCE_FILES} )

if(DEFINED HAVE_CUDA)
    target_compile_definitions(fishing_bot PRIVATE HAVE_CUDA=${HAVE_CUDA})
endif()
if(DEFINED HAVE_CL)
    target_compile_definitions(fishing_bot PRIVATE HAVE_CL=${HAVE_CL})
endif()

# Link libraries
target_link_libraries(fishing_bot ${TORCH_LIBRARIES})
target_link_libraries(fishing_bot ${OpenCV_LIBRARIES})

# Include directories
target_include_directories(fishing_bot PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set properties for LibTorch
set_property(TARGET fishing_bot PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET fishing_bot
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:fishing_bot>)
  file(GLOB OPENCV_DLLS "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/*.dll")
  add_custom_command(TARGET fishing_bot
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:fishing_bot>)
endif (MSVC)

add_custom_command(TARGET fishing_bot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_SOURCE_DIR}/config.json"
    $<TARGET_FILE_DIR:fishing_bot>)

add_custom_command(TARGET fishing_bot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_SOURCE_DIR}/fishing_classifier_traced.pt"
    $<TARGET_FILE_DIR:fishing_bot>)

add_custom_command(TARGET fishing_bot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/templates"
    "$<TARGET_FILE_DIR:fishing_bot>/templates")
