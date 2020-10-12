cmake_minimum_required(VERSION 3.4.1)


set(distribution_DIR ${CMAKE_SOURCE_DIR}/../distribution)

add_library(libonnxruntime SHARED IMPORTED)

set_target_properties(libonnxruntime PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so)



add_library( native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp  src/main/cpp/onnxruntime_inference.cpp)


include_directories(native-lib PRIVATE ${distribution_DIR}/include)
include_directories(${distribution_DIR}/include/onnxruntime/core/session)



find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )
find_library( # Sets the name of the path variable.
           jnigraphics-lib

           # Specifies the name of the NDK library that
           # you want CMake to locate.
           jnigraphics )


target_link_libraries( # Specifies the target library.
                       native-lib
                       libonnxruntime

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}

                       ${jnigraphics-lib})