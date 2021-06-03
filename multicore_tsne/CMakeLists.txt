PROJECT(tsne_multicore CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)
CMAKE_POLICY(VERSION 3.9)

ADD_LIBRARY(tsne_multicore STATIC tsne.cpp splittree.cpp)
ADD_LIBRARY(tsne_multicore::tsne_multicore ALIAS tsne_multicore)
TARGET_LINK_LIBRARIES(tsne_multicore PRIVATE m)

# enable -fPIC
SET_TARGET_PROPERTIES(tsne_multicore PROPERTIES POSITION_INDEPENDENT_CODE True)

TARGET_COMPILE_OPTIONS(tsne_multicore PRIVATE -Wall)

IF(MSVC)
    TARGET_COMPILE_OPTIONS(tsne_multicore
    PRIVATE
        $<$<CONFIG:Release>:/fp:fast>
        $<$<CONFIG:RelWithDebInfo>:/fp:fast>
    )
ELSE()
    TARGET_COMPILE_OPTIONS(tsne_multicore
    PRIVATE
        $<$<CONFIG:Release>:-ffast-math -funroll-loops>
        $<$<CONFIG:RelWithDebInfo>:-ffast-math -funroll-loops>
    )
ENDIF()

FIND_PACKAGE(OpenMP 2)
IF(OPENMP_FOUND)
    TARGET_LINK_LIBRARIES(tsne_multicore PRIVATE OpenMP::OpenMP_CXX)
ELSE()
    MESSAGE(WARNING "Not using OpenMP. Performance will suffer.")
ENDIF()

TARGET_INCLUDE_DIRECTORIES(tsne_multicore INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

