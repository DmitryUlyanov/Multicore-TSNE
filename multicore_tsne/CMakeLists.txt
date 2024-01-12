PROJECT(tsne_multicore CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)
CMAKE_POLICY(VERSION 3.9)

SET(TSNE_MULTICORE_HEADER_FILES
    splittree.h
    tsne.h
    vptree.h
)

ADD_LIBRARY(tsne_multicore
STATIC
    tsne.cpp
    splittree.cpp

    ${TSNE_MULTICORE_HEADER_FILES}
    ${TSNE_MULTICORE_HEADER_FILES_PRIVATE}
)

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

include(GNUInstallDirs)

set(TSNE_MULTICORE_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/tsne_multicore)
set(TSNE_MULTICORE_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/tsne_multicore)
set(TSNE_MULTICORE_LIBDIR ${CMAKE_INSTALL_LIBDIR})

TARGET_INCLUDE_DIRECTORIES(tsne_multicore
INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${TSNE_MULTICORE_INCLUDEDIR}>
)

install(TARGETS tsne_multicore
EXPORT
    tsne_multicore_targets
LIBRARY DESTINATION ${TSNE_MULTICORE_LIBDIR}
ARCHIVE DESTINATION ${TSNE_MULTICORE_LIBDIR}
)

install(FILES ${TSNE_MULTICORE_HEADER_FILES}
DESTINATION ${TSNE_MULTICORE_INCLUDEDIR}
COMPONENT Devel
)

install(EXPORT tsne_multicore_targets
  FILE tsne_multicore_targets.cmake
  NAMESPACE tsne_multicore::
  DESTINATION ${TSNE_MULTICORE_CONFIGDIR}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/tsne_multicoreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/tsne_multicoreConfig.cmake
  INSTALL_DESTINATION ${TSNE_MULTICORE_CONFIGDIR}
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/tsne_multicoreConfig.cmake
  DESTINATION ${TSNE_MULTICORE_CONFIGDIR}
)
