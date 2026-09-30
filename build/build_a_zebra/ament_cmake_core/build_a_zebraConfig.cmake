# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_build_a_zebra_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED build_a_zebra_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(build_a_zebra_FOUND FALSE)
  elseif(NOT build_a_zebra_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(build_a_zebra_FOUND FALSE)
  endif()
  return()
endif()
set(_build_a_zebra_CONFIG_INCLUDED TRUE)

# output package information
if(NOT build_a_zebra_FIND_QUIETLY)
  message(STATUS "Found build_a_zebra: 0.1.0 (${build_a_zebra_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'build_a_zebra' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${build_a_zebra_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(build_a_zebra_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${build_a_zebra_DIR}/${_extra}")
endforeach()
