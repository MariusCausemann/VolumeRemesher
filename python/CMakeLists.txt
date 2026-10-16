# volume_remesher._core: the nanobind module over driver.h.
#
# Driven by scikit-build-core (pip install .) or by a plain configure with
# -DVOLUMEREMESHER_BUILD_PYTHON=ON. In the second case the module is placed next to the Python
# sources in python/volume_remesher/ so the package imports straight from the tree.

# SKBUILD_SABI_COMPONENT is "Development.SABIModule" when scikit-build-core builds a stable-ABI
# wheel (wheel.py-api in pyproject.toml), empty otherwise.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

# nanobind comes from the Python package of the same name (pip install nanobind).
execute_process(
	COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
	OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
	RESULT_VARIABLE nanobind_query_result)
if(NOT nanobind_query_result EQUAL 0)
	message(FATAL_ERROR "nanobind is not installed for ${Python_EXECUTABLE}: pip install nanobind")
endif()
find_package(nanobind CONFIG REQUIRED)

# STABLE_ABI: one wheel per platform for CPython >= 3.12 (ignored on older interpreters).
# NB_STATIC: nanobind's runtime is linked into the module, so it has no sibling to find.
nanobind_add_module(_core STABLE_ABI NB_STATIC LTO bindings.cpp)
target_link_libraries(_core PRIVATE VolumeRemesher::VolumeRemesher)
target_compile_features(_core PRIVATE cxx_std_20)

if(SKBUILD)
	install(TARGETS _core LIBRARY DESTINATION volume_remesher)
else()
	set_target_properties(_core PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/volume_remesher"
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/volume_remesher")
	foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
		string(TOUPPER "${cfg}" CFG)
		set_target_properties(_core PROPERTIES
			LIBRARY_OUTPUT_DIRECTORY_${CFG} "${CMAKE_CURRENT_SOURCE_DIR}/volume_remesher"
			RUNTIME_OUTPUT_DIRECTORY_${CFG} "${CMAKE_CURRENT_SOURCE_DIR}/volume_remesher")
	endforeach()
endif()
