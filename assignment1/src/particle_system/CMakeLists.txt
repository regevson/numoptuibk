set(SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/exercise.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/particlesystem.cpp"
    )
    
    
set(HDR
        "${CMAKE_CURRENT_SOURCE_DIR}/particlesystem.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/scene.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/../common/octree.h"
    )

add_executable( particlesystem ${SRC} ${HDR} )
target_link_libraries( particlesystem PRIVATE viewer_static )

target_compile_features( particlesystem PUBLIC cxx_std_17 )
set_target_properties( particlesystem PROPERTIES CXX_EXTENSIONS OFF )
