cmake_minimum_required(VERSION 3.0.2)
project(dh_nonlinear_control)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
	dh_casadi_tools
)

# catkin_python_setup()

# file(GLOB_RECURSE MSG_FILES RELATIVE ${PROJECT_SOURCE_DIR}/msg msg/[^.]*.msg)
# add_message_files(FILES ${MSG_FILES})
# file(GLOB_RECURSE SRV_FILES RELATIVE ${PROJECT_SOURCE_DIR}/srv srv/[^.]*.srv)
# add_service_files(FILES ${SRV_FILES})
# file(GLOB_RECURSE ACTION_FILES RELATIVE ${PROJECT_SOURCE_DIR}/action action/[^.]*.action)
# add_action_files(FILES ${ACTION_FILES})

# generate_messages(DEPENDENCIES
# 	actionlib_msgs
# )

catkin_package(
	INCLUDE_DIRS include
	LIBRARIES ${PROJECT_NAME}
)

include_directories(${catkin_INCLUDE_DIRS} /usr/include/eigen3)

# ライブラリを作成
file(GLOB_RECURSE LIB_CPP_FILES RELATIVE ${PROJECT_SOURCE_DIR} src/${PROJECT_NAME}/[^.]*.cpp)
add_library(${PROJECT_NAME} STATIC ${LIB_CPP_FILES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

# ノードの実行ファイルを作成
# file(GLOB_RECURSE NODE_FILES RELATIVE ${PROJECT_SOURCE_DIR} src/node/[^.]*.cpp) # src内のノードへのパスを取得
# foreach(FILE IN LISTS NODE_FILES)
# 	get_filename_component(NODE_NAME ${FILE} NAME_WE) # 拡張子を除いたファイル名を取得
# 	add_executable(${NODE_NAME} ${FILE}) # コンパイル
# 	target_link_libraries(${NODE_NAME} ${catkin_LIBRARIES}) # リンク
# endforeach()
