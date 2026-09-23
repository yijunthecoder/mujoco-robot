# CMake Version
# Project Name ${PROJECT_NAME}
cmake_minimum_required(VERSION 3.8) 
project(build_a_zebra) 

# Use C++ 17 IF No Version is already selected
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  # -Wall       Enable common warnings.
  # -Wextra     Enable additional useful warnings.
  # -Wpedantic  Warn when code is not fully standard C++.
endif()


find_package(ament_cmake REQUIRED)         # ament_cmake is ROS 2’s CMake build system.
find_package(rclcpp REQUIRED)              # rclcpp: ROS 2 C++ tools.
find_package(std_msgs REQUIRED)            # std_msgs: basic ROS messages such as String.
find_package(geometry_msgs REQUIRED)       # geometry_msgs: ROS geometry types, such as a 3D Point.
find_package(ament_index_cpp REQUIRED)     # ament_index_cpp: lets your code locate installed resource files.
find_package(behaviortree_cpp REQUIRED)    # behaviortree_cpp: the Behavior Tree library.

# Compiling, also search inside the `include/` folder for header files.”
# Headers (hpp) are stored here
include_directories(include)

add_executable(zebra_bt_node
  src/main.cpp                            # Starts ROS, creates objects, runs the BT loop.
  src/world_model.cpp                     # Stores live part state and perception updates.
  src/task_model.cpp                      # Reads the Zebra BOM/recipe.
  src/recovery_manager.cpp                # Decides retry, relocate, or escalate.
  src/disturbance_generator.cpp           # Creates simulated loss/noise/relocation events.
)

ament_target_dependencies(zebra_bt_node
  rclcpp                                  # Use ROS nodes/topics.
  std_msgs                                # Use String messages.
  geometry_msgs                           # Use Point messages.
  ament_index_cpp                         # Locate JSON/XML resources.
)

# BehaviorTree.CPP v4 exports this target name on this system:
target_link_libraries(zebra_bt_node
  behaviortree_cpp::behaviortree_cpp
)

# ROS where to place the built executable.
install(TARGETS
  zebra_bt_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  stationlite_mujoco
  DESTINATION share/${PROJECT_NAME}
)

# Make the tree XML and BOM json available at runtime via the install share
# dir, so main.cpp can find them with ament_index_cpp::get_package_share_directory.
install(DIRECTORY
  trees
  resources
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # ament_lint_auto disabled -- it enforces upstream ROS style rules
  # (copyright headers, uncrustify, cpplint, flake8) that don't matter
  # for this project. Only the unit tests below are what we care about.
  # find_package(ament_lint_auto REQUIRED)
  # ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(task_model_test
    test/task_model_test.cpp
    src/task_model.cpp
  )
endif()

install(PROGRAMS
  scripts/mock_executor.py
  scripts/mujoco_test_executor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()                          # Directory is a complete ROS 2 package.