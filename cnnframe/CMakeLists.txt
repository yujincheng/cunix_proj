cmake_minimum_required(VERSION 2.8)
PROJECT (main)
SET(SRC_LIST main.c cnn.c net.c)
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE(main ${SRC_LIST})
TARGET_LINK_LIBRARIES(main m ${OpenCV_LIBS})

