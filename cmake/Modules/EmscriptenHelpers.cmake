FUNCTION(mr_emscripten_pack_directory SRC DST)
  set(PRELOAD_ARGUMENTS "--preload-file ${SRC}@${DST} ")
  string(FIND ${CMAKE_EXE_LINKER_FLAGS} ${PRELOAD_ARGUMENTS} FOUND)
  IF(FOUND EQUAL -1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PRELOAD_ARGUMENTS}" PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

FUNCTION(mr_emscripten_set_async_func_list FUNC_LIST_FILE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY_IGNORE_INDIRECT -s  ASYNCIFY_ADD=@${FUNC_LIST_FILE}" PARENT_SCOPE)
ENDFUNCTION()
