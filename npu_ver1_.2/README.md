# NPU ver 1.2


## Generating Project
1. Xiliinx Vivado 실행
2. 메뉴바의 `Tools > Run Tcl Script...` 선택
3. npu.tcl 선택 후 `open`
4. `npu.tcl` 파일이 열리고 기다리면 `npu.xsa` 출력됨

## Caution
Tcl run을 하면 Vivado를 실행시킨 현재 디렉토리의 상위 디렉토리에서 프로젝트가 생성됩니다.
Tcl 파일이 있는 위치에서 Vivado 실행을 해야 `File or Directory does not exist Error`가 발생하지 않습니다.