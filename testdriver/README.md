# NPU Core Test Driver

## How to test
1. Vitis 실행 후 Workspace 선택
2. 'File > New > Application Project...' 이후, Platform에서 'Create a new platform from hardware (XSA)' 탭에서 생성된 npu.xsa 선택 (Tcl에 의해 생성된 NPU directory에 있음)
3. 'Application Project name' 작성 후, ps7_cortexa9_0 선택하고 'next' 'freertos10_Xilinx'로 변경하고 Finish
4. Project Explorer의 src를 right-click & 'Import Sources...'
5. '/npu/testdriver' 선택 후, 'Select All & Finish'

## Single Core Test
* NPU 단일 Core에 대한 연산 기능 테스트
* 10, 11번의 연산자와 Target할 Core ID 변경하여 테스트

## Multi Core Test
* NPU 다중 Core에 대한 연산 기능 테스트
* 9, 10번의 연산자 변경하여 테스트
* 기본은 4개의 코어 동시에 연산