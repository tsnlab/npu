# NPU Core Test Driver

## Caution
* `lscript.ld` 파일에서 Stack Size와 Heap Size를 0x2000에서 0x20000으로 변경 필수
* 최대 Data Size는 2730개

## Single Core Test
* NPU 단일 Core에 대한 연산 기능 테스트
* 10, 11번의 연산자와 Target할 Core ID 변경하여 테스트

## Multi Core Test
* NPU 다중 Core에 대한 연산 기능 테스트
* 9, 10번의 연산자 변경하여 테스트
* 기본은 4개의 코어 동시에 연산