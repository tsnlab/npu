/*
 * fpc header file
 */

#ifndef FPC_H
#define FPC_H

// custom macros
#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)

#define ROCC_INSTRUCTION_R_R_R(x, rd, rs1, rs2, func7)                               \
  {                                                                                  \
    asm volatile(                                                                    \
        ".insn r " STR(CAT(CUSTOM_, x)) ", " STR(0x7) ", " STR(func7) ", %0, %1, %2" \
        : "=r"(rd)                                                                   \
        : "r"(rs1), "r"(rs2));                                                       \
  }

#define ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, func7)                                   \
  {                                                                                  \
    asm volatile(                                                                    \
        ".insn r " STR(CAT(CUSTOM_, x)) ", " STR(0x3) ", " STR(func7) ", x0, %0, %1" \
        :                                                                            \
        : "r"(rs1), "r"(rs2));                                                       \
  }

// commands
#define FPC_WRITE	0
#define FPC_READ	1

// custom2
#define FPC_CUSTOM	2

// macros
#define FpcWrite(adr, wd) ROCC_INSTRUCTION_0_R_R(FPC_CUSTOM, adr, wd, FPC_WRITE)
#define FpcRead(adr, rd) { uint64_t _placeholder = 0; ROCC_INSTRUCTION_R_R_R(FPC_CUSTOM, rd, adr, _placeholder, FPC_READ) }

#endif  // FPC_H
