/*
 * Main Body for the RISCV Simulator
 *
 * Created by He, Hao at 2019-3-11
 * Revised by MINGYI at 2025-04-03
 */

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "BranchPredictor.h"
#include "MemoryManager.h"

namespace RISCV {
constexpr std::uint8_t REGISTERS_COUNT = 32;
constexpr std::array<std::string_view, REGISTERS_COUNT> REGISTER_NAME = {
    "zero",  // x0
    "ra",    // x1
    "sp",    // x2
    "gp",    // x3
    "tp",    // x4
    "t0",    // x5
    "t1",    // x6
    "t2",    // x7
    "s0",    // x8
    "s1",    // x9
    "a0",    // x10
    "a1",    // x11
    "a2",    // x12
    "a3",    // x13
    "a4",    // x14
    "a5",    // x15
    "a6",    // x16
    "a7",    // x17
    "s2",    // x18
    "s3",    // x19
    "s4",    // x20
    "s5",    // x21
    "s6",    // x22
    "s7",    // x23
    "s8",    // x24
    "s9",    // x25
    "s10",   // x26
    "s11",   // x27
    "t3",    // x28
    "t4",    // x29
    "t5",    // x30
    "t6",    // x31
};
constexpr std::array<std::string_view, 64> INSTNAME = {
    "lui",   "auipc",  "jal",    "jalr",  "beq",  "bne",  "blt",   "bge",
    "bltu",  "bgeu",   "lb",     "lh",    "lw",   "ld",   "lbu",   "lhu",
    "sb",    "sh",     "sw",     "sd",    "addi", "slti", "sltiu", "xori",
    "ori",   "andi",   "slli",   "srli",  "srai", "add",  "sub",   "sll",
    "slt",   "sltu",   "xor",    "srl",   "sra",  "or",   "and",   "ecall",
    "addiw", "mul",    "mulh",   "div",   "rem",  "lwu",  "slliw", "srliw",
    "sraiw", "addw",   "subw",   "sllw",  "srlw", "sraw", "fmadd", "fmaddu",
    "fmsub", "fmsubu", "fnmadd", "fnmsub"};
using RegId = std::uint32_t;

enum Reg : std::uint8_t {
  REG_ZERO = 0,
  REG_RA,
  REG_SP,
  REG_GP,
  REG_TP,
  REG_T0,
  REG_T1,
  REG_T2,
  REG_S0,
  REG_S,
  REG_A0,
  REG_A1,
  REG_A2,
  REG_A3,
  REG_A4,
  REG_A5,
  REG_A6,
  REG_A7,
  REG_S2,
  REG_S3,
  REG_S4,
  REG_S5,
  REG_S6,
  REG_S7,
  REG_S8,
  REG_S9,
  REG_S10,
  REG_S11,
  REG_T3,
  REG_T4,
  REG_T5,
  REG_T6,
};

enum InstructionType : std::uint8_t {
  R_TYPE,
  I_TYPE,
  S_TYPE,
  SB_TYPE,
  U_TYPE,
  UJ_TYPE,
};

enum Instruction : std::int8_t {
  UNKNOWN = -1,
  LUI,
  AUIPC,
  JAL,
  JALR,
  BEQ,
  BNE,
  BLT,
  BGE,
  BLTU,
  BGEU,
  LB,
  LH,
  LW,
  LD,
  LBU,
  LHU,
  SB,
  SH,
  SW,
  SD,
  ADDI,
  SLTI,
  SLTIU,
  XORI,
  ORI,
  ANDI,
  SLLI,
  SRLI,
  SRAI,
  ADD,
  SUB,
  SLL,
  SLT,
  SLTU,
  XOR,
  SRL,
  SRA,
  OR,
  AND,
  ECALL,
  ADDIW,
  MUL,
  MULH,
  DIV,
  REM,
  LWU,
  SLLIW,
  SRLIW,
  SRAIW,
  ADDW,
  SUBW,
  SLLW,
  SRLW,
  SRAW,
  FMADD,
  FMADDU,
  FMSUB,
  FMSUBU,
  FNMADD,
  FNMSUB,
};

enum OpCode : std::uint8_t {
  OP_REG = 0x33,
  OP_IMM = 0x13,
  OP_LUI = 0x37,
  OP_BRANCH = 0x63,
  OP_STORE = 0x23,
  OP_LOAD = 0x03,
  OP_SYSTEM = 0x73,
  OP_AUIPC = 0x17,
  OP_JAL = 0x6F,
  OP_JALR = 0x67,
  OP_IMM32 = 0x1B,
  OP_32 = 0x3B,
  OP_FMA = 0x0B,  // FMA opcode
};

inline auto isBranch(const Instruction& instruction) -> bool {
  return instruction == BEQ || instruction == BNE || instruction == BLT ||
         instruction == BGE || instruction == BLTU || instruction == BGEU;
}

inline auto isJump(const Instruction& instruction) -> bool {
  return instruction == JAL || instruction == JALR;
}

inline auto isReadMem(const Instruction& instruction) -> bool {
  return instruction == LB || instruction == LH || instruction == LW ||
         instruction == LD || instruction == LBU || instruction == LHU ||
         instruction == LWU;
}
}  // namespace RISCV

class Simulator {
 public:
  bool isSingleStep;
  bool verbose;
  bool shouldDumpHistory;
  bool dataForwarding;
  uint32_t pc;
  std::array<uint32_t, RISCV::REGISTERS_COUNT> reg;
  uint32_t stackBase;
  uint32_t maximumStackSize;
  MemoryManager* memory;
  BranchPredictor* branchPredictor;

  Simulator(MemoryManager* memory, BranchPredictor* predictor);
  ~Simulator();

  void initStack(const uint32_t& baseaddr, const uint32_t& maxSize);

  [[noreturn]] void simulate();

  void dumpHistory() const;

  void printInfo() const;

  void printStatistics() const;

 private:
  struct FReg {
    // Control Signals
    bool bubble;
    uint32_t stall;

    uint32_t pc;
    uint32_t inst;
    uint32_t len;
  } fReg, fRegNew;

  struct DReg {
    // Control Signals
    bool bubble;
    uint32_t stall;

    // Registers
    RISCV::RegId dest;
    RISCV::RegId rs1;
    RISCV::RegId rs2;
    RISCV::RegId rs3;

    // True values
    int32_t op1;
    int32_t op2;
    int32_t op3;
    int32_t offset;

    uint32_t pc;
    RISCV::Instruction inst;
    std::string rawAssemblyInstruction;

    bool predictedBranch;
    uint32_t
        predictedPC;  // for branch prediction module, predicted PC destination
    uint32_t anotherPC;  // another possible prediction destination
  } dReg, dRegNew;

  struct EReg {
    // Control Signals
    bool bubble;
    uint32_t stall;

    std::string rawAssemblyInstruction;
    uint32_t pc;
    RISCV::Instruction inst;
    int32_t op2;
    bool enableWriteRegister;
    RISCV::RegId destReg;
    int32_t out;
    bool enableWriteMemory;
    bool enableReadMemory;
    bool readSignExt;
    uint32_t memLen;
  } eReg, eRegNew;

  struct MReg {
    bool bubble;

    RISCV::Instruction inst;
    std::string rawAssemblyInstruction;
    int32_t out;
    bool enableWriteBack;
    RISCV::RegId destReg;
  } mReg, mRegNew;

  // Pipeline Related Variables
  // To avoid older values(in MEM) overriding newer values(in EX)
  bool executeWriteBack;
  RISCV::RegId executeWBReg;

  struct History {
    uint32_t instructionCount;
    uint32_t cycleCount;
    uint32_t stalledCycleCount;

    uint32_t
        predictedBranch;  // Number of branch that is predicted successfully
    uint32_t unpredictedBranch;  // Number of branch that is not predicted
    // successfully

    uint32_t dataHazardCount;
    uint32_t controlHazardCount;
    uint32_t memoryHazardCount;

    std::vector<std::string> instRecord;
    std::vector<std::string> regRecord;

    std::string memoryDump;
  } history;

  void fetch();
  void decode();
  void execute();
  void memoryAccess();
  void writeBack();
  void newWB();

  auto handleSystemCall(int32_t op1, int32_t op2) -> int32_t;

  auto getRegInfoStr() const -> std::string;
  void panic(const char* format, ...);
  void verbosePrint(const std::string& str) const;
};

#endif
