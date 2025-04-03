/*
 * Main Body for the RISCV Simulator
 *
 * Created by He, Hao at 2019-3-11
 * Revised by MINGYI at 2025-04-03
 */

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <array>
#include <cstdarg>
#include <cstdint>
#include <string>
#include <vector>

#include "BranchPredictor.h"
#include "MemoryManager.h"

namespace RISCV {
constexpr int REGNUM = 32;
extern const char* REGNAME[32];
using RegId = uint32_t;

enum Reg {
  REG_ZERO = 0,
  REG_RA = 1,
  REG_SP = 2,
  REG_GP = 3,
  REG_TP = 4,
  REG_T0 = 5,
  REG_T1 = 6,
  REG_T2 = 7,
  REG_S0 = 8,
  REG_S1 = 9,
  REG_A0 = 10,
  REG_A1 = 11,
  REG_A2 = 12,
  REG_A3 = 13,
  REG_A4 = 14,
  REG_A5 = 15,
  REG_A6 = 16,
  REG_A7 = 17,
  REG_S2 = 18,
  REG_S3 = 19,
  REG_S4 = 20,
  REG_S5 = 21,
  REG_S6 = 22,
  REG_S7 = 23,
  REG_S8 = 24,
  REG_S9 = 25,
  REG_S10 = 26,
  REG_S11 = 27,
  REG_T3 = 28,
  REG_T4 = 29,
  REG_T5 = 30,
  REG_T6 = 31,
};

enum InstType {
  R_TYPE,
  I_TYPE,
  S_TYPE,
  SB_TYPE,
  U_TYPE,
  UJ_TYPE,
};

enum Inst {
  LUI = 0,
  AUIPC = 1,
  JAL = 2,
  JALR = 3,
  BEQ = 4,
  BNE = 5,
  BLT = 6,
  BGE = 7,
  BLTU = 8,
  BGEU = 9,
  LB = 10,
  LH = 11,
  LW = 12,
  LD = 13,
  LBU = 14,
  LHU = 15,
  SB = 16,
  SH = 17,
  SW = 18,
  SD = 19,
  ADDI = 20,
  SLTI = 21,
  SLTIU = 22,
  XORI = 23,
  ORI = 24,
  ANDI = 25,
  SLLI = 26,
  SRLI = 27,
  SRAI = 28,
  ADD = 29,
  SUB = 30,
  SLL = 31,
  SLT = 32,
  SLTU = 33,
  XOR = 34,
  SRL = 35,
  SRA = 36,
  OR = 37,
  AND = 38,
  ECALL = 39,
  ADDIW = 40,
  MUL = 41,
  MULH = 42,
  DIV = 43,
  REM = 44,
  LWU = 45,
  SLLIW = 46,
  SRLIW = 47,
  SRAIW = 48,
  ADDW = 49,
  SUBW = 50,
  SLLW = 51,
  SRLW = 52,
  SRAW = 53,
  FMADD = 54,
  FMADDU = 55,
  FMSUB = 56,
  FMSUBU = 57,
  FNMADD = 58,
  FNMSUB = 59,
  UNKNOWN = -1,
};

extern const char* INSTNAME[];

// Opcode field
constexpr int OP_REG = 0x33;
constexpr int OP_IMM = 0x13;
constexpr int OP_LUI = 0x37;
constexpr int OP_BRANCH = 0x63;
constexpr int OP_STORE = 0x23;
constexpr int OP_LOAD = 0x03;
constexpr int OP_SYSTEM = 0x73;
constexpr int OP_AUIPC = 0x17;
constexpr int OP_JAL = 0x6F;
constexpr int OP_JALR = 0x67;
constexpr int OP_IMM32 = 0x1B;
constexpr int OP_32 = 0x3B;
constexpr int32_t OP_FMA = 0x0B;  // FMA opcode

inline auto isBranch(const Inst& inst) -> bool {
  return inst == BEQ || inst == BNE || inst == BLT || inst == BGE ||
         inst == BLTU || inst == BGEU;
}

inline auto isJump(const Inst& inst) -> bool {
  return inst == JAL || inst == JALR;
}

inline auto isReadMem(const Inst& inst) -> bool {
  return inst == LB || inst == LH || inst == LW || inst == LD || inst == LBU ||
         inst == LHU || inst == LWU;
}
}  // namespace RISCV

class Simulator {
 public:
  bool isSingleStep;
  bool verbose;
  bool shouldDumpHistory;
  uint32_t pc;
  std::array<int32_t, RISCV::REGNUM> reg;
  uint32_t stackBase;
  uint32_t maximumStackSize;
  MemoryManager* memory;
  BranchPredictor* branchPredictor;

  Simulator(MemoryManager* memory, BranchPredictor* predictor);
  ~Simulator();

  void initStack(const uint32_t& baseaddr, const uint32_t& maxSize);

  [[noreturn]] void simulate();

  void dumpHistory();

  void printInfo();

  void printStatistics();

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
    RISCV::Inst inst;

    bool predictedBranch;
    uint32_t
        predictedPC;  // for branch prediction module, predicted PC destination
    uint32_t anotherPC;  // another possible prediction destination
  } dReg, dRegNew;

  struct EReg {
    // Control Signals
    bool bubble;
    uint32_t stall;

    uint32_t pc;
    RISCV::Inst inst;
    int32_t op1;
    int32_t op2;
    bool writeReg;
    RISCV::RegId destReg;
    int32_t out;
    bool writeMem;
    bool readMem;
    bool readSignExt;
    uint32_t memLen;
    bool branch;
  } eReg, eRegNew;

  struct MReg {
    // Control Signals
    bool bubble;
    uint32_t stall;

    uint32_t pc;
    RISCV::Inst inst;
    int32_t op1;
    int32_t op2;
    int32_t out;
    bool writeReg;
    RISCV::RegId destReg;
  } mReg, mRegNew;

  // Pipeline Related Variables
  // To avoid older values(in MEM) overriding newer values(in EX)
  bool executeWriteBack;
  RISCV::RegId executeWBReg;
  bool memoryWriteBack;
  RISCV::RegId memoryWBReg;

  struct History {
    uint32_t instCount;
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

  auto handleSystemCall(int32_t op1, int32_t op2) -> int32_t;

  auto getRegInfoStr() -> std::string;
  void panic(const char* format, ...);
};

#endif
