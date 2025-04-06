/*
 * Created by He, Hao at 2019-3-11
 * Revised by MINGYI at 2025-04-03
 */

#include "Simulator.h"

#include <fstream>
#include <sstream>
#include <string>

#include "Debug.h"

using namespace RISCV;

Simulator::Simulator(MemoryManager* memory, BranchPredictor* predictor)
    : isSingleStep(false),
      verbose(false),
      shouldDumpHistory(false),
      dataForwarding(true),
      pc(0),
      reg(),
      stackBase(0),
      maximumStackSize(0),
      memory(memory),
      branchPredictor(predictor),
      fReg(),
      fRegNew(),
      dReg(),
      dRegNew(),
      eReg(),
      eRegNew(),
      mReg(),
      mRegNew(),
      executeWriteBack(false),
      executeWBReg(-1),
      history() {
  this->fReg.bubble = true;
  this->dReg.bubble = true;
  this->eReg.bubble = true;
  this->mReg.bubble = true;
}

Simulator::~Simulator() = default;

void Simulator::initStack(const uint32_t& baseaddr, const uint32_t& maxSize) {
  this->reg.at(REG_SP) = baseaddr;
  this->stackBase = baseaddr;
  this->maximumStackSize = maxSize;
}

[[noreturn]] void Simulator::simulate() {
  while (true) {
    this->verbosePrint("\n");
    this->reg.at(REG_ZERO) = 0U;  // Register 0 is always zero

    if (this->reg[REG_SP] < this->stackBase - this->maximumStackSize) {
      std::cerr << "Stack Overflow!\n";
    }

    this->executeWriteBack = false;
    this->executeWBReg = -1;

    this->writeBack();
    this->fetch();
    this->decode();
    this->execute();
    this->memoryAccess();

    if (this->fReg.stall == 0U) {
      this->fReg = this->fRegNew;
      this->verbosePrint("Assigned fRegNew to fReg\n");
    } else {
      this->fReg.stall--;
      this->pc -= this->fReg.len;  // Rollback PC if stalled
    }

    if (this->dReg.stall == 0U) {
      this->dReg = this->dRegNew;
      this->verbosePrint("Assigned dRegNew to dReg\n");
    } else {
      this->dReg.stall--;
    }
    this->eReg = this->eRegNew;  // Execute stage would not stall
    this->mReg = this->mRegNew;  // Memory stage would not stall

    fRegNew = {};
    dRegNew = {};
    eRegNew = {};
    mRegNew = {};

    // The Branch prediction happens here to avoid strange bugs in branch
    // prediction
    if (!this->dReg.bubble && (this->dReg.stall == 0U) &&
        (this->fReg.stall == 0U) && this->dReg.predictedBranch) {
      this->pc = this->dReg.predictedPC;
    }

    this->history.cycleCount++;  // Increment cycle count

    if (shouldDumpHistory) {
      this->history.regRecord.push_back(this->getRegInfoStr());
      if (this->history.regRecord.size() >= 100000) {
        // Avoid using up memory
        this->history.regRecord.clear();
        this->history.instRecord.clear();
      }
    }

    if (verbose) {
      this->printInfo();
    }

    if (this->isSingleStep) {
      printf("Type d to dump memory in dump.txt, press ENTER to continue: ");
      char ch;
      while ((ch = getchar()) != '\n') {
        if (ch == 'd') {
          this->dumpHistory();
        }
      }
    }
  }
}

void Simulator::fetch() {
  if (this->pc % 2 != 0) {
    std::cerr << "Illegal PC 0x" << std::hex << this->pc << "!\n";
  }

  const uint32_t instruction = this->memory->getInt(this->pc);
  constexpr uint32_t len = 4;

  if (this->verbose) {
    std::cout << "Fetched instruction: 0x" << std::hex << instruction
              << " at 0x" << std::hex << this->pc << '\n';
  }

  this->fRegNew.bubble = false;
  this->fRegNew.inst = instruction;
  this->fRegNew.len = len;
  this->fRegNew.pc = this->pc;

  this->pc += len;  // Update PC to fetch next instruction
}

void Simulator::decode() {
  if (this->fReg.bubble || this->fReg.inst == 0U) {
    if (verbose) {
      std::cout << "Decode: Bubble\n";
    }
    this->dRegNew.bubble = true;
    return;
  }

  std::string instructionName;
  std::string instructionStr;
  std::string destinationStr;
  std::string op1Str;
  std::string op2Str;
  std::string op3Str;
  std::string offsetStr;
  Instruction instructionType = UNKNOWN;
  const uint32_t instruction = this->fReg.inst;

  // True numbers
  int32_t op1 = 0;
  int32_t op2 = 0;
  int32_t op3 = 0;  // For FMA instructions
  int32_t offset = 0;

  // Register id
  RegId destination = 0;
  RegId reg1 = -1;
  RegId reg2 = -1;
  RegId reg3 = -1;  // For FMA instructions

  const uint32_t opcode = instruction & 0x7F;
  const uint32_t funct3 = (instruction >> 12) & 0x7;
  const uint32_t funct7 = (instruction >> 25) & 0x7F;
  const RegId rd = (instruction >> 7) & 0x1F;
  const RegId rs1 = (instruction >> 15) & 0x1F;
  const RegId rs2 = (instruction >> 20) & 0x1F;
  const RegId rs3 = (instruction >> 27) & 0x1F;  // For FMA instructions
  const int32_t imm_i = static_cast<int32_t>(instruction) >> 20;
  const int32_t imm_s = static_cast<int32_t>(((instruction >> 7) & 0x1F) |
                                             ((instruction >> 20) & 0xFE0))
                            << 20 >>
                        20;
  const int32_t imm_sb = static_cast<int32_t>(((instruction >> 7) & 0x1E) |
                                              ((instruction >> 20) & 0x7E0) |
                                              ((instruction << 4) & 0x800) |
                                              ((instruction >> 19) & 0x1000))
                             << 19 >>
                         19;
  const int32_t imm_u = static_cast<int32_t>(instruction) >> 12;
  const int32_t imm_uj = static_cast<int32_t>(((instruction >> 21) & 0x3FF) |
                                              ((instruction >> 10) & 0x400) |
                                              ((instruction >> 1) & 0x7F800) |
                                              ((instruction >> 12) & 0x80000))
                             << 12 >>
                         11;

  switch (opcode) {
    case OP_FMA: {
      op1 = this->reg.at(rs1);
      op2 = this->reg.at(rs2);
      op3 = this->reg.at(rs3);

      reg1 = rs1;
      reg2 = rs2;
      reg3 = rs3;
      destination = rd;

      const uint32_t rm = (instruction >> 12) & 0x7;
      const uint32_t fmt = (instruction >> 25) & 0x3;

      switch (rm) {
        case 0x0: {  // fmadd, fmaddu, fmsub, fmsubu
          switch (fmt) {
            case 0x0: {  // fmadd
              instructionName = "fmadd";
              instructionType = FMADD;
              break;
            }
            case 0x1: {  // fmaddu
              instructionName = "fmaddu";
              instructionType = FMADDU;
              break;
            }
            case 0x2: {  // fmsub
              instructionName = "fmsub";
              instructionType = FMSUB;
              break;
            }
            case 0x3: {  // fmsubu
              instructionName = "fmsubu";
              instructionType = FMSUBU;
              break;
            }
            default:;
          }
          break;
        }
        case 0x1: {  // fnmadd, fnmsub
          switch (fmt) {
            case 0x0: {  // fnmadd
              instructionName = "fnmadd";
              instructionType = FNMADD;
              break;
            }
            case 0x1: {  // fnmsub
              instructionName = "fnmsub";
              instructionType = FNMSUB;
              break;
            }
            default:;
          }
          break;
        }
        default:;
      }

      op1Str = REGISTER_NAME[rs1];
      op2Str = REGISTER_NAME[rs2];
      op3Str = REGISTER_NAME[rs3];
      destinationStr = REGISTER_NAME[rd];
      instructionStr = instructionName + " " + destinationStr + "," + op1Str +
                       "," + op2Str + "," + op3Str;

      break;
    }
    case OP_REG:
      op1 = this->reg[rs1];
      op2 = this->reg[rs2];

      reg1 = rs1;
      reg2 = rs2;

      destination = rd;
      switch (funct3) {
        case 0x0:  // add, mul, sub
          if (funct7 == 0x00) {
            instructionName = "add";
            instructionType = ADD;
          } else if (funct7 == 0x01) {
            instructionName = "mul";
            instructionType = MUL;
          } else if (funct7 == 0x20) {
            instructionName = "sub";
            instructionType = SUB;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x1:  // sll, mulh
          if (funct7 == 0x00) {
            instructionName = "sll";
            instructionType = SLL;
          } else if (funct7 == 0x01) {
            instructionName = "mulh";
            instructionType = MULH;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x2:  // slt
          if (funct7 == 0x00) {
            instructionName = "slt";
            instructionType = SLT;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x3:  // sltu
          if (funct7 == 0x00) {
            instructionName = "sltu";
            instructionType = SLTU;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x4:  // xor div
          if (funct7 == 0x00) {
            instructionName = "xor";
            instructionType = XOR;
          } else if (funct7 == 0x01) {
            instructionName = "div";
            instructionType = DIV;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x5:  // srl, sra
          if (funct7 == 0x00) {
            instructionName = "srl";
            instructionType = SRL;
          } else if (funct7 == 0x20) {
            instructionName = "sra";
            instructionType = SRA;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x6:  // or, rem
          if (funct7 == 0x00) {
            instructionName = "or";
            instructionType = OR;
          } else if (funct7 == 0x01) {
            instructionName = "rem";
            instructionType = REM;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        case 0x7:  // and
          if (funct7 == 0x00) {
            instructionName = "and";
            instructionType = AND;
          } else {
            this->panic("Unknown funct7 0x%x for funct3 0x%x\n", funct7,
                        funct3);
          }
          break;
        default:
          this->panic("Unknown Funct3 field %x\n", funct3);
      }
      op1Str = REGISTER_NAME[rs1];
      op2Str = REGISTER_NAME[rs2];
      destinationStr = REGISTER_NAME[rd];
      instructionStr =
          instructionName + " " + destinationStr + "," + op1Str + "," + op2Str;
      break;
    case OP_IMM:
      op1 = this->reg[rs1];
      reg1 = rs1;
      op2 = imm_i;
      destination = rd;
      switch (funct3) {
        case 0x0:
          instructionName = "addi";
          instructionType = ADDI;
          break;
        case 0x2:
          instructionName = "slti";
          instructionType = SLTI;
          break;
        case 0x3:
          instructionName = "sltiu";
          instructionType = SLTIU;
          break;
        case 0x4:
          instructionName = "xori";
          instructionType = XORI;
          break;
        case 0x6:
          instructionName = "ori";
          instructionType = ORI;
          break;
        case 0x7:
          instructionName = "andi";
          instructionType = ANDI;
          break;
        case 0x1:
          instructionName = "slli";
          instructionType = SLLI;
          op2 = op2 & 0x3F;
          break;
        case 0x5:
          if (((instruction >> 26) & 0x3F) == 0x0) {
            instructionName = "srli";
            instructionType = SRLI;
            op2 = op2 & 0x3F;
          } else if (((instruction >> 26) & 0x3F) == 0x10) {
            instructionName = "srai";
            instructionType = SRAI;
            op2 = op2 & 0x3F;
          } else {
            this->panic("Unknown funct7 0x%x for OP_IMM\n",
                        (instruction >> 26) & 0x3F);
          }
          break;
        default:
          this->panic("Unknown Funct3 field %x\n", funct3);
      }
      op1Str = REGISTER_NAME[rs1];
      op2Str = std::to_string(op2);
      destinationStr = REGISTER_NAME[destination];
      instructionStr =
          instructionName + " " + destinationStr + "," + op1Str + "," + op2Str;
      break;
    case OP_LUI:
      op1 = imm_u;
      op2 = 0;
      offset = imm_u;
      destination = rd;
      instructionName = "lui";
      instructionType = LUI;
      op1Str = std::to_string(imm_u);
      destinationStr = REGISTER_NAME[destination];
      instructionStr = instructionName + " " + destinationStr + "," + op1Str;
      break;
    case OP_AUIPC:
      op1 = imm_u;
      op2 = 0;
      offset = imm_u;
      destination = rd;
      instructionName = "auipc";
      instructionType = AUIPC;
      op1Str = std::to_string(imm_u);
      destinationStr = REGISTER_NAME[destination];
      instructionStr = instructionName + " " + destinationStr + "," + op1Str;
      break;
    case OP_JAL:
      op1 = imm_uj;
      op2 = 0;
      offset = imm_uj;
      destination = rd;
      instructionName = "jal";
      instructionType = JAL;
      op1Str = std::to_string(imm_uj);
      destinationStr = REGISTER_NAME[destination];
      instructionStr = instructionName + " " + destinationStr + "," + op1Str;
      break;
    case OP_JALR:
      op1 = this->reg[rs1];
      reg1 = rs1;
      op2 = imm_i;
      destination = rd;
      instructionName = "jalr";
      instructionType = JALR;
      op1Str = REGISTER_NAME[rs1];
      op2Str = std::to_string(op2);
      destinationStr = REGISTER_NAME[destination];
      instructionStr =
          instructionName + " " + destinationStr + "," + op1Str + "," + op2Str;
      break;
    case OP_BRANCH:
      op1 = this->reg[rs1];
      op2 = this->reg[rs2];
      reg1 = rs1;
      reg2 = rs2;
      offset = imm_sb;
      switch (funct3) {
        case 0x0:
          instructionName = "beq";
          instructionType = BEQ;
          break;
        case 0x1:
          instructionName = "bne";
          instructionType = BNE;
          break;
        case 0x4:
          instructionName = "blt";
          instructionType = BLT;
          break;
        case 0x5:
          instructionName = "bge";
          instructionType = BGE;
          break;
        case 0x6:
          instructionName = "bltu";
          instructionType = BLTU;
          break;
        case 0x7:
          instructionName = "bgeu";
          instructionType = BGEU;
          break;
        default:
          this->panic("Unknown funct3 0x%x at OP_BRANCH\n", funct3);
      }
      op1Str = REGISTER_NAME[rs1];
      op2Str = REGISTER_NAME[rs2];
      offsetStr = std::to_string(offset);
      instructionStr =
          instructionName + " " + op1Str + "," + op2Str + "," + offsetStr;
      break;
    case OP_STORE:
      op1 = this->reg[rs1];
      op2 = this->reg[rs2];
      reg1 = rs1;
      reg2 = rs2;
      offset = imm_s;
      switch (funct3) {
        case 0x0:
          instructionName = "sb";
          instructionType = SB;
          break;
        case 0x1:
          instructionName = "sh";
          instructionType = SH;
          break;
        case 0x2:
          instructionName = "sw";
          instructionType = SW;
          break;
        case 0x3:
          instructionName = "sd";
          instructionType = SD;
          break;
        default:
          this->panic("Unknown funct3 0x%x for OP_STORE\n", funct3);
      }
      op1Str = REGISTER_NAME[rs1];
      op2Str = REGISTER_NAME[rs2];
      offsetStr = std::to_string(offset);
      instructionStr =
          instructionName + " " + op2Str + "," + offsetStr + "(" + op1Str + ")";
      break;
    case OP_LOAD:
      op1 = this->reg[rs1];
      reg1 = rs1;
      op2 = imm_i;
      offset = imm_i;
      destination = rd;
      switch (funct3) {
        case 0x0:
          instructionName = "lb";
          instructionType = LB;
          break;
        case 0x1:
          instructionName = "lh";
          instructionType = LH;
          break;
        case 0x2:
          instructionName = "lw";
          instructionType = LW;
          break;
        case 0x3:
          instructionName = "ld";
          instructionType = LD;
          break;
        case 0x4:
          instructionName = "lbu";
          instructionType = LBU;
          break;
        case 0x5:
          instructionName = "lhu";
          instructionType = LHU;
          break;
        case 0x6:
          instructionName = "lwu";
          instructionType = LWU;
        default:
          this->panic("Unknown funct3 0x%x for OP_LOAD\n", funct3);
      }
      op1Str = REGISTER_NAME[rs1];
      op2Str = std::to_string(op2);
      destinationStr = REGISTER_NAME[rd];
      instructionStr = instructionName + " " + destinationStr + "," + op2Str +
                       "(" + op1Str + ")";
      break;
    case OP_SYSTEM:
      if (funct3 == 0x0 && funct7 == 0x000) {
        instructionName = "ecall";
        op1 = this->reg[REG_A0];
        op2 = this->reg[REG_A7];
        reg1 = REG_A0;
        reg2 = REG_A7;
        destination = REG_A0;
        instructionType = ECALL;
      } else {
        this->panic("Unknown OP_SYSTEM inst with funct3 0x%x and funct7 0x%x\n",
                    funct3, funct7);
      }
      instructionStr = instructionName;
      break;
    case OP_IMM32:
      op1 = this->reg[rs1];
      reg1 = rs1;
      op2 = imm_i;
      destination = rd;
      switch (funct3) {
        case 0x0:
          instructionName = "addiw";
          instructionType = ADDIW;
          break;
        case 0x1:
          instructionName = "slliw";
          instructionType = SLLIW;
          break;
        case 0x5:
          if (((instruction >> 25) & 0x7F) == 0x0) {
            instructionName = "srliw";
            instructionType = SRLIW;
          } else if (((instruction >> 25) & 0x7F) == 0x20) {
            instructionName = "sraiw";
            instructionType = SRAIW;
          } else {
            this->panic("Unknown shift inst type 0x%x\n",
                        ((instruction >> 25) & 0x7F));
          }
          break;
        default:
          this->panic("Unknown funct3 0x%x for OP_ADDIW\n", funct3);
      }
      op1Str = REGISTER_NAME[rs1];
      op2Str = std::to_string(op2);
      destinationStr = REGISTER_NAME[rd];
      instructionStr =
          instructionName + " " + destinationStr + "," + op1Str + "," + op2Str;
      break;
    case OP_32: {
      op1 = this->reg[rs1];
      op2 = this->reg[rs2];
      reg1 = rs1;
      reg2 = rs2;
      destination = rd;

      uint32_t temp = (instruction >> 25) & 0x7F;  // 32bit funct7 field
      switch (funct3) {
        case 0x0:
          if (temp == 0x0) {
            instructionName = "addw";
            instructionType = ADDW;
          } else if (temp == 0x20) {
            instructionName = "subw";
            instructionType = SUBW;
          } else {
            this->panic("Unknown 32bit funct7 0x%x\n", temp);
          }
          break;
        case 0x1:
          if (temp == 0x0) {
            instructionName = "sllw";
            instructionType = SLLW;
          } else {
            this->panic("Unknown 32bit funct7 0x%x\n", temp);
          }
          break;
        case 0x5:
          if (temp == 0x0) {
            instructionName = "srlw";
            instructionType = SRLW;
          } else if (temp == 0x20) {
            instructionName = "sraw";
            instructionType = SRAW;
          } else {
            this->panic("Unknown 32bit funct7 0x%x\n", temp);
          }
          break;
        default:
          this->panic("Unknown 32bit funct3 0x%x\n", funct3);
      }
    } break;
    default:
      this->panic("Unsupported opcode 0x%x!\n", opcode);
  }

  char buf[4096];
  sprintf(buf, "0x%llx: %s\n", this->fReg.pc, instructionStr.c_str());
  this->history.instRecord.push_back(buf);

  if (verbose) {
    printf("Decoded instruction 0x%.8x as %s\n", instruction,
           instructionStr.c_str());
  }

  if (instructionName != INSTNAME[instructionType]) {
    this->panic("Unmatch instname %s with insttype %d\n",
                instructionName.c_str(), instructionType);
  }

  bool predictedBranch = false;
  if (isBranch(instructionType)) {
    predictedBranch = this->branchPredictor->predict(
        this->fReg.pc, instructionType, op1, op2, offset);
    if (predictedBranch) {
      this->dRegNew.predictedPC = this->fReg.pc + offset;
      this->dRegNew.anotherPC = this->fReg.pc + 4;
      this->fRegNew.bubble = true;
    } else {
      this->dRegNew.anotherPC = this->fReg.pc + offset;
    }
  }

  this->dRegNew.rawAssemblyInstruction = instructionStr;
  this->dRegNew.bubble = false;
  this->dRegNew.rs1 = reg1;
  this->dRegNew.rs2 = reg2;
  this->dRegNew.rs3 = reg3;
  this->dRegNew.pc = this->fReg.pc;
  this->dRegNew.inst = instructionType;
  this->dRegNew.predictedBranch = predictedBranch;
  this->dRegNew.dest = destination;
  this->dRegNew.op1 = op1;
  this->dRegNew.op2 = op2;
  this->dRegNew.op3 = op3;
  this->dRegNew.offset = offset;
}

void Simulator::execute() {
  if (this->dReg.stall != 0U || this->dReg.bubble) {
    if (verbose) {
      std::cout << "Execute: Bubble\n";
    }
    this->eRegNew.bubble = true;
    return;
  }

  this->verbosePrint(
      std::format("Execute: {}\n", this->dReg.rawAssemblyInstruction));

  this->history.instructionCount++;

  const Instruction inst = this->dReg.inst;
  int32_t op1 = this->dReg.op1;
  int32_t op2 = this->dReg.op2;
  int32_t op3 = this->dReg.op3;
  int32_t offset = this->dReg.offset;
  bool predictedBranch = this->dReg.predictedBranch;

  uint32_t dRegPC = this->dReg.pc;
  bool enableWriteRegister = false;
  const RegId destReg = this->dReg.dest;
  int32_t out = 0;
  bool enableWriteMemory = false;
  bool enableReadMemory = false;
  bool readSignExt = false;
  uint32_t memLen = 0;
  bool branch = false;

  switch (inst) {
    case FMADD: {
      enableWriteRegister = true;
      out = op1 * op2 + op3;
      this->history.cycleCount += 3;
      break;
    }
    case FMADDU: {
      enableWriteRegister = true;
      out = static_cast<uint32_t>(op1) * static_cast<uint32_t>(op2) + op3;
      this->history.cycleCount += 3;
      break;
    }
    case FMSUB: {
      enableWriteRegister = true;
      out = op1 * op2 - op3;
      this->history.cycleCount += 3;
      break;
    }
    case FMSUBU: {
      enableWriteRegister = true;
      out = op1 * op2 - op3;
      this->history.cycleCount += 3;
      break;
    }
    case FNMADD: {
      enableWriteRegister = true;
      out = -(op1 * op2) + op3;
      this->history.cycleCount += 3;
      break;
    }
    case FNMSUB: {
      enableWriteRegister = true;
      out = -(op1 * op2) - op3;
      this->history.cycleCount += 3;
      break;
    }
    case LUI:
      enableWriteRegister = true;
      out = offset << 12;
      break;
    case AUIPC:
      enableWriteRegister = true;
      out = dRegPC + (offset << 12);
      break;
    case JAL:
      enableWriteRegister = true;
      out = dRegPC + 4;
      dRegPC = dRegPC + op1;
      branch = true;
      break;
    case JALR:
      enableWriteRegister = true;
      out = dRegPC + 4;
      dRegPC = (op1 + op2) & (~(uint32_t)1);
      branch = true;
      break;
    case BEQ:
      if (op1 == op2) {
        branch = true;
        dRegPC = dRegPC + offset;
      }
      break;
    case BNE:
      if (op1 != op2) {
        branch = true;
        dRegPC = dRegPC + offset;
      }
      break;
    case BLT:
      if (op1 < op2) {
        branch = true;
        dRegPC = dRegPC + offset;
      }
      break;
    case BGE:
      if (op1 >= op2) {
        branch = true;
        dRegPC = dRegPC + offset;
      }
      break;
    case BLTU:
      if ((uint32_t)op1 < (uint32_t)op2) {
        branch = true;
        dRegPC = dRegPC + offset;
      }
      break;
    case BGEU:
      if ((uint32_t)op1 >= (uint32_t)op2) {
        branch = true;
        dRegPC = dRegPC + offset;
      }
      break;
    case LB:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 1;
      out = op1 + offset;
      readSignExt = true;
      break;
    case LH:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 2;
      out = op1 + offset;
      readSignExt = true;
      break;
    case LW:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 4;
      out = op1 + offset;
      readSignExt = true;
      break;
    case LD:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 8;
      out = op1 + offset;
      readSignExt = true;
      break;
    case LBU:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 1;
      out = op1 + offset;
      break;
    case LHU:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 2;
      out = op1 + offset;
      break;
    case LWU:
      enableReadMemory = true;
      enableWriteRegister = true;
      memLen = 4;
      out = op1 + offset;
      break;
    case SB:
      enableWriteMemory = true;
      memLen = 1;
      out = op1 + offset;
      op2 = op2 & 0xFF;
      break;
    case SH:
      enableWriteMemory = true;
      memLen = 2;
      out = op1 + offset;
      op2 = op2 & 0xFFFF;
      break;
    case SW:
      enableWriteMemory = true;
      memLen = 4;
      out = op1 + offset;
      op2 = op2 & 0xFFFFFFFF;
      break;
    case SD:
      enableWriteMemory = true;
      memLen = 8;
      out = op1 + offset;
      break;
    case ADDI:
    case ADD:
      enableWriteRegister = true;
      out = op1 + op2;
      break;
    case ADDIW:
    case ADDW:
      enableWriteRegister = true;
      out = (int32_t)((int32_t)op1 + (int32_t)op2);
      break;
    case SUB:
      enableWriteRegister = true;
      out = op1 - op2;
      break;
    case SUBW:
      enableWriteRegister = true;
      out = (int32_t)((int32_t)op1 - (int32_t)op2);
      break;
    case MUL:
      enableWriteRegister = true;
      out = op1 * op2;
      this->history.cycleCount += 3;
      break;
    case DIV:
      enableWriteRegister = true;
      out = op1 / op2;
      break;
    case SLTI:
    case SLT:
      enableWriteRegister = true;
      out = op1 < op2 ? 1 : 0;
      break;
    case SLTIU:
    case SLTU:
      enableWriteRegister = true;
      out = (uint32_t)op1 < (uint32_t)op2 ? 1 : 0;
      break;
    case XORI:
    case XOR:
      enableWriteRegister = true;
      out = op1 ^ op2;
      break;
    case ORI:
    case OR:
      enableWriteRegister = true;
      out = op1 | op2;
      break;
    case ANDI:
    case AND:
      enableWriteRegister = true;
      out = op1 & op2;
      break;
    case SLLI:
    case SLL:
      enableWriteRegister = true;
      out = op1 << op2;
      break;
    case SLLIW:
    case SLLW:
      enableWriteRegister = true;
      out = int32_t(int32_t(op1 << op2));
      break;
      break;
    case SRLI:
    case SRL:
      enableWriteRegister = true;
      out = (uint32_t)op1 >> (uint32_t)op2;
      break;
    case SRLIW:
    case SRLW:
      enableWriteRegister = true;
      out = uint32_t(uint32_t((uint32_t)op1 >> (uint32_t)op2));
      break;
    case SRAI:
    case SRA:
      enableWriteRegister = true;
      out = op1 >> op2;
      break;
    case SRAW:
    case SRAIW:
      enableWriteRegister = true;
      out = int32_t(int32_t((int32_t)op1 >> (int32_t)op2));
      break;
    case ECALL:
      out = handleSystemCall(op1, op2);
      enableWriteRegister = true;
      break;
    default:
      this->panic("Unknown instruction type %d\n", inst);
  }

  // Pipeline Related Code
  if (isBranch(inst)) {
    if (predictedBranch == branch) {
      this->history.predictedBranch++;
      this->verbosePrint(std::format(
          "Branch Predicted: PC = 0x{:x} -> 0x{:x}\n", dRegPC, this->pc));
    } else {
      // Control Hazard Here
      this->pc = this->dReg.anotherPC;
      this->fRegNew.bubble = true;
      this->dRegNew.bubble = true;
      this->history.unpredictedBranch++;
      this->history.controlHazardCount++;
      this->verbosePrint(std::format("Control Hazard: PC = 0x{:x} -> 0x{:x}\n",
                                     dRegPC, this->pc));
    }
    // this->dReg.pc: fetch original inst addr, not the modified one
    this->branchPredictor->update(this->dReg.pc, branch);
  }
  if (isJump(inst)) {
    // Control hazard here
    this->pc = dRegPC;
    this->fRegNew.bubble = true;
    this->dRegNew.bubble = true;
    this->history.controlHazardCount++;
    this->verbosePrint(std::format("Control Hazard: PC = 0x{:x} -> 0x{:x}\n",
                                   dRegPC, this->pc));
  }
  if (isReadMem(inst)) {
    if (this->dRegNew.rs1 == destReg || this->dRegNew.rs2 == destReg ||
        this->dRegNew.rs3 == destReg) {
      if (dataForwarding) {
        this->fRegNew.stall = 2;
        this->dRegNew.stall = 2;
        this->history.cycleCount--;
      } else {
        this->fReg.stall = 2;
        this->dReg.stall = 2;
        this->dReg.bubble = true;
      }
      this->history.memoryHazardCount++;
      this->verbosePrint("EXE stage detected memory hazard\n");
    }
  }

  // Check for data hazard and forward data
  if (enableWriteRegister && destReg != REG_ZERO && !isReadMem(inst)) {
    if (dataForwarding) {
      if (this->dRegNew.rs1 == destReg) {
        this->dRegNew.op1 = out;
        this->executeWBReg = destReg;
        this->executeWriteBack = true;
        this->history.dataHazardCount++;
        if (verbose) {
          std::cout << "  Forward Data " << REGISTER_NAME.at(destReg)
                    << " to Decode op1\n";
        }
      }
      if (this->dRegNew.rs2 == destReg) {
        this->dRegNew.op2 = out;
        this->executeWBReg = destReg;
        this->executeWriteBack = true;
        this->history.dataHazardCount++;
        if (verbose) {
          std::cout << "  Forward Data " << REGISTER_NAME.at(destReg)
                    << " to Decode op2\n";
        }
      }
      if (this->dRegNew.rs3 == destReg) {
        this->dRegNew.op3 = out;
        this->executeWBReg = destReg;
        this->executeWriteBack = true;
        this->history.dataHazardCount++;
        if (verbose) {
          std::cout << "  Forward Data " << REGISTER_NAME.at(destReg)
                    << " to Decode op3\n";
        }
      }
    } else {  // w/o data forwarding
      if (this->dRegNew.rs1 == destReg || this->dRegNew.rs2 == destReg ||
          this->dRegNew.rs3 == destReg) {
        this->verbosePrint(
            "EXE stage detected data hazard (w/o data forwarding)\n");
        this->fReg.stall = 2;
        this->dReg.stall = 2;
        this->dReg.bubble = true;
        this->history.dataHazardCount++;
      }
    }
  }

  this->eRegNew.bubble = false;
  this->eRegNew.pc = dRegPC;
  this->eRegNew.inst = inst;
  this->eRegNew.rawAssemblyInstruction = this->dReg.rawAssemblyInstruction;
  this->eRegNew.op2 = op2;
  this->eRegNew.enableWriteRegister = enableWriteRegister;
  this->eRegNew.destReg = destReg;
  this->eRegNew.out = out;
  this->eRegNew.enableWriteMemory = enableWriteMemory;
  this->eRegNew.enableReadMemory = enableReadMemory;
  this->eRegNew.readSignExt = readSignExt;
  this->eRegNew.memLen = memLen;
}

void Simulator::memoryAccess() {
  if (this->eReg.stall != 0U || this->eReg.bubble) {  // Ex bubble pass to Mem
    this->mRegNew.bubble = true;
    this->verbosePrint("Memory Access: Bubble\n");
    return;
  }

  const Instruction inst = this->eReg.inst;
  const bool enableWriteBack2Register = this->eReg.enableWriteRegister;
  const RegId destReg = this->eReg.destReg;
  const int32_t op2 = this->eReg.op2;  // for store
  int32_t out = this->eReg.out;
  const bool enableWriteMemory = this->eReg.enableWriteMemory;
  const bool enableReadMemory = this->eReg.enableReadMemory;
  const bool readSignExt = this->eReg.readSignExt;
  const uint32_t memLen = this->eReg.memLen;

  bool good = true;
  uint32_t memoryOperationCycleCount = 0;

  if (enableWriteMemory) {
    switch (memLen) {
      case 1: {
        good = this->memory->setByte(out, op2, &memoryOperationCycleCount);
        break;
      }
      case 2: {
        good = this->memory->setShort(out, op2, &memoryOperationCycleCount);
        break;
      }
      case 4: {
        good = this->memory->setInt(out, op2, &memoryOperationCycleCount);
        break;
      }
      default: {
        std::cerr << "Unknown memLen " << memLen << '\n';
      }
    }
  }

  if (!good) {
    std::cerr << "Invalid Mem Access!\n";
  }

  if (enableReadMemory) {
    switch (memLen) {
      case 1: {
        if (readSignExt) {
          out = static_cast<int32_t>(
              this->memory->getByte(out, &memoryOperationCycleCount));
        } else {
          out = static_cast<uint32_t>(
              this->memory->getByte(out, &memoryOperationCycleCount));
        }
        break;
      }
      case 2: {
        if (readSignExt) {
          out = static_cast<int32_t>(
              this->memory->getShort(out, &memoryOperationCycleCount));
        } else {
          out = static_cast<uint32_t>(
              this->memory->getShort(out, &memoryOperationCycleCount));
        }
        break;
      }
      case 4: {
        if (readSignExt) {
          out = static_cast<int32_t>(
              this->memory->getInt(out, &memoryOperationCycleCount));
        } else {
          out = this->memory->getInt(out, &memoryOperationCycleCount);
        }
        break;
      }
      default:
        std::cerr << "Unknown memLen " << memLen << '\n';
    }
  }

  this->history.cycleCount += memoryOperationCycleCount;

  this->verbosePrint(
      std::format("Memory Access: {}\n", this->eReg.rawAssemblyInstruction));

  if (enableWriteBack2Register && destReg != REG_ZERO) {
    if (dataForwarding) {
      if (!this->executeWriteBack || this->executeWBReg != destReg) {
        if (this->dRegNew.rs1 == destReg) {
          this->dRegNew.op1 = out;
          this->history.dataHazardCount++;
          this->verbosePrint(std::format("  Forward Data {} to Decode op1\n",
                                         REGISTER_NAME.at(destReg)));
        }
        if (this->dRegNew.rs2 == destReg) {
          this->dRegNew.op2 = out;
          this->history.dataHazardCount++;
          this->verbosePrint(std::format("  Forward Data {} to Decode op2\n",
                                         REGISTER_NAME.at(destReg)));
        }
        if (this->dRegNew.rs3 == destReg) {
          this->dRegNew.op3 = out;
          this->history.dataHazardCount++;
          this->verbosePrint(std::format("  Forward Data {} to Decode op3\n",
                                         REGISTER_NAME.at(destReg)));
        }
      }

      // Corner case of forwarding mem load data to stalled decode reg
      if (this->dReg.stall != 0U) {
        if (this->dReg.rs1 == destReg) {
          this->dReg.op1 = out;
          this->history.dataHazardCount++;
          this->verbosePrint(std::format("  Forward Data {} to Decode op1\n",
                                         REGISTER_NAME.at(destReg)));
        }
        if (this->dReg.rs2 == destReg) {
          this->dReg.op2 = out;
          this->history.dataHazardCount++;
          this->verbosePrint(std::format("  Forward Data {} to Decode op2\n",
                                         REGISTER_NAME.at(destReg)));
        }
        if (this->dReg.rs3 == destReg) {
          this->dReg.op3 = out;
          this->history.dataHazardCount++;
          this->verbosePrint(std::format("  Forward Data {} to Decode op3\n",
                                         REGISTER_NAME.at(destReg)));
        }
      }
    } else {
      if (!this->dRegNew.bubble &&
          (this->dRegNew.rs1 == destReg || this->dRegNew.rs2 == destReg ||
           this->dRegNew.rs3 == destReg)) {
        this->verbosePrint(
            "MEM stage detected data hazard (w/o data forwarding)\n");
        this->fReg.stall = 1;
        this->dReg.stall = 1;
        this->dReg.bubble = true;
        this->history.dataHazardCount++;
      }
    }
  }

  this->mRegNew.bubble = false;
  this->mRegNew.inst = inst;
  this->mRegNew.rawAssemblyInstruction = this->eReg.rawAssemblyInstruction;
  this->mRegNew.destReg = destReg;
  this->mRegNew.enableWriteBack = enableWriteBack2Register;
  this->mRegNew.out = out;
}

void Simulator::writeBack() {
  if (this->mReg.bubble) {
    this->verbosePrint("WriteBack: Bubble\n");
    return;
  }

  this->verbosePrint(
      std::format("WriteBack: {}\n", this->mReg.rawAssemblyInstruction));

  if (this->mReg.enableWriteBack && this->mReg.destReg != REG_ZERO) {
    this->reg.at(mReg.destReg) = this->mReg.out;
  }
}

int32_t Simulator::handleSystemCall(int32_t op1, int32_t op2) {
  int32_t type = op2;  // reg a7
  int32_t arg1 = op1;  // reg a0
  switch (type) {
    case 0: {
      // print string
      uint32_t addr = arg1;
      char ch = this->memory->getByte(addr);
      while (ch != '\0') {
        printf("%c", ch);
        ch = this->memory->getByte(++addr);
      }
      break;
    }
    case 1:  // print char
      printf("%c", (char)arg1);
      break;
    case 2:  // print num
      printf("%d", (int32_t)arg1);
      break;
    case 3:
    case 93:  // exit
      printf("Program exit from an exit() system call\n");
      if (shouldDumpHistory) {
        printf("Dumping history to dump.txt...");
        this->dumpHistory();
      }
      this->printStatistics();
      exit(0);
    case 4:  // read char
      scanf(" %c", (char*)&op1);
      break;
    case 5:  // read num
      scanf(" %lld", &op1);
      break;
    default:
      this->panic("Unknown syscall type %d\n", type);
  }
  return op1;
}

void Simulator::printInfo() {
  printf("------------ CPU STATE ------------\n");
  printf("PC: 0x%llx\n", this->pc);
  for (uint32_t i = 0; i < 32; ++i) {
    printf("%s: 0x%.8llx(%lld) ", REGISTER_NAME[i], this->reg[i], this->reg[i]);
    if (i % 4 == 3) printf("\n");
  }
  printf("-----------------------------------\n");
}

void Simulator::printStatistics() const {
  printf("------------ STATISTICS -----------\n");
  printf("Number of Instructions: %u\n", this->history.instructionCount);
  printf("Number of Cycles: %u\n", this->history.cycleCount);
  printf("Avg Cycles per Instrcution: %.4f\n",
         (float)this->history.cycleCount / this->history.instructionCount);
  printf("Branch Perdiction Accuacy: %.4f (Strategy: %s)\n",
         (float)this->history.predictedBranch /
             (this->history.predictedBranch + this->history.unpredictedBranch),
         this->branchPredictor->strategyName().c_str());
  printf("Number of Control Hazards: %u\n", this->history.controlHazardCount);
  printf("Number of Data Hazards: %u\n", this->history.dataHazardCount);
  printf("Number of Memory Hazards: %u\n", this->history.memoryHazardCount);
  printf("-----------------------------------\n");
  // this->memory->printStatistics();
}

std::string Simulator::getRegInfoStr() {
  std::string str;
  char buf[65536];

  str += "------------ CPU STATE ------------\n";
  sprintf(buf, "PC: 0x%llx\n", this->pc);
  str += buf;
  for (uint32_t i = 0; i < 32; ++i) {
    sprintf(buf, "%s: 0x%.8llx(%lld) ", REGISTER_NAME[i], this->reg[i],
            this->reg[i]);
    str += buf;
    if (i % 4 == 3) {
      str += "\n";
    }
  }
  str += "-----------------------------------\n";

  return str;
}

void Simulator::dumpHistory() {
  std::ofstream ofile("dump.txt");
  ofile << "================== Excecution History =================="
        << std::endl;
  for (uint32_t i = 0; i < this->history.instRecord.size(); ++i) {
    ofile << this->history.instRecord[i];
    ofile << this->history.regRecord[i];
  }
  ofile << "========================================================"
        << std::endl;
  ofile << std::endl;

  ofile << "====================== Memory Dump ======================"
        << std::endl;
  ofile << this->memory->dumpMemory();
  ofile << "========================================================="
        << std::endl;
  ofile << std::endl;

  ofile.close();
}

void Simulator::panic(const char* format, ...) {
  char buf[BUFSIZ];
  va_list args;
  va_start(args, format);
  vsprintf(buf, format, args);
  fprintf(stderr, "%s", buf);
  va_end(args);
  this->dumpHistory();
  fprintf(stderr, "Execution history and memory dump in dump.txt\n");
  exit(-1);
}

void Simulator::verbosePrint(const std::string& str) const {
  if (verbose) {
    std::cout << str;
  }
}
