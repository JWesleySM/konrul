//opt -load-pass-plugin libArithOpSim.so -passes=arith-op-sim -disable-output

#include <map>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

// No need to expose the internal so fht pass to the outside world -
// keep everything in an anonymous namespace.

namespace{
  
std::map<llvm::StringRef, char> NormalizedVars;

bool isArithmetic(llvm::Instruction* I){
  if(!I->isBinaryOp())
    return false;
  
  switch (I->getOpcode()){
    case llvm::Instruction::Add  :
    case llvm::Instruction::FAdd :
    case llvm::Instruction::Sub  :
    case llvm::Instruction::FSub :
    case llvm::Instruction::Mul  :
    case llvm::Instruction::FMul :
    case llvm::Instruction::UDiv :
    case llvm::Instruction::SDiv :
    case llvm::Instruction::FDiv :
      return true;

    default:
      return false;
  }
}
  
  
bool isUsedAsIndex(llvm::Value* V, std::set<llvm::Value*>* Visited){
  if(Visited->find(V) != Visited->end())
    return false;
  
  Visited->insert(V);
  
  if(llvm::Instruction* VInst = llvm::dyn_cast<llvm::Instruction>(V)){
    if(llvm::isa<llvm::GetElementPtrInst>(VInst) || llvm::isa<llvm::CmpInst>(VInst))
        return true;
  }
    
  for(llvm::User* U : V->users()){
    if(isUsedAsIndex(U, Visited))
      return true;
  }
    
  return false;
}


char getNormalizedName(llvm::Value* V, int* base){
  if(NormalizedVars.find(V->getName()) == NormalizedVars.end()){
    NormalizedVars[V->getName()] = char(*base);
    (*base)++;
  }
  
  return NormalizedVars[V->getName()];
}

void getVariableUsedInInstruction(llvm::Value* V, std::set<llvm::Value*>* Variables, std::set<llvm::Value*>*Visited){
  if(Visited->find(V) != Visited->end())
    return;
  
  Visited->insert(V);
  if(llvm::Instruction* VInst = llvm::dyn_cast<llvm::Instruction>(V)){
    if(llvm::isa<llvm::GetElementPtrInst>(VInst))
      getVariableUsedInInstruction(VInst->getOperand(0), Variables, Visited);
    else{
      for(llvm::Use& U : VInst->operands())
        getVariableUsedInInstruction(U.get(), Variables, Visited);
    }
  }
  else{
    if(V->hasName())
      Variables->insert(V);
  }
}

void getArithInstsOpcode2(llvm::Loop* L){
  int Base = 98;
  for(llvm::BasicBlock* BB : L->getBlocks()){
    if(BB == L->getHeader())
      continue;
    
    for(llvm::Instruction& I : *BB){
      if(I.isTerminator())
        continue;
              
      if(I.mayReadOrWriteMemory())
         continue;

      if(I.getOpcode() == llvm::Instruction::GetElementPtr)
        continue;
      
      if(I.getOpcode() == llvm::Instruction::PHI)
        continue;
      
      if(llvm::isa<llvm::CmpInst>(I))
        continue;
      
      std::set<llvm::Value*> Visited;
      if(isUsedAsIndex(&I, &Visited))
          continue;
      
      
      std::set<llvm::Value*> Vars;
      std::set<llvm::Value*> VisitedVars;
      getVariableUsedInInstruction(&I, &Vars, &VisitedVars);
      
      llvm::outs() << I.getOpcodeName() << " ";
      for(auto& V : Vars){
        llvm::outs() << getNormalizedName(V, &Base) << " ";
      }
      
    }
  }

}

std::vector<const char*> getArithInstsOpcode(llvm::Loop* L){
  std::vector<const char*> OpCodes;
  int Base = 98;
  for(llvm::BasicBlock* BB : L->getBlocks()){
    if(BB == L->getHeader())
      continue;
    
    for(llvm::Instruction& I : *BB){
      if(I.isTerminator())
        continue;
              
      if(I.mayReadOrWriteMemory())
         continue;

      if(I.getOpcode() == llvm::Instruction::GetElementPtr)
        continue;
      
      if(I.getOpcode() == llvm::Instruction::PHI)
        continue;
      
      if(llvm::isa<llvm::CmpInst>(I))
        continue;
      
      std::set<llvm::Value*> Visited;
      if(isUsedAsIndex(&I, &Visited))
          continue;
      
      std::set<llvm::Value*> Vars;
      std::set<llvm::Value*> VisitedVars;
      getVariableUsedInInstruction(&I, &Vars, &VisitedVars);
      OpCodes.push_back(I.getOpcodeName());
      
      
      for(auto& V : Vars){
        char* NormalizedName = new char[0];
        *NormalizedName = getNormalizedName(V, &Base);
        OpCodes.push_back(NormalizedName);
      }
      
    }
  }

  return OpCodes;
}


llvm::Loop* getInnermostLoop(llvm::Loop* L){
  if(L->isInnermost()){
    return L;
  }
  else{
    for(llvm::Loop* SubL : L->getSubLoops()){
      if(getInnermostLoop(SubL))
        return SubL;
    }
  }

  return nullptr;
}


void visitor(llvm::Function& F, llvm::LoopInfo& LI){
  for(llvm::Loop* L : LI){
    getArithInstsOpcode2(getInnermostLoop(L));
  }
} 

// New Pass Manager implementation
struct ArithOpSim : llvm::PassInfoMixin<ArithOpSim>{
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if needed)

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM){
    llvm::LoopInfo &LI = FAM.getResult<llvm::LoopAnalysis>(F);
    visitor(F, LI);

    return llvm::PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated eith the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.

  static bool isRequired() { return true; }
};

} // End of namespace

// New Pass Manager implementation
llvm::PassPluginLibraryInfo getArithOpSimPluginInfo(){
  return {LLVM_PLUGIN_API_VERSION, "ArithOpSim", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB){
            PB.registerPipelineParsingCallback(
              [](llvm::StringRef Name, llvm::FunctionPassManager &FPM, llvm::ArrayRef<llvm::PassBuilder::PipelineElement>){
                if(Name == "arith-op-sim"){
                  FPM.addPass(ArithOpSim());
                  return true;
                }

                return false;
              });
            }
          };
}

// This is the core interface for pass pluginsl It guarantees that 'opt; will
// be able to recognize ArithOpSim when added to the pass pipeline on the
// command line, i.e., via '-passes=test-pass'
extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo llvmGetPassPluginInfo(){
  return getArithOpSimPluginInfo();
}

