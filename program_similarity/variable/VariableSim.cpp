//opt -load-pass-plugin libVariableSim.so -passes=var-sim -disable-output

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/InstIterator.h" // To use the instructions iterator
#include "llvm/Support/raw_ostream.h"


// No need to expose the internal so fht pass to the outside world -
// keep everything in an anonymous namespace.

namespace{

bool isUsedByGep(llvm::Value* V){
  for(llvm::User* U : V->users()){
    if(llvm::Instruction* UInst = llvm::dyn_cast<llvm::Instruction>(U)){
      if(llvm::isa<llvm::GetElementPtrInst>(UInst))
        return true;
    }
  }
  
  return false;
}
  
  
bool isScalar(llvm::Value* V){
  llvm::Type* VType = V->getType();
  if(!VType->isPointerTy())
    return true;
  
  VType = VType->getPointerElementType();
  if(VType->isStructTy()){
    // For a taco_tensor_t object, if the value has one use, it means only its 
    // 'vals' field is accessed. Therefore, no 'dims' fields are acessed which
    // menas that object represents a scalar
    if(VType->getStructName().str() == "struct.taco_tensor_t"){
      if(V->hasOneUse()){
        llvm::GetElementPtrInst* GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V->user_back());
        llvm::BitCastInst* BCI = llvm::dyn_cast<llvm::BitCastInst>(GEP->user_back());
        return !isUsedByGep(BCI->user_back());
      }
    }
  }
    
  return false;
}


void printNormalizedNames(std::set<llvm::Value*> Variables, int* base){
  for(llvm::Value* V : Variables){
    if(!isScalar(V))
      llvm::outs() << "t_";
    llvm::outs() << char(*base) << " ";
    (*base)++;
  }
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


void visitor(llvm::Function& F){
  std::set<llvm::Value*> Loaded;
  std::set<llvm::Value*> Stored;
  std::set<llvm::Value*> Visited;
  
  for(llvm::Instruction& I : instructions(F)){
    // Only inspect instructions relevants to the computation, i.e., inside loops
    if(I.getParent()->getName() == "entry")
      continue;
    
    Visited.clear();
    if(I.mayReadOrWriteMemory()){

      if(I.getOpcode() == llvm::Instruction::Load){
        getVariableUsedInInstruction(I.getOperand(0), &Loaded, &Visited);
        
      }
      if(I.getOpcode() == llvm::Instruction::Store){
        getVariableUsedInInstruction(I.getOperand(0), &Loaded, &Visited);
        Visited.clear();
        getVariableUsedInInstruction(I.getOperand(1), &Stored, &Visited);
      }
     }
     
    if(llvm::ReturnInst* RI = llvm::dyn_cast<llvm::ReturnInst>(&I)){
      llvm::Value* RetValue = RI->getReturnValue();
      if(RetValue && !llvm::isa<llvm::ConstantInt>(RetValue)){
        Stored.insert(RI);
        getVariableUsedInInstruction(RetValue, &Loaded, &Visited);
      }
    }
  }

  int base = 97;
  printNormalizedNames(Stored, &base);
  printNormalizedNames(Loaded, &base);
} 

// New Pass Manager implementation
struct VariableSim : llvm::PassInfoMixin<VariableSim>{
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if needed)

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM){
    visitor(F);
    return llvm::PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated eith the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.

  static bool isRequired() { return true; }
};

} // End of namespace

// New Pass Manager implementation
llvm::PassPluginLibraryInfo getVariableSimPluginInfo(){
  return {LLVM_PLUGIN_API_VERSION, "VariableSim", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB){
            PB.registerPipelineParsingCallback(
              [](llvm::StringRef Name, llvm::FunctionPassManager &FPM, llvm::ArrayRef<llvm::PassBuilder::PipelineElement>){
                if(Name == "var-sim"){
                  FPM.addPass(VariableSim());
                  return true;
                }

                return false;
              });
            }
          };
}

// This is the core interface for pass pluginsl It guarantees that 'opt; will
// be able to recognize VariableSim when added to the pass pipeline on the
// command line, i.e., via '-passes=test-pass'
extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo llvmGetPassPluginInfo(){
  return getVariableSimPluginInfo();
}
