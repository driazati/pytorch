#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/symbolic_variable.h"

namespace torch { namespace jit {


// propagate undefined information through a gradient graph and
// remove grad_of blocks if present.
// Note: this is a very limited pass. It only propagates undefines for
// operations generated by the symbolic autodiff code and cleans up
// AutogradAdds when possible. Outputs of other nodes are conservatively
// marked Unknown and not optimized.
void specializeUndef(Graph & g) {
  enum class State { Defined, Undefined, Unknown };
  std::unordered_map<Value*, State> state;

  auto inputs = g.inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tp = inputs[i]->type();
    if (tp->isSubtypeOf(UndefinedTensorType::get())) {
      state[inputs[i]] = State::Undefined;
    } else if (tp->isSubtypeOf(DynamicType::get())) {
      state[inputs[i]] = State::Defined;
    } else {
      state[inputs[i]] = State::Unknown;
    }
  }

  for(auto it = g.nodes().begin(); it != g.nodes().end(); ++it) {
    auto n = *it;
    switch(n->kind()) {
      case prim::GradOf: {
        auto all_undefined =
            std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
              return state[v] == State::Undefined;
            });
        // Property 1: if all the gradInputs to the GradOf are undefined
        // then the gradOutputs are also zero and will be represented as undefined nodes
        if(all_undefined) {
          auto undef = g.createUndefined()->insertAfter(n)->output();
          for(auto o : n->outputs()) {
            o->replaceAllUsesWith(undef);
          }
        } else {
        // Property 2: GradOfs are required to correctly handle combinations
        // of defined and undefined inputs. They are expected to produce defined
        // output tensors in this case.

          // Remove the GradOf, splicing its body back into the surrounding block
          auto body = n->blocks().at(0);
          for(auto input : n->inputs()){
            // we should never get into a situation when specializing a GradOf
            // where we do not know if a value is defined since at the top level
            // a gradient graph is composed of Linear nodes and AutogradAdds
            // and LinearNodes only appear in these graphs
            JIT_ASSERT(state[input] != State::Unknown);
          }
          // hoist the nodes in the GradOf body to be before the linear block
          for(auto it = body->nodes().begin(); it != body->nodes().end();) {
            auto block_node = *it++;
            block_node->moveBefore(n);
          }

          for(size_t i = 0; i < n->outputs().size(); ++i)
            n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
        }
        it.destroyCurrent();
      } break;
      case prim::AutogradAdd: {
        auto a = n->input(0);
        auto b = n->input(1);
        // if one is undefined, we can just drop the add
        if(state[a] == State::Undefined) {
          // Undef + b == b
          n->output()->replaceAllUsesWith(b);
          it.destroyCurrent();
        } else if(state[b] == State::Undefined) {
          // a + Undef == a
          n->output()->replaceAllUsesWith(a);
          it.destroyCurrent();
        } else if(state[a] == State::Defined && state[b] == State::Defined) {
          // when both are defined, we can use a normal, optimizable add instruction
          WithInsertPoint guard(n);
          Value* new_add = toVar(a) + toVar(b);
          state[new_add] = State::Defined;
          n->output()->replaceAllUsesWith(new_add);
          it.destroyCurrent();
        } else {
          // otherwise we have conditionally-defined things, and we need
          // to actually run an AutogradAdd which will guard for undefs
          // so we leave the op as is
          state[n->output()] = State::Unknown;
        }
      } break;
      case prim::Undefined: {
        state[n->output()] = State::Undefined;
      } break;
      default:
        for(auto o : n->outputs()) {
          state[o] = State::Unknown;
        }
        break;
    }
  }
}

}}
