import struct

import flatbuffers
import gen.TsetlinMachine.AutomatonStatesTensor as AutomatonStatesTensor
import gen.TsetlinMachine.ClauseWeightsTensor as ClauseWeightsTensor
import gen.TsetlinMachine.TMParameters as TMParameters
import gen.TsetlinMachine.TsetlinMachineModel as TsetlinMachineModel
import green_tsetlin as gt
import numpy as np




def save_to_bin(tm: gt.TsetlinMachine, filename: str):
    threshold: int = tm.threshold
    n_literals: int = tm.n_literals
    n_clauses: int = tm.n_clauses
    n_classes: int = tm.n_classes
    max_state: int = 127    # hardcoded in green_tsetlin/src/func_tm.hpp
    min_state: int = -127   # hardcoded in green_tsetlin/src/func_tm.hpp
    boost_true_positive_feedback: int = int(tm.boost_true_positives)
    s: float = tm.s[0]

    weights: np.ndarray = tm._state.w  # shape=(n_clauses, n_classes), dtype=np.int16
    clauses: np.ndarray = tm._state.c  # shape=(n_clauses, n_literals*2), dtype=np.int8
    clauses_reordered = clauses.reshape(n_clauses, 2, n_literals).transpose(0, 2, 1).reshape(n_clauses, -1)

    print("threshold", threshold,
          "n_literals", n_literals,
          "n_clauses", n_clauses,
          "n_classes", n_classes,
          "max_state", max_state,
          "min_state", min_state,
          "boost_true_positive_feedback", boost_true_positive_feedback,
          "s", s,
          "weights", weights.shape, weights,
          "clauses (reordered)", clauses_reordered.shape, clauses_reordered,
          sep='\n')

    with open(filename, "wb") as f:
        # Write metadata
        f.write(threshold.to_bytes(4, "little", signed=False))
        f.write(n_literals.to_bytes(4, "little", signed=False))
        f.write(n_clauses.to_bytes(4, "little", signed=False))
        f.write(n_classes.to_bytes(4, "little", signed=False))
        f.write(max_state.to_bytes(1, "little", signed=True))
        f.write(min_state.to_bytes(1, "little", signed=True))
        f.write(boost_true_positive_feedback.to_bytes(1, "little", signed=False))
        f.write(struct.pack('<d', s))

        # Write weights and clauses
        f.write(weights.astype(np.int16).tobytes())
        f.write(clauses_reordered.astype(np.int8).tobytes())

def save_to_fbs(tm: 'gt.TsetlinMachine', filename: str):
    threshold: int = tm.threshold
    n_literals: int = tm.n_literals
    n_clauses: int = tm.n_clauses
    n_classes: int = tm.n_classes
    max_state: int = 127    # hardcoded in green_tsetlin/src/func_tm.hpp
    min_state: int = -127   # hardcoded in green_tsetlin/src/func_tm.hpp
    boost_true_positive_feedback: int = int(tm.boost_true_positives)
    s: float = tm.s[0]

    weights: np.ndarray = tm._state.w  # shape=(n_clauses, n_classes), dtype=np.int16
    clauses: np.ndarray = tm._state.c  # shape=(n_clauses, n_literals*2), dtype=np.int8
    clauses_reordered = clauses.reshape(n_clauses, 2, n_literals).transpose(0, 2, 1).reshape(n_clauses, -1)

    print("threshold", threshold,
          "n_literals", n_literals,
          "n_clauses", n_clauses,
          "n_classes", n_classes,
          "max_state", max_state,
          "min_state", min_state,
          "boost_true_positive_feedback", boost_true_positive_feedback,
          "s", s,
          "weights", weights.shape, weights,
          "clauses (reordered)", clauses_reordered.shape, clauses_reordered,
          sep='\n')
    
    builder = flatbuffers.Builder()

    TMParameters.Start(builder)
    TMParameters.AddThreshold(builder, threshold)
    TMParameters.AddNumLiterals(builder, n_literals)
    TMParameters.AddNumClauses(builder, n_clauses)
    TMParameters.AddNumClasses(builder, n_classes)
    TMParameters.AddMaxState(builder, max_state)
    TMParameters.AddMinState(builder, min_state)
    TMParameters.AddBoostTp(builder, boost_true_positive_feedback)
    TMParameters.AddS(builder, s)
    tm_params = TMParameters.End(builder)
    

    weights_vector = builder.CreateNumpyVector(weights.flatten().astype(np.int16))
    weights_shape_vector = builder.CreateNumpyVector(np.array(weights.shape, dtype=np.uint32))
    ClauseWeightsTensor.Start(builder)
    ClauseWeightsTensor.AddWeights(builder, weights_vector)
    ClauseWeightsTensor.AddShape(builder, weights_shape_vector)
    clause_weights = ClauseWeightsTensor.End(builder)

    states_vector = builder.CreateNumpyVector(clauses_reordered.flatten().astype(np.int8))
    states_shape_vector = builder.CreateNumpyVector(np.array(clauses_reordered.shape, dtype=np.uint32))
    AutomatonStatesTensor.Start(builder)
    AutomatonStatesTensor.AddStates(builder, states_vector)
    AutomatonStatesTensor.AddShape(builder, states_shape_vector)
    automaton_states = AutomatonStatesTensor.End(builder)

    # TODO: add literal_names
    TsetlinMachineModel.Start(builder)
    TsetlinMachineModel.AddParams(builder, tm_params)
    TsetlinMachineModel.AddAutomatonStates(builder, automaton_states)
    TsetlinMachineModel.AddClauseWeights(builder, clause_weights)
    tm_model = TsetlinMachineModel.End(builder)
    builder.Finish(tm_model)
    
    buf = builder.Output()

    print("buf", len(buf))  # TODO: remove