import flatbuffers
import green_tsetlin as gt
import numpy as np
import TsetlinMachine.AutomatonStatesTensor as AutomatonStatesTensor
import TsetlinMachine.ClauseWeightsTensor as ClauseWeightsTensor
import TsetlinMachine.Model as Model
import TsetlinMachine.Parameters as Parameters


def get_literal_names(tm: "gt.TsetlinMachine") -> list[str]:
    # TODO: Implement a way to retrieve literal names from the TsetlinMachine.
    return [f"Literal {i}" for i in range(tm.n_literals)]


def save_to_fbs(
    tm: "gt.TsetlinMachine", filename: str, save_literal_names: bool = False
):
    threshold: int = tm.threshold
    n_literals: int = tm.n_literals
    n_clauses: int = tm.n_clauses
    n_classes: int = tm.n_classes
    max_state: int = 127  # hardcoded in green_tsetlin/src/func_tm.hpp
    min_state: int = -127  # hardcoded in green_tsetlin/src/func_tm.hpp
    boost_tp: int = int(tm.boost_true_positives)
    learn_s: float = tm.s[0]  # type: ignore

    weights = np.astype(tm._state.w, np.int16)
    weights_shape = np.array(weights.shape, dtype=np.int32)

    clauses = np.astype(tm._state.c, np.int8)
    clauses_reshaped = clauses.reshape(n_clauses, n_literals, 2)
    clauses_reshaped_shape = np.array(clauses_reshaped.shape, dtype=np.int32)

    if not (literal_names := get_literal_names(tm)):
        save_literal_names = False
        print("No literal names found in green_tsetlin Model.")

    # create FlatBuffer Model
    builder = flatbuffers.Builder()

    Parameters.Start(builder)
    Parameters.AddThreshold(builder, threshold)
    Parameters.AddNLiterals(builder, n_literals)
    Parameters.AddNClauses(builder, n_clauses)
    Parameters.AddNClasses(builder, n_classes)
    Parameters.AddMaxState(builder, max_state)
    Parameters.AddMinState(builder, min_state)
    Parameters.AddBoostTp(builder, boost_tp)
    Parameters.AddLearnS(builder, learn_s)
    tm_params = Parameters.End(builder)

    weights_vec = builder.CreateNumpyVector(weights.flatten())
    weights_shape_vec = builder.CreateNumpyVector(weights_shape)
    ClauseWeightsTensor.Start(builder)
    ClauseWeightsTensor.AddWeights(builder, weights_vec)
    ClauseWeightsTensor.AddShape(builder, weights_shape_vec)
    clause_weights = ClauseWeightsTensor.End(builder)

    states_vec = builder.CreateNumpyVector(clauses_reshaped.flatten())
    states_shape_vec = builder.CreateNumpyVector(clauses_reshaped_shape)
    AutomatonStatesTensor.Start(builder)
    AutomatonStatesTensor.AddStates(builder, states_vec)
    AutomatonStatesTensor.AddShape(builder, states_shape_vec)
    automaton_states = AutomatonStatesTensor.End(builder)

    if save_literal_names:
        literal_name_offsets = [builder.CreateString(name) for name in literal_names]
        Model.StartLiteralNamesVector(builder, n_literals)
        for literal_name_offset in reversed(literal_name_offsets):
            builder.PrependUOffsetTRelative(literal_name_offset)
        literal_names_vec = builder.EndVector()

    Model.Start(builder)
    Model.AddParams(builder, tm_params)
    Model.AddAutomatonStates(builder, automaton_states)
    Model.AddClauseWeights(builder, clause_weights)
    if save_literal_names:
        Model.AddLiteralNames(builder, literal_names_vec)
    tm_model = Model.End(builder)
    builder.Finish(tm_model)

    buf = builder.Output()
    with open(filename, "wb") as f:
        f.write(buf)

    # print complete model by deserializing the buffer
    print("\n=== Complete FlatBuffer Model ===")
    model = Model.Model.GetRootAs(buf)

    params = model.Params()
    if params is not None:
        print("Parameters:")
        print(f"  Threshold: {params.Threshold()}")
        print(f"  N Literals: {params.NLiterals()}")
        print(f"  N Clauses: {params.NClauses()}")
        print(f"  N Classes: {params.NClasses()}")
        print(f"  Max State: {params.MaxState()}")
        print(f"  Min State: {params.MinState()}")
        print(f"  Boost TP: {params.BoostTp()}")
        print(f"  Learn S: {params.LearnS()}")
    else:
        print("Parameters: None (not found in model)")

    automaton_states = model.AutomatonStates()
    if automaton_states is not None:
        states_data = automaton_states.StatesAsNumpy()
        states_shape = automaton_states.ShapeAsNumpy()
        print("\nAutomaton States:")
        print(f"  Shape: {states_shape}")
        print(f"  Data: {states_data}")
    else:
        print("\nAutomaton States: None (not found in model)")

    clause_weights = model.ClauseWeights()
    if clause_weights is not None:
        weights_data = clause_weights.WeightsAsNumpy()
        weights_shape = clause_weights.ShapeAsNumpy()
        print("\nClause Weights:")
        print(f"  Shape: {weights_shape}")
        print(f"  Data: {weights_data}")
    else:
        print("\nClause Weights: None (not found in model)")

    literal_names = [model.LiteralNames(j) for j in range(model.LiteralNamesLength())]
    if literal_names:
        print("\nLiteral Names:")
        print(f"  Data: {literal_names}")
    else:
        print("\nLiteral Names: None (not found in model)")

    print(f"\nTotal buffer size: {len(buf)} bytes")
