package golgi

import (
	G "gorgonia.org/gorgonia"
)

// LSTMData represents a basic LSTM layer
type LSTMData struct {
	inputGateWeight       G.Value
	inputGateHiddenWeight G.Value
	inputBias             G.Value

	forgetGateWeight       G.Value
	forgetGateHiddenWeight G.Value
	forgetBias             G.Value

	outputGateWeight       G.Value
	outputGateHiddenWeight G.Value
	outputBias             G.Value

	cellGateWeight       G.Value
	cellGateHiddenWeight G.Value
	cellBias             G.Value
}

func (l *LSTMData) makeGate(g *G.ExprGraph, name string) lstmGate {
	return makeLSTMGate(
		G.NodeFromAny(g, l.inputGateWeight, G.WithName("wx"+name)),
		G.NodeFromAny(g, l.inputGateHiddenWeight, G.WithName("wh_"+name)),
		G.NodeFromAny(g, l.inputBias, G.WithName("b_"+name)),
	)
}

func (l *LSTMData) Make(g *G.ExprGraph, name string) (Layer, error) {
	var retVal LSTM
	retVal.g = g
	retVal.name = name
	retVal.input = l.makeGate(g, "_input_"+name)
	retVal.forget = l.makeGate(g, "_forget_"+name)
	retVal.output = l.makeGate(g, "_output_"+name)
	retVal.cell = l.makeGate(g, "_cell_"+name)
	return &retVal, nil
}
