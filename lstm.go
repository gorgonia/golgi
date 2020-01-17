package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func newLSTM(g *gorgonia.ExprGraph, layer *LSTMLayer, name string) (lp *LSTM) {
	var l LSTM
	l.inputGateWeight = gorgonia.NodeFromAny(g, layer.inputGateWeight, gorgonia.WithName("wix_"+name))
	l.inputGateHiddenWeight = gorgonia.NodeFromAny(g, layer.inputGateHiddenWeight, gorgonia.WithName("wih_"+name))
	l.inputBias = gorgonia.NodeFromAny(g, layer.inputBias, gorgonia.WithName("bias_i_"+name))

	l.forgetGateWeight = gorgonia.NodeFromAny(g, layer.forgetGateWeight, gorgonia.WithName("wfx_"+name))
	l.forgetGateHiddenWeight = gorgonia.NodeFromAny(g, layer.forgetGateHiddenWeight, gorgonia.WithName("wfh_"+name))
	l.forgetBias = gorgonia.NodeFromAny(g, layer.forgetBias, gorgonia.WithName("bias_f_"+name))

	l.outputGateWeight = gorgonia.NodeFromAny(g, layer.outputGateWeight, gorgonia.WithName("wox_"+name))
	l.outputGateHiddenWeight = gorgonia.NodeFromAny(g, layer.outputGateHiddenWeight, gorgonia.WithName("woh_"+name))
	l.outputBias = gorgonia.NodeFromAny(g, layer.outputBias, gorgonia.WithName("bias_o_"+name))

	l.cellGateWeight = gorgonia.NodeFromAny(g, layer.cellGateWeight, gorgonia.WithName("wcx_"+name))
	l.cellGateHiddenWeight = gorgonia.NodeFromAny(g, layer.cellGateHiddenWeight, gorgonia.WithName("wch_"+name))
	l.cellBias = gorgonia.NodeFromAny(g, layer.cellBias, gorgonia.WithName("bias_c_"+name))
	return &l
}

// LSTM represents an LSTM RNN
type LSTM struct {
	name string

	inputGateWeight       *gorgonia.Node
	inputGateHiddenWeight *gorgonia.Node
	inputBias             *gorgonia.Node

	forgetGateWeight       *gorgonia.Node
	forgetGateHiddenWeight *gorgonia.Node
	forgetBias             *gorgonia.Node

	outputGateWeight       *gorgonia.Node
	outputGateHiddenWeight *gorgonia.Node
	outputBias             *gorgonia.Node

	cellGateWeight       *gorgonia.Node
	cellGateHiddenWeight *gorgonia.Node
	cellBias             *gorgonia.Node
}

// Model will return the gorgonia.Nodes associated with this LSTM
func (l *LSTM) Model() gorgonia.Nodes {
	return gorgonia.Nodes{l.inputBias, l.forgetBias, l.outputBias, l.cellBias}
}

// Fwd runs the equation forwards
// TODO: Convert this to a proper Fwd, this is still a crude copy of charRNN
func (l *LSTM) Fwd(x gorgonia.Input) gorgonia.Result {
	var inputVector, prevHidden, prevCell *gorgonia.Node
	var h0, h1, inputGate *gorgonia.Node
	h0 = gorgonia.Must(gorgonia.Mul(l.inputGateWeight, inputVector))
	h1 = gorgonia.Must(gorgonia.Mul(l.inputGateHiddenWeight, prevHidden))
	inputGate = gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h0, h1)), l.inputBias))))

	var h2, h3, forgetGate *gorgonia.Node
	h2 = gorgonia.Must(gorgonia.Mul(l.forgetGateWeight, inputVector))
	h3 = gorgonia.Must(gorgonia.Mul(l.forgetGateHiddenWeight, prevHidden))
	forgetGate = gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h2, h3)), l.forgetBias))))

	var h4, h5, outputGate *gorgonia.Node
	h4 = gorgonia.Must(gorgonia.Mul(l.outputGateWeight, inputVector))
	h5 = gorgonia.Must(gorgonia.Mul(l.outputGateHiddenWeight, prevHidden))
	outputGate = gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h4, h5)), l.outputBias))))

	var h6, h7, cellWrite *gorgonia.Node
	h6 = gorgonia.Must(gorgonia.Mul(l.cellGateWeight, inputVector))
	h7 = gorgonia.Must(gorgonia.Mul(l.cellGateHiddenWeight, prevHidden))
	cellWrite = gorgonia.Must(gorgonia.Tanh(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h6, h7)), l.cellBias))))

	// cell activations
	var retain, write *gorgonia.Node
	retain = gorgonia.Must(gorgonia.HadamardProd(forgetGate, prevCell))
	write = gorgonia.Must(gorgonia.HadamardProd(inputGate, cellWrite))
	cell := gorgonia.Must(gorgonia.Add(retain, write))
	hidden := gorgonia.Must(gorgonia.HadamardProd(outputGate, gorgonia.Must(gorgonia.Tanh(cell))))
	fmt.Println("Hidden", hidden)
	return nil
}

// Type will return the hm.Type of the LSTM
func (l *LSTM) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Shape will return the tensor.Shape of the LSTM
func (l *LSTM) Shape() tensor.Shape {
	return l.inputBias.Shape()
}

// Name will return the name of the LSTM
func (l *LSTM) Name() string {
	return l.Name()
}

// Describe will describe a LSTM
func (l *LSTM) Describe() {
	panic("not implemented")
}

// SetName will set the name of a fully connected layer
func (l *LSTM) SetName(a string) error {
	l.name = a
	return nil
}

// LSTMLayer represents a basic LSTM layer
type LSTMLayer struct {
	inputGateWeight       gorgonia.Value
	inputGateHiddenWeight gorgonia.Value
	inputBias             gorgonia.Value

	forgetGateWeight       gorgonia.Value
	forgetGateHiddenWeight gorgonia.Value
	forgetBias             gorgonia.Value

	outputGateWeight       gorgonia.Value
	outputGateHiddenWeight gorgonia.Value
	outputBias             gorgonia.Value

	cellGateWeight       gorgonia.Value
	cellGateHiddenWeight gorgonia.Value
	cellBias             gorgonia.Value
}
