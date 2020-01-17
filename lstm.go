package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func newLSTM(g *gorgonia.ExprGraph, layer *LSTMLayer, name string) (lp *LSTM) {
	var l LSTM
	l.g = g
	l.name = name
	l.input = layer.getInput(g, name)
	l.forget = layer.getInput(g, name)
	l.output = layer.getInput(g, name)
	l.cell = layer.getInput(g, name)
	return &l
}

// LSTM represents an LSTM RNN
type LSTM struct {
	name string

	g *gorgonia.ExprGraph

	input  whb
	forget whb
	output whb
	cell   whb
}

// Model will return the gorgonia.Nodes associated with this LSTM
func (l *LSTM) Model() gorgonia.Nodes {
	return gorgonia.Nodes{l.input.b, l.forget.b, l.output.b, l.cell.b}
}

// Fwd runs the equation forwards
// TODO: Convert this to a proper Fwd, this is still a crude copy of charRNN
func (l *LSTM) Fwd(x gorgonia.Input) gorgonia.Result {
	var inputVector, prevHidden, prevCell *gorgonia.Node
	var h0, h1, inputGate *gorgonia.Node
	h0 = gorgonia.Must(gorgonia.Mul(l.input.wx, inputVector))
	h1 = gorgonia.Must(gorgonia.Mul(l.input.wh, prevHidden))
	inputGate = gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h0, h1)), l.input.b))))

	var h2, h3, forgetGate *gorgonia.Node
	h2 = gorgonia.Must(gorgonia.Mul(l.forget.wx, inputVector))
	h3 = gorgonia.Must(gorgonia.Mul(l.forget.wh, prevHidden))
	forgetGate = gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h2, h3)), l.forget.b))))

	var h4, h5, outputGate *gorgonia.Node
	h4 = gorgonia.Must(gorgonia.Mul(l.output.wx, inputVector))
	h5 = gorgonia.Must(gorgonia.Mul(l.output.wh, prevHidden))
	outputGate = gorgonia.Must(gorgonia.Sigmoid(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h4, h5)), l.output.b))))

	var h6, h7, cellWrite *gorgonia.Node
	h6 = gorgonia.Must(gorgonia.Mul(l.cell.wx, inputVector))
	h7 = gorgonia.Must(gorgonia.Mul(l.cell.wh, prevHidden))
	cellWrite = gorgonia.Must(gorgonia.Tanh(gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Add(h6, h7)), l.cell.b))))

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
	return l.input.b.Shape()
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
