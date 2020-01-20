package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func newLSTM(g *gorgonia.ExprGraph, layer *LSTMLayer, name string) (lp *LSTM) {
	var l LSTM
	l.g = g
	l.name = name
	l.input = layer.makeGate(g, name)
	l.forget = layer.makeGate(g, name)
	l.output = layer.makeGate(g, name)
	l.cell = layer.makeGate(g, name)
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

func (l *LSTM) activateGate(w *whb, inputVector, prevHidden *gorgonia.Node, act ActivationFunction) (gate *gorgonia.Node, err error) {
	var h0 *gorgonia.Node
	if h0, err = gorgonia.Mul(w.wx, inputVector); err != nil {
		return
	}

	var h1 *gorgonia.Node
	if h1, err = gorgonia.Mul(w.wh, prevHidden); err != nil {
		return
	}

	// Set gate as the sum of h0 and h1
	if gate, err = gorgonia.Add(h0, h1); err != nil {
		return
	}

	// Set the gate as the sum of current gate and the whb bias
	if gate, err = gorgonia.Add(gate, w.b); err != nil {
		return
	}

	// Return gate with activation func performed on it
	return act(gate)
}

// Model will return the gorgonia.Nodes associated with this LSTM
func (l *LSTM) Model() gorgonia.Nodes {
	return gorgonia.Nodes{
		l.input.wx, l.input.wh, l.input.b,
		l.forget.wx, l.forget.wh, l.forget.b,
		l.output.wx, l.output.wh, l.output.b,
		l.cell.wx, l.cell.wh, l.cell.b,
	}
}

// Fwd runs the equation forwards
func (l *LSTM) Fwd(x gorgonia.Input) gorgonia.Result {
	var (
		inputGate *gorgonia.Node
		err       error

		inputVector *gorgonia.Node
		prevHidden  *gorgonia.Node
		prevCell    *gorgonia.Node
	)

	if err = gorgonia.CheckOne(x); err != nil {
		return gorgonia.Err(err)
	}

	ns := x.Nodes()
	switch len(ns) {
	case 0:
		err = errors.New("input value does not contain any nodes")
		return gorgonia.Err(err)
	case 1:
		inputVector = ns[0]
	case 2:
		err = fmt.Errorf("invalid number of nodes, expected %d and received %d", 3, 2)
		return gorgonia.Err(err)
	case 3:
		inputVector = ns[0]
		prevHidden = ns[1]
		prevCell = ns[2]
	}

	if inputGate, err = l.activateGate(&l.input, inputVector, prevHidden, gorgonia.Sigmoid); err != nil {
		return gorgonia.Err(err)
	}

	var forgetGate *gorgonia.Node
	if forgetGate, err = l.activateGate(&l.forget, inputVector, prevHidden, gorgonia.Sigmoid); err != nil {
		return gorgonia.Err(err)
	}

	var outputGate *gorgonia.Node
	if outputGate, err = l.activateGate(&l.output, inputVector, prevHidden, gorgonia.Sigmoid); err != nil {
		return gorgonia.Err(err)
	}

	var cellWrite *gorgonia.Node
	if cellWrite, err = l.activateGate(&l.cell, inputVector, prevHidden, gorgonia.Tanh); err != nil {
		return gorgonia.Err(err)
	}

	// Perform cell activations

	var retain *gorgonia.Node
	if retain, err = gorgonia.HadamardProd(forgetGate, prevCell); err != nil {
		return gorgonia.Err(err)
	}

	var write *gorgonia.Node
	if write, err = gorgonia.HadamardProd(inputGate, cellWrite); err != nil {
		return gorgonia.Err(err)
	}

	var cell *gorgonia.Node
	if cell, err = gorgonia.Add(retain, write); err != nil {
		return gorgonia.Err(err)
	}

	var tahnCell *gorgonia.Node
	if tahnCell, err = gorgonia.Tanh(cell); err != nil {
		return gorgonia.Err(err)
	}

	var hidden *gorgonia.Node
	if hidden, err = gorgonia.HadamardProd(outputGate, tahnCell); err != nil {
		return gorgonia.Err(err)
	}

	result := makeLSTMValue(inputVector, hidden, cell, nil)
	return &result
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
	return l.name
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
