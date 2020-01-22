package golgi

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LSTM represents an LSTM RNN
type LSTM struct {
	name string

	g *gorgonia.ExprGraph

	input  whb
	forget whb
	output whb
	cell   whb

	size        int // for construction
	initialized bool
	dummyCell   *gorgonia.Node
	dummyHidden *gorgonia.Node
}

// FromLSTMLayer will initialize a new LSTM model
func FromLSTMLayer(g *gorgonia.ExprGraph, layer *LSTMLayer, name string) (lp *LSTM) {
	var l LSTM
	l.g = g
	l.name = name
	l.input = layer.makeGate(g, name)
	l.forget = layer.makeGate(g, name)
	l.output = layer.makeGate(g, name)
	l.cell = layer.makeGate(g, name)
	return &l
}

// ConsLSTM is a LSTM construction function. It takes a gorgonia.Input that has a *gorgonia.Node.
func ConsLSTM(in gorgonia.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()
	if x == nil {
		return nil, errors.Errorf("LSTM expects a *Node. Got input %v of  %T instead", in, in)
	}

	// TODO: Ensure shape is being set correctly
	inshape := x.Shape()
	if inshape.Dims() > 2 || inshape.Dims() == 0 {
		return nil, errors.Errorf("Expected shape is either a vector or a matrix")
	}

	l := &LSTM{}
	for _, opt := range opts {
		var (
			o  Layer
			ok bool
		)

		if o, err = opt(l); err != nil {
			return
		}

		if l, ok = o.(*LSTM); !ok {
			err = errors.Errorf("Construction Option returned a non LSTM. Got %T instead", o)
			return
		}
	}

	// TODO: Handle initialization

	retVal = l
	return
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
		inputVector *gorgonia.Node
		prevHidden  *gorgonia.Node
		prevCell    *gorgonia.Node

		err error
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
		prevHidden = l.dummyHidden
		prevCell = l.dummyCell
	case 2:
		err = errors.Errorf("invalid number of nodes, expected %d and received %d", 3, 2)
		return gorgonia.Err(err)
	case 3:
		inputVector = ns[0]
		prevHidden = ns[1]
		prevCell = ns[2]
	}

	var inputGate *gorgonia.Node
	if inputGate, err = l.input.activateGate(inputVector, prevHidden); err != nil {
		return gorgonia.Err(err)
	}

	var forgetGate *gorgonia.Node
	if forgetGate, err = l.forget.activateGate(inputVector, prevHidden); err != nil {
		return gorgonia.Err(err)
	}

	var outputGate *gorgonia.Node
	if outputGate, err = l.output.activateGate(inputVector, prevHidden); err != nil {
		return gorgonia.Err(err)
	}

	var cellWrite *gorgonia.Node
	if cellWrite, err = l.cell.activateGate(inputVector, prevHidden); err != nil {
		return gorgonia.Err(err)
	}

	// Perform cell activations

	// TODO : broadcast op these things
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

// Init will initialize the fully connected layer
func (l *LSTM) Init(xs ...*gorgonia.Node) (err error) {
	x := xs[0]
	g := x.Graph()
	of := x.Dtype()
	X := x
	inner := X.Shape()[0]

	// initialize input gate
	l.input.init(g, of, inner, l.size, l.name+"_i", gorgonia.Sigmoid)
	l.forget.init(g, of, inner, l.size, l.name+"_f", gorgonia.Sigmoid)
	l.output.init(g, of, inner, l.size, l.name+"_o", gorgonia.Sigmoid)
	l.cell.init(g, of, inner, l.size, l.name+"_c", gorgonia.Tanh)

	// initialize dummyPrev and dummyCell
	l.dummyHidden = gorgonia.NewMatrix(g, of, gorgonia.WithShape(1, l.size), gorgonia.WithName(l.name+"dummyHidden"), gorgonia.WithInit(gorgonia.Zeroes()))
	l.dummyCell = gorgonia.NewMatrix(g, of, gorgonia.WithShape(1, l.size), gorgonia.WithName(l.name+"dummySize"), gorgonia.WithInit(gorgonia.Zeroes()))
	l.initialized = true
	return nil
}
