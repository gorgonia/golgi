package golgi

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LSTM represents an LSTM RNN
type LSTM struct {
	name string

	g *G.ExprGraph

	input  lstmGate
	forget lstmGate
	output lstmGate
	cell   lstmGate

	size        int // for construction
	initialized bool
	dummyCell   *G.Node
	dummyHidden *G.Node
}

// FromLSTMData will initialize a new LSTM model
func FromLSTMData(g *G.ExprGraph, layer *LSTMData, name string) *LSTM {
	retVal, err := layer.Make(g, name)
	if err != nil {
		panic(err)
	}
	return retVal.(*LSTM)

}

// ConsLSTM is a LSTM construction function. It takes a gorgonia.Input that has a *gorgonia.Node.
func ConsLSTM(in G.Input, opts ...ConsOpt) (retVal Layer, err error) {
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
			return nil, err
		}

		if l, ok = o.(*LSTM); !ok {
			err = errors.Errorf("Construction Option returned a non LSTM. Got %T instead", o)
			return nil, err
		}
	}

	if err = l.Init(x); err != nil {
		return
	}

	retVal = l

	return retVal, nil
}

// Model will return the gorgonia.Nodes associated with this LSTM
func (l *LSTM) Model() G.Nodes {
	return G.Nodes{
		l.input.wx, l.input.wh, l.input.b,
		l.forget.wx, l.forget.wh, l.forget.b,
		l.output.wx, l.output.wh, l.output.b,
		l.cell.wx, l.cell.wh, l.cell.b,
	}
}

// Fwd runs the equation forwards.
//
// While a *LSTM can take any gorgonia.Input as an input, it returns a gorgonia.Result,
// of which the concrete type is a lstimIO.
//
// The lstmIO type is not exported. Instead, to query the *Node of the gorgonia.Input or gorgonia.Result,
// use the Nodes() method.
//
// The Result will always be organized as such: [previousHidden, previousCell]
//
// e.g.
// 	out := lstm.Fwd(x)
// 	outNodes := out.Nodes()
//	prevHidden := outNodes[0]
//	prevCell := outNodes[1]
func (l *LSTM) Fwd(x G.Input) G.Result {
	var (
		inputVector *G.Node
		prevHidden  *G.Node
		prevCell    *G.Node

		err error
	)

	if err = G.CheckOne(x); err != nil {
		return G.Err(err)
	}

	ns := x.Nodes()
	switch len(ns) {
	case 0:
		err = errors.New("input value does not contain any nodes")
		return G.Err(err)
	case 1:
		inputVector = ns[0]
		prevHidden = l.dummyHidden
		prevCell = l.dummyCell
	case 2:
		err = errors.Errorf("invalid number of nodes, expected %d and received %d", 3, 2)
		return G.Err(err)
	case 3:
		inputVector = ns[0]
		prevHidden = ns[1]
		prevCell = ns[2]
	}

	var inputGate *G.Node
	if inputGate, err = l.input.activate(inputVector, prevHidden); err != nil {
		return G.Err(err)
	}

	var forgetGate *G.Node
	if forgetGate, err = l.forget.activate(inputVector, prevHidden); err != nil {
		return G.Err(err)
	}

	var outputGate *G.Node
	if outputGate, err = l.output.activate(inputVector, prevHidden); err != nil {
		return G.Err(err)
	}

	var cellWrite *G.Node
	if cellWrite, err = l.cell.activate(inputVector, prevHidden); err != nil {
		return G.Err(err)
	}

	// Perform cell activations
	var retain *G.Node
	if retain, err = BroadcastHadamardProd(forgetGate, prevCell, nil, []byte{0}); err != nil {
		return G.Err(err)
	}

	var write *G.Node
	if write, err = BroadcastHadamardProd(inputGate, cellWrite, nil, []byte{0}); err != nil {
		return G.Err(err)
	}

	var cell *G.Node
	if cell, err = G.Add(retain, write); err != nil {
		return G.Err(err)
	}

	var tahnCell *G.Node
	if tahnCell, err = G.Tanh(cell); err != nil {
		return G.Err(err)
	}

	var hidden *G.Node
	if hidden, err = BroadcastHadamardProd(outputGate, tahnCell, nil, []byte{0}); err != nil {
		return G.Err(err)
	}

	result := makeLSTMIO(inputVector, hidden, cell, nil)
	return &result
}

// Type will return the hm.Type of the LSTM
func (l *LSTM) Type() hm.Type { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b')) }

// Shape will return the tensor.Shape of the LSTM
func (l *LSTM) Shape() tensor.Shape { return l.input.b.Shape() }

// Name will return the name of the LSTM
func (l *LSTM) Name() string { return l.name }

// Describe will describe a LSTM
func (l *LSTM) Describe() { panic("not implemented") }

// SetName will set the name of a fully connected layer
func (l *LSTM) SetName(a string) error {
	l.name = a
	return nil
}

// Init will initialize the fully connected layer
func (l *LSTM) Init(xs ...*G.Node) (err error) {
	if len(xs) != 1 {
		return errors.Errorf("Tried to initialize an LSTM with %d input nodes. Expected 1 only.", len(xs))
	}
	x := xs[0]
	g := x.Graph()
	of := x.Dtype()
	X := x
	inner := X.Shape()[1]

	// initialize input gate
	l.input.init(g, of, inner, l.size, l.name+"_i", G.Sigmoid)
	l.forget.init(g, of, inner, l.size, l.name+"_f", G.Sigmoid)
	l.output.init(g, of, inner, l.size, l.name+"_o", G.Sigmoid)
	l.cell.init(g, of, inner, l.size, l.name+"_c", G.Tanh)

	// initialize dummyPrev and dummyCell
	l.dummyHidden = G.NewMatrix(g, of, G.WithShape(1, l.size), G.WithName(l.name+"dummyHidden"), G.WithInit(G.Zeroes()))
	l.dummyCell = G.NewMatrix(g, of, G.WithShape(1, l.size), G.WithName(l.name+"dummySize"), G.WithInit(G.Zeroes()))
	l.initialized = true
	return nil
}
