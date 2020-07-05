package golgi

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func makeLSTMGate(wx, wh, b *G.Node) (out lstmGate) {
	out.wx = wx
	out.wh = wh
	out.b = b
	return
}

type lstmGate struct {
	wx *G.Node
	wh *G.Node
	b  *G.Node

	act ActivationFunction
}

func (w *lstmGate) init(g *G.ExprGraph, of tensor.Dtype, inner, size int, name string, act ActivationFunction) {
	w.wh = G.NewMatrix(g, of, G.WithShape(size, size), G.WithName(name+"_wh"), G.WithInit(G.GlorotU(1)))
	w.wx = G.NewMatrix(g, of, G.WithShape(inner, size), G.WithName(name+"_wx"), G.WithInit(G.GlorotU(1)))
	w.b = G.NewMatrix(g, of, G.WithShape(1, size), G.WithName(name+"_b"), G.WithInit(G.Zeroes()))
	w.act = act
}

// activate activates the gate.
//
// some metainformation
// 	shapes
// 	-------
//  	x  : (n, inputSize)
//  	h  : (1, hiddenSize)
//  	Wx : (inputSize, hiddenSize)
//  	Wh : (hiddenSize, hiddenSize)
//  	b  : (1, hiddenSize)
//
// 	h0 = xWx = (n, inputSize) × (inputSize, hiddenSize) = (n, hiddenSize)
// 	h1 = hWh = (1, hiddenSize) × (hiddenSize, hiddenSize) = (1, hiddenSize)
//
// OBSERVATION: h0 and h1 cannot be added  together, unless n is 1. A broadcast operation is required.
// gate = h0 +̅ h1 = (n, hiddenSize) +̅ (1, hiddenSize) = (n, hiddenSize)
// ditto for adding the biases.
func (w *lstmGate) activate(inputVector, prevHidden *G.Node) (gate *G.Node, err error) {

	var h0 *G.Node
	if h0, err = G.Mul(inputVector, w.wx); err != nil {
		return
	}

	var h1 *G.Node
	if h1, err = G.Mul(prevHidden, w.wh); err != nil {
		return
	}

	// Set gate as the sum of h0 and h1
	if gate, err = BroadcastAdd(h0, h1, nil, []byte{0}); err != nil {
		return
	}

	// Set the gate as the sum of current gate and the whb bias
	if gate, err = BroadcastAdd(gate, w.b, nil, []byte{0}); err != nil {
		return
	}

	// Return gate with activation func performed on it
	return w.act(gate)
}
