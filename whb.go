package golgi

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func makeWHB(wx, wh, b *gorgonia.Node) (out whb) {
	out.wx = wx
	out.wh = wh
	out.b = b
	return
}

type whb struct {
	wx *gorgonia.Node
	wh *gorgonia.Node
	b  *gorgonia.Node

	act ActivationFunction
}

func (w *whb) init(g *gorgonia.ExprGraph, of tensor.Dtype, inner, size int, name string, act ActivationFunction) {
	w.wh = gorgonia.NewMatrix(g, of, gorgonia.WithShape(size, size), gorgonia.WithName(name+"_wh"), gorgonia.WithInit(gorgonia.Gaussian(0, 0.08)))
	w.wx = gorgonia.NewMatrix(g, of, gorgonia.WithShape(inner, size), gorgonia.WithName(name+"_wx"), gorgonia.WithInit(gorgonia.Gaussian(0, 0.08)))
	w.b = gorgonia.NewMatrix(g, of, gorgonia.WithShape(1, size), gorgonia.WithName(name+"_b"), gorgonia.WithInit(gorgonia.Zeroes()))
	w.act = act
}

func (w *whb) activateGate(inputVector, prevHidden *gorgonia.Node) (gate *gorgonia.Node, err error) {
	// brief explanation
	// shapes
	// -------
	// x  : (n, inputSize)
	// h  : (1, hiddenSize)
	// Wx : (inputSize, hiddenSize)
	// Wh : (hiddenSize, hiddenSize)
	// b  : (1, hiddenSize)
	//
	// h0 = xWx = (n, inputSize) × (inputSize, hiddenSize) = (n, hiddenSize)
	// h1 = hWh = (1, hiddenSize) × (hiddenSize, hiddenSize) = (1, hiddenSize)
	// OBSERVATION: h0 and h1 cannot be added  together, unless n is 1. A broadcast operation is required.
	// gate = h0 +̅ h1 = (n, hiddenSize) +̅ (1, hiddenSize) = (n, hiddenSize)
	// ditto for adding the biases.

	var h0 *gorgonia.Node
	if h0, err = gorgonia.Mul(inputVector, w.wx); err != nil {
		return
	}

	var h1 *gorgonia.Node
	if h1, err = gorgonia.Mul(prevHidden, w.wh); err != nil {
		return
	}

	// Set gate as the sum of h0 and h1
	if gate, err = gorgonia.BroadcastAdd(h0, h1, nil, []byte{0}); err != nil {
		return
	}

	// Set the gate as the sum of current gate and the whb bias
	if gate, err = gorgonia.BroadcastAdd(gate, w.b, nil, []byte{0}); err != nil {
		return
	}

	// Return gate with activation func performed on it
	return w.act(gate)
}
