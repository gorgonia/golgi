package golgi

import "gorgonia.org/gorgonia"

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
}

func (w *whb) activateGate(inputVector, prevHidden *gorgonia.Node, act ActivationFunction) (gate *gorgonia.Node, err error) {
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
