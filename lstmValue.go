package golgi

import "gorgonia.org/gorgonia"

// Ensure that lstmInput matches both gorgonia.Input and gorgonia.Result interfaces
var (
	_ gorgonia.Input  = &lstmValue{}
	_ gorgonia.Result = &lstmValue{}
)

// makeLSTMInput will return a new lstmValue
func makeLSTMInput(x, prevHidden, prevCell *gorgonia.Node, err error) (l lstmValue) {
	l.x = x
	l.prevHidden = prevHidden
	l.prevCell = prevCell
	l.err = err
	return
}

// lstmValue represents an LSTM input type
type lstmValue struct {
	x          *gorgonia.Node
	prevHidden *gorgonia.Node
	prevCell   *gorgonia.Node

	err error
}

// Node will return the node associated with the LSTM input
func (l *lstmValue) Node() *gorgonia.Node {
	return nil
}

// Nodes will return the nodes associated with the LSTM input
func (l *lstmValue) Nodes() (ns gorgonia.Nodes) {
	if l.err != nil {
		return
	}

	return gorgonia.Nodes{l.x, l.prevHidden, l.prevCell}
}

// Err will return any error associated with the LSTM input
func (l *lstmValue) Err() error {
	return l.err
}
