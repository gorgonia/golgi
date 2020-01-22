package golgi

import "gorgonia.org/gorgonia"

// Ensure that lstmInput matches both gorgonia.Input and gorgonia.Result interfaces
var (
	_ gorgonia.Input  = &lstmValue{}
	_ gorgonia.Result = &lstmValue{}
)

// makeLSTMValue will return a new lstmValue
func makeLSTMValue(x, prevHidden, prevCell *gorgonia.Node, err error) (l lstmValue) {
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

// Mk makes a new Input, given the xs. This is useful for replacing values in the tuple
//
// CAVEAT: the replacements depends on the length of xs
// 	1: replace x
//	3: replace x, prevCell, prevHidden in this order
//	other: no replacement. l is returned
func (l *lstmValue) Mk(xs ...gorgonia.Input) gorgonia.Input {
	switch len(xs) {
	case 0:
		return l
	case 1:
		return &lstmValue{x: xs[0].Node(), prevCell: l.prevCell, prevHidden: l.prevHidden}
	case 2:
		return l
	case 3:
		return &lstmValue{x: xs[0].Node(), prevCell: xs[1].Node(), prevHidden: xs[2].Node()}
	default:
		return l
	}
}
