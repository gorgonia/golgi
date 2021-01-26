package golgi

import G "gorgonia.org/gorgonia"

// Ensure that lstmInput matches both gorgonia.Input and gorgonia.Result interfaces
var (
	_ G.Input  = &lstmIO{}
	_ G.Result = &lstmIO{}
)

// makeLSTMIO will return a new lstmIO
func makeLSTMIO(x, prevHidden, prevCell *G.Node, err error) (l lstmIO) {
	l.x = x
	l.prevHidden = prevHidden
	l.prevCell = prevCell
	l.err = err

	return l
}

// lstmIO represents an LSTM input/output value
type lstmIO struct {
	x          *G.Node
	prevHidden *G.Node
	prevCell   *G.Node

	err error
}

// Node will return the node associated with the LSTM input
func (l *lstmIO) Node() *G.Node { return nil }

// Nodes will return the nodes associated with the LSTM input
func (l *lstmIO) Nodes() (ns G.Nodes) {
	if l.err != nil {
		return
	}

	return G.Nodes{l.x, l.prevHidden, l.prevCell}
}

// Err will return any error associated with the LSTM input
func (l *lstmIO) Err() error { return l.err }

// Mk makes a new Input, given the xs. This is useful for replacing values in the tuple
//
// CAVEAT: the replacements depends on the length of xs
// 	1: replace x
//	3: replace x, prevCell, prevHidden in this order
//	other: no replacement. l is returned
func (l *lstmIO) Mk(xs ...G.Input) G.Input {
	switch len(xs) {
	case 0:
		return l
	case 1:
		return &lstmIO{x: xs[0].Node(), prevCell: l.prevCell, prevHidden: l.prevHidden}
	case 2:
		return l
	case 3:
		return &lstmIO{x: xs[0].Node(), prevCell: xs[1].Node(), prevHidden: xs[2].Node()}
	default:
		return l
	}
}
